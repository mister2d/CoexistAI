import hashlib
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import chromadb
from chromadb.config import Settings

# Set up logger
logger = logging.getLogger(__name__)

# Global persistent ChromaDB client for better performance
_chroma_client = None
_chroma_persistent_path = "./chroma_db"

def get_chroma_client():
    """Get or create a persistent ChromaDB client with optimized settings."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=_chroma_persistent_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # Optional: Test connection
        try:
            _chroma_client.heartbeat()  # Verify client is working
        except Exception as e:
            logger.warning(f"ChromaDB client health check failed: {e}")
    
    return _chroma_client

async def create_vectorstore_async(docs, collection_name, hf_embeddings, top_k, ensemble_weights=[0.25, 0.75]):
    """
    Asynchronously creates a vectorstore from the given documents using Chroma and returns an ensemble retriever.
    Uses persistent ChromaDB client with optimized settings for better performance.
    Each subquery gets its own collection for query isolation.

    Args:
        docs (list): A list of documents to be added to the vectorstore.
        collection_name (str): The name of the collection to be used for the vectorstore.
        hf_embeddings (object): The embedding model to be used for the vectorstore.
        top_k (int): The number of documents to retrieve from the vectorstore.
        ensemble_weights (list): Weights for BM25 and semantic retrievers [bm25_weight, semantic_weight]

    Returns:
        EnsembleRetriever: An ensemble retriever that combines BM25 and semantic retrievers.
    """
    # Create unique collection name with timestamp to avoid conflicts
    timestamp = str(int(time.time() * 1000))  # millisecond precision
    unique_collection_name = f"{collection_name}_{timestamp}"
    
    # Use thread pool for CPU-intensive operations
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Run ChromaDB operations in thread pool
        ensemble_retriever = await loop.run_in_executor(
            executor,
            _create_vectorstore_sync,
            docs, unique_collection_name, hf_embeddings, top_k, ensemble_weights
        )
    
    return ensemble_retriever

def _create_vectorstore_sync(docs, unique_collection_name, hf_embeddings, top_k, ensemble_weights):
    """
    Synchronous helper function for creating vectorstore.
    This runs in a thread pool to avoid blocking the event loop.
    """
    try:
        # Use persistent client for better performance
        client = get_chroma_client()
        
        # Create vectorstore with default settings (more stable)
        # ChromaDB will handle collection creation automatically with sensible defaults
        
        # Create vectorstore using the collection
        vectorstore = Chroma(
            client=client,
            collection_name=unique_collection_name,
            embedding_function=hf_embeddings
        )
        
        # Add documents to vectorstore
        vectorstore.add_documents(docs)
        
        # Create retrievers
        sem_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = top_k
        
        # Create ensemble retriever with configurable weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, sem_retriever], 
            weights=ensemble_weights
        )
        
        logger.info(f"Created vectorstore with collection: {unique_collection_name}")
        return ensemble_retriever
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise

# Keep synchronous version for backward compatibility
def create_vectorstore(docs, collection_name, hf_embeddings, top_k, ensemble_weights=[0.25, 0.75]):
    """
    Synchronous version of create_vectorstore for backward compatibility.
    For better performance, use create_vectorstore_async() instead.
    """
    timestamp = str(int(time.time() * 1000))
    unique_collection_name = f"{collection_name}_{timestamp}"
    return _create_vectorstore_sync(docs, unique_collection_name, hf_embeddings, top_k, ensemble_weights)

async def cleanup_old_collections_async(max_collections=20):
    """
    Asynchronously clean up old ChromaDB collections to prevent memory buildup.
    Keeps only the most recent collections.
    
    Args:
        max_collections (int): Maximum number of collections to keep
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, _cleanup_collections_sync, max_collections)

def _cleanup_collections_sync(max_collections):
    """
    Synchronous helper function for cleaning up collections.
    This runs in a thread pool to avoid blocking the event loop.
    """
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        if len(collections) > max_collections:
            # Sort collections by name (which includes timestamp) and delete oldest
            sorted_collections = sorted(collections, key=lambda x: x.name)
            collections_to_delete = sorted_collections[:-max_collections]
            
            logger.info(f"Cleaning up {len(collections_to_delete)} old collections")
            
            for collection in collections_to_delete:
                try:
                    client.delete_collection(collection.name)
                    logger.info(f"Deleted old collection: {collection.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete collection {collection.name}: {e}")
        else:
            logger.info(f"Collection count ({len(collections)}) within limit ({max_collections})")
                    
    except Exception as e:
        logger.error(f"Error during collection cleanup: {e}")

# Keep synchronous version for backward compatibility
def cleanup_old_collections(max_collections=20):
    """
    Synchronous version of cleanup_old_collections for backward compatibility.
    For better performance, use cleanup_old_collections_async() instead.
    """
    _cleanup_collections_sync(max_collections)
