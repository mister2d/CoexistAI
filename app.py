from utils.websearch_utils import *
from utils.reddit_utils import *
from utils.map import *
from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils.utils import *
from utils.map import *
from utils.git_utils import *
from utils.startup_banner import display_startup_banner, display_shutdown_banner
import subprocess
from utils.tts_utils import *
from fastapi_mcp import FastApiMCP
import json
import os
import atexit
from model_config import *

# Startup configuration validation
def validate_environment_config():
    """Validate and log all environment variable configurations"""
    logger.info("=== Environment Configuration Validation ===")
    
    # SearxNG configuration
    searxng_host = os.environ.get('SEARXNG_HOST', 'not set')
    searxng_port = os.environ.get('SEARXNG_PORT', 'not set')
    searxng_protocol = os.environ.get('SEARXNG_PROTOCOL', 'not set')
    logger.info(f"SearxNG Config - Host: {searxng_host}, Port: {searxng_port}, Protocol: {searxng_protocol}")
    
    # Application configuration
    app_port = os.environ.get('PORT', 'not set')
    app_host = os.environ.get('HOST', 'not set')
    logger.info(f"App Config - Host: {app_host}, Port: {app_port}")
    
    # LLM configuration
    llm_type = os.environ.get('LLM_TYPE', 'not set')
    llm_model = os.environ.get('LLM_MODEL_NAME', 'not set')
    google_api_key = os.environ.get('GOOGLE_API_KEY', 'not set')
    openai_api_key = os.environ.get('OPENAI_API_KEY', 'not set')
    logger.info(f"LLM Config - Type: {llm_type}, Model: {llm_model}")
    logger.info(f"API Keys - Google: {'set' if google_api_key else 'not set'}, OpenAI: {'set' if openai_api_key else 'not set'}")
    
    # Embedding configuration
    embed_mode = os.environ.get('EMBED_MODE', 'not set')
    embedding_model = os.environ.get('EMBEDDING_MODEL_NAME', 'not set')
    logger.info(f"Embedding Config - Mode: {embed_mode}, Model: {embedding_model}")
    
    # Other configuration
    user_agent = os.environ.get('USER_AGENT', 'not set')
    place = os.environ.get('PLACE', 'not set')
    log_level = os.environ.get('LOG_LEVEL', 'not set')
    logger.info(f"Other Config - User Agent: {user_agent}, Place: {place}, Log Level: {log_level}")
    
    logger.info("=== Configuration Validation Complete ===")

# Run configuration validation at startup
validate_environment_config()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from uuid import uuid4
import requests


# Use config values for model and embedding paths
llm_model_name = model_config.get("llm_model_name", 'google/gemma-3-12b')
llm_type = model_config.get("llm_type", 'local')
llm_tools = model_config.get("llm_tools",None)
llm_base_url = openai_compatible.get(model_config['llm_type'], 
                                     'https://api.openai.com/v1')



llm_kwargs = model_config.get("llm_kwargs", {'temperature': 0.1, 
                                            'max_tokens': None, 
                                            'timeout': None, 
                                            'api_key':llm_api_key,
                                            'max_retries': 2})

embed_kwargs = model_config.get("embed_kwargs", {})
embedding_model_name = model_config.get("embedding_model_name", "models/embedding-001")
embed_mode = model_config.get("embed_mode", "gemini")
cross_encoder_name = model_config.get("cross_encoder_name", "BAAI/bge-reranker-base")


logger.info("Checking SearxNG service availability...")
# Use the new service health check function instead of Docker orchestration
from utils.utils import check_service_health

# Debug: Log environment variables for SearxNG
logger.info(f"Environment variables - SEARXNG_HOST: {os.environ.get('SEARXNG_HOST', 'not set')}")
logger.info(f"Environment variables - SEARXNG_PORT: {os.environ.get('SEARXNG_PORT', 'not set')}")
logger.info(f"Environment variables - SEARXNG_PROTOCOL: {os.environ.get('SEARXNG_PROTOCOL', 'not set')}")
logger.info(f"Hardcoded values - HOST_SEARXNG: {HOST_SEARXNG}, PORT_NUM_SEARXNG: {PORT_NUM_SEARXNG}")

# Use environment variables if available, otherwise fall back to hardcoded values
searxng_host = os.environ.get('SEARXNG_HOST', HOST_SEARXNG)
searxng_port = os.environ.get('SEARXNG_PORT', PORT_NUM_SEARXNG)
searxng_protocol = os.environ.get('SEARXNG_PROTOCOL', 'http')

searxng_health_url = f"{searxng_protocol}://{searxng_host}:{searxng_port}"
logger.info(f"Attempting to connect to SearxNG at: {searxng_health_url}")

if check_service_health(searxng_health_url, "SearxNG"):
    logger.info("SearxNG service is available and healthy")
else:
    logger.warning("SearxNG service is not available")
    logger.warning("Web search functionality will be disabled or limited")
    logger.info("This is expected when SearxNG is not running as a separate service")

llm = get_generative_model(
    model_name=llm_model_name,
    type=llm_type,
    base_url=llm_base_url,
    _tools=None,
    kwargs=llm_kwargs
)

hf_embeddings, cross_encoder = load_model(embedding_model_name, 
                                          _embed_mode=embed_mode,
                                          cross_encoder_name=cross_encoder_name,
                                          kwargs=embed_kwargs)

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)

searcher = SearchWeb(PORT_NUM_SEARXNG, HOST_SEARXNG)
date, day = get_local_data()
app = FastAPI(title='coexistai')

# Register shutdown handler
atexit.register(display_shutdown_banner)

origins = [
    "*",  # Allow all origins (use specific domains in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # e.g. ["http://localhost", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods (including OPTIONS)
    allow_headers=["*"],    # Allow all headers
)

@app.get('/')
async def root():
    return {"message": "Welcome to CoexistAI!"}

class WebSearchRequest(BaseModel):
    query: str
    rerank: bool = True
    num_results: int = 2
    local_mode: bool = False
    split: bool = True
    document_paths: list[str] = []  # List of paths for local documents

class YouTubeSearchRequest(BaseModel):
    query: str
    prompt: str
    n: int = 1  # Number of videos to summarize, default is 1

class RedditSearchRequest(BaseModel):
    subreddit: str = None
    url_type: str = "hot"
    n: int = 3
    k: int = 1
    custom_url: str = None
    time_filter: str = "all"
    search_query: str = None
    sort_type: str = "relevance"

class MapSearchRequest(BaseModel):
    start_location: Optional[str] = None  # Start location can be a string or None
    end_location: Optional[str] = None  # End location can be a string or None
    pois_radius: int = 500  # Default radius for POIs in meters
    amenities: str = "restaurant|cafe|bar|hotel"  # Default amenities to search for
    limit: int = 3  # Default number of results to return
    task: str = "route_and_pois"  # Default task is to find a route

class WebSummarizeRequest(BaseModel):
    query: str
    url: str
    local_mode: bool = False

class GitTreeRequest(BaseModel):
    repobaseurl: str  

class GitSearchRequest(BaseModel):
    repobaseurl: str 
    parttoresearch: str
    query: str
    type: str

class LocalFolderTreeRequest(BaseModel):
    folder_path:str
    level: str = 'broad-first'
    prefix: str = ''

class ResearchCheckRequest(BaseModel):
    query: str
    toolsshorthand: str  # Default budget for deep research, can be adjusted as needed

class ClickableElementRequest(BaseModel):
    url:str
    query:str
    topk:int=10

class PodcastRequest(BaseModel):
    text: str = None
    prompt: str = None  # Optional theme for the podcast

class BasicTTSRequest(BaseModel):
    text: str = None
    voice: str = "am_santa"
    lang: str = "en-us"
    filename: str = ""

@app.post('/clickable-elements', operation_id="get_website_structure")
async def get_website_structure(request: ClickableElementRequest):
    """
    Retrieves the top-k clickable elements from a given URL based on a query.
    This will help you to find out if there are any clickable elements on the page that match the query.
    You can use this to find deeper links since connected pieces of information are often linked together.
    RECOMMENDATION: Be specific with the query to get the most relevant clickable elements.
    Args:
        url (str): The URL to search for clickable elements.
        query (str): The query to filter the clickable elements.
        topk (int): The number of top clickable elements to return.
    Returns:
        list: A list of dictionaries containing the title, URL, and score of each clickable element.
    """
    return await get_topk_bm25_clickable_elements(request.url, request.query, request.topk)

@app.post('/local-folder-tree', operation_id="get_local_folder_tree")
async def get_local_folder_tree(request: LocalFolderTreeRequest):
    """
    Async Markdown folder tree.
    Args:
        folder_path (str): Root directory.
        level (str):
            - 'full': Show all folders and files, recursively, except hidden/system/cache entries.
            - 'broad-first': Only show immediate (top-level) folders and files (no nesting).
            - 'broad-second': Show top-level folders/files and their immediate child folders/files (two levels, no deeper).
        prefix (str): Indentation (internal)
    Returns:
        str: Markdown tree string
    """
    return await folder_tree(request.folder_path, level=request.level, prefix=request.prefix)

@app.post('/git-tree-search',operation_id="get_git_tree")
async def get_git_tree(request:GitTreeRequest):
   """
    Retrieves and returns the directory tree structure of a GitHub repository or a local Git repository.

    Args:
        url (str): The base URL of the GitHub repository (e.g., 'https://github.com/user/repo')
                   or the path to the local repository on your system.

    Returns:
        str: The directory tree structure as a string.
    """
   return await git_tree_search(request.repobaseurl)

@app.post('/git-search',operation_id="get_git_search")
async def get_git_search(request:GitSearchRequest):
   """
    Fetches the content of a specific part (directory or file) from either and does what asked in users query.
    First use get_git_tree to understand the structure of the repo and which part might be useful to answer users query
    - a GitHub repository (via URL), or
    - a local Git repository (via local path).

    Args:
        base_url (str): The base URL of the GitHub repository (e.g., 'https://github.com/user/repo'),
                        or the local path to the root of the repository.
        part (str): The path inside the repository you wish to access (e.g., 'basefolder/subfolder'). use get_git_tree for getting specific part if needed
        query (str): Users query
        type (str): "Folder" or "file"
    Returns:
        str: Response of the users query based on the content fetched
    """
   content = await git_specific_content(request.repobaseurl,request.parttoresearch,request.type)
   prompt = f"""You are a professional coder, your task is to answer the users query based on the content fetched from git repo
User Query: {request.query}
Fetched Content: {content}
"""

   result = await llm.ainvoke(
        prompt
    )
   return result.content
   
@app.post('/web-search',operation_id="get_web_search")
async def websearch(request: WebSearchRequest):
    """
    Performs a web search and retrieves results, then generates a response based on those results.
    It also throws back the next steps, you should carry out your research until there are no next steps left.
    RECOMMENDATION: Be specific with the query to get the most relevant results. and Set num_results to 2 (for better results)
    Args:
        query (str): The input query.
        rerank (bool): Whether to rerank results.
        num_results (int, optional): Number of search results to retrieve. Defaults to 3. (can take values from 1-5)
        document_paths (list of str, optional): List of paths for local documents/folders. Defaults to empty list. for an example [path1,path2,path3]. if different tasks are related to different documents
        local_mode (bool, optional): Whether to process local documents. Defaults to False.
        split (bool, optional): Whether to split documents into chunks. Defaults to True.

    Returns:
        str: Generated response to query based on the retrieved and reranked search results and sources
    """
    # You may need to adjust these arguments based on your actual setup
    # For demonstration, using None for models and embeddings
    try:
        result = await query_web_response(
            query=request.query,
            date=date,
            day=day,
            websearcher=searcher,  # Replace with your actual searcher instance if needed
            hf_embeddings=hf_embeddings,
            rerank=request.rerank,
            cross_encoder=cross_encoder,
            model=llm,
            text_model=llm,
            num_results=min(2,request.num_results),
            document_paths=request.document_paths,
            local_mode=request.local_mode,
            split=request.split
        )
        return "result:" + result[0] + '\n\nsources:' + result[1]
    except:
        return "No Websites found, Try rephrasing query"

@app.post('/web-summarize', operation_id="get_web_summarize")
async def websummarize(request: WebSummarizeRequest):
    """Generates a summary of a web page based on the provided query and URL.
    Args:
        query (str): The input query.
        url (str): The URL of the web page to summarize.
        model (str): The model to use for summarization.
        local_mode (bool): Whether to process local documents.
    Returns:
        str:  The generated summary of the url provided to answer query"""
    try:
        result = await summary_of_url(
            query=request.query,
            url=request.url,
            model=llm,  # Replace with your actual model if needed
            local_mode=request.local_mode
        )
        return result
    except:
        return "URL is not reacheable, try different URL"

@app.post('/youtube-search', operation_id="get_youtube_search")
async def youtube_search(request: YouTubeSearchRequest):
    """Performs a YouTube search and return summaries of it.
    Args:
        query (str): The YouTube video URL if provided else search term
        prompt (str): The prompt to generate a response from the transcript.
        n (int): Number of videos to summarize if search term is provided instead of URL.
    Returns:
        str: response from the YouTube transcripts based on the given query"""
    # You may need to adjust the model argument as per your setup
    result = youtube_transcript_response(
        request.query,
        request.prompt,
        n = request.n, #number of videos to summarise
        model=llm  # Replace with your actual model if needed
    )
    return result

@app.post('/reddit-search', operation_id="get_reddit_search")
async def reddit_search(request: RedditSearchRequest):
    """Performs a Reddit search and retrieves posts based on the provided parameters.
    Args:
        subreddit (str): The subreddit to search in. When search_query is provided
        url_type (str): The type of Reddit URL to fetch (e.g., 'search','hot', 'new','top','best','controversial','rising').
                        set to 'search' if specific search_query is provided
        n (int): Number of posts to retrieve.
        k (int): Number of comments on each post to return after processing. When more perspectives needed increase this.
        custom_url (str): Custom URL for Reddit search.
        time_filter (str): Time filter for the search (e.g., 'all', 'day').
        search_query (str): Search query for Reddit posts. IF NOT SEARCHING FOR A QUERY, dont set this value, keep it ""
        sort_type (str): Sorting type for the results.
        Returns:                                            
            str: A response containing the summary of the Reddit search results"""  
    # You may need to adjust the model argument as per your setup
    if request.search_query:
        request.url_type = 'search'
    result = reddit_reader_response(
        subreddit=request.subreddit,
        url_type=request.url_type,
        n=request.n,
        k=request.k,
        custom_url=request.custom_url,
        time_filter=request.time_filter,
        search_query=request.search_query,
        sort_type=request.sort_type,
        model=llm  # Replace with your actual model if needed
    )
    return result

@app.post('/map-search', operation_id="get_map_search")
async def map_search(request: MapSearchRequest):
    """Performs a map search and retrieves the route and points of interest like  (POIs) between two locations.
    Args:
        start_location (optional str): The starting location for the route. can be None as well
        end_location (optional str): The destination location for the route.can be None as well
        pois_radius (int): Radius in meters to search for points of interest around the route.
        amenities (str): Types of amenities to search for, separated by '|'. For example, "restaurant|cafe|bar|hotel".
        limit (int): Maximum number of POIs to return.
        task (str): The task to perform, either "location_only" - if lat long of start and end location is needed,
            else by default is "route_and_pois" - if route and POIs are needed.
    Returns:
        dict: location or route and POIs or both"""
    result = generate_map(request.start_location,
                        request.end_location,
                        pois_radius=request.pois_radius,
                        amenities=request.amenities,
                        limit=request.limit,
                        task=request.task,
                        )
    return result

@app.post('/check-response', operation_id="get_response_check")
async def check_response(request: ResearchCheckRequest):
    """
    Evaluates whether the agent's collected information is complete for writing answer to the user's query. 
    If any aspect is missing, list them all in bullet format
    Args:
        query (str): The user's original query.
        toolsshorthand (str):  Exact Facts/Information collected in bullets from every past tool usage which would be useful to answer
    Returns:
        str: Suggestions for improvement or confirmation that all aspects are addressed.
    """
    system_prompt = f"""You are a professional researcher.
Review the following user query and the agent's short hand of informations collected. 
If not explicitly asked for deep research, you should just check if most necessary information and all aspects present in query are covered, NO NEED TO SUGGEST EXTRA, SINCE ITS QUICK QUERY
Determine if the shorthand fully addresses every aspect and intent of the query.
If any part is missing or could be improved, list the specific aspects or suggestions for further research or value addition.(IF DEEP RESEARCH ASKED EXPLICITLY)
If the response is complete, state that all aspects have been addressed.

User Query: {request.query}
Agent Shorthand: {request.toolsshorthand}
"""

    result = await llm.ainvoke(
        system_prompt
    )
    return result.content

@app.post('/text-to-podcast', operation_id="get_podcast")
async def podcaster(request: PodcastRequest):
    """
    Converts a list of sentences with specified voices into a podcast audio file.
    Each sentence is spoken in the specified voice, and random pauses are added between sentences for natural flow.
    
    Args:
        prompt: The theme or topic of the podcast episode. You can even provide length instructions, like shorter/longer duration, tone, etc.
        text: The detailed content over which the podcast is to be made.
    Returns:
        FileResponse: The generated podcast .wav file. or str
    """
    system_prompt = f"""You are an experienced podcaster who can create engaging episodes on any topic.
Your style makes complex concepts simple, clear, and enjoyable to listen to.

When writing scripts:

Use natural, conversational language.

Avoid special characters (like *, #, etc.) and TTS markup (such as <prosody> tags).

Do not include background descriptions or stage directions.

Always stay on the provided theme (if one is given). If no theme is provided, use the given text to generate engaging, informative content.

The podcast script should be formatted as follows:

<podcast>
[Person1] What Person1 says [Person2] What Person2 says ...
</podcast>


Where each [Person] represents a speaker, followed by their dialogue.

Theme: {request.prompt}
Text: {request.text}
"""

    result = await llm.ainvoke(
        system_prompt
    )
    voice_choices = ["af_heart","am_michael","am_adam","am_eric","am_echo","am_puck",
                     "am_fenrir","am_santa","am_liam","af_river"
                     ]
    podcast_segments = await parse_podcast(result.content, voice_choices)

    try:
        if os.path.exists("output/podcasts") is False:
            os.makedirs("output/podcasts")
        file_path = f"output/podcasts/podcast_{str(uuid4())[:8]}.wav"
        _ = await podcasting(podcast_segments, filename=file_path)
        logger.info(f"Current working directory: {os.getcwd()}")
        
        logger.info(f"Podcast file created at: {file_path}")
        try:
            return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=os.path.basename(file_path)
            )
        except:
            return f"Generated podcast and stored at {file_path}"
    except Exception as e:
        return {"error": f"Error occurred while creating podcast: {e}"}

@app.post('/basic-tts', operation_id="get_basic_tts")
async def basic_tts(request: BasicTTSRequest):
    """Converts input text to speech using the specified voice and language, and returns the generated audio file.
    Args:
        request (BasicTTSRequest): The request object containing the following fields:
            - text (str): The text to be converted to speech.
            - voice (str): The voice to use for speech synthesis.
            - lang (str): The language code for speech synthesis.
            - filename (str, optional): The output filename for the generated audio file.
    Returns:
        FileResponse: The generated audio file in WAV format if successful.
        dict: An error message if text is not provided or if an exception occurs during TTS generation.
    """
    text = request.text
    voice = request.voice
    lang = request.lang
    filename = request.filename

    if not filename:
        filename = f"output/basic_tts_{str(uuid4())[:8]}.wav"

    if not text:
        return {"error": "Text is required for TTS."}

    try:
        await text_to_speech(text, voice, filename, lang)
        return FileResponse(
            filename,
            media_type="audio/wav",
            filename=os.path.basename(filename)
        )
    except Exception as e:
        return {"error": f"Error occurred while creating TTS: {e}"}
    
    
mcp = FastApiMCP(app,include_operations=['get_web_search',
                                         'get_web_summarize',
                                         'get_youtube_search',
                                         'get_reddit_search',
                                         'get_map_search',
                                         "get_git_tree",
                                         "get_git_search",
                                         "get_local_folder_tree",
                                         "get_response_check",
                                         "get_website_structure",
                                         "get_podcast",
                                         "get_basic_tts"
                                         ],)
mcp.mount()

# Display startup banner when the app starts
display_startup_banner(host=HOST_APP, port=PORT_NUM_APP)