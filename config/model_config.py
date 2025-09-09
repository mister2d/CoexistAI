import os
"""
This module defines the configuration for language model (LLM) and embedding models.
Attributes:
    api_key (str): The OpenAI API key, loaded from the environment variable 'OPENAI_API_KEY'.
    model_config (dict): A dictionary containing configuration parameters for LLM and embedding models.
        Keys:
            - "llm_model_name" (str): Name of the LLM model to use.
            - "llm_type" (str): Type of the LLM provider (e.g., "openai"). "local" is for lmstudio, 
            for ollama and other local models use "others" with base_url updated in openai_compatible.
            - If you using others llm type, then check the openai_compatible url dict for others key, you can generally 
            find it by "googling YOUR provider name openai api base compatilble url"
            - "llm_base_url" (str): Base URL for the LLM API endpoint.
            - "llm_tools" (list): List of tools or plugins to use with the LLM.
            - "llm_kwargs" (dict): Additional keyword arguments for LLM initialization.
                - "temperature" (float): Sampling temperature for generation.
                - "max_tokens" (int or None): Maximum number of tokens to generate.
                - "timeout" (int or None): Timeout for API requests.
                - "max_retries" (int): Maximum number of retries for failed requests.
                - "api_key" (str): API key for authentication.
            - "embedding_model_name" (str): Name of the embedding model to use.
            - "embed_mode" (str): Embedding mode or backend.
            - "cross_encoder_name" (str): Name of the cross-encoder model for reranking.
"""
############## PORT and HOST SETTINGS
# Use environment variables if available, otherwise use defaults
PORT_NUM_SEARXNG = int(os.environ.get('SEARXNG_PORT', '8085'))
PORT_NUM_APP = int(os.environ.get('PORT', '8000'))
HOST_APP = os.environ.get('HOST', '0.0.0.0')
HOST_SEARXNG = os.environ.get('SEARXNG_HOST', 'localhost')
###############

## USER INPUTS NEEDED
#for open source model you can replace it by 'DUMMY' (for both llm and embed), else respective providers 
llm_api_key = os.environ.get('GOOGLE_API_KEY', 'DUMMY') # either paste llm key, based on provider (for an instance, Google) here directly or export it in the env, else dummy for local
embed_api_key = os.environ.get('GOOGLE_API_KEY', 'DUMMY') # either paste embeder key, based on provider (for an instance, Google) here directly or export it in the env, else dummy for local

model_config = {
    # Name of the LLM model to use. For local models, use the model name served by your local server.
    "llm_model_name": "gemini-2.0-flash",

    # LLM provider type: choose from 'google', 'local', 'groq', or 'openai' or 'others' 
    # in case of 'others' (base url needs to be updated in openai_compatible given below accordingly).
    # Make sure to update the api_key variable above to match the provider.
    # "local" is for lmstudio, for ollama and other local models use "others" with base_url updated in openai_compatible.
    # You can generally find it by "googling YOUR PROVIDER (example ollama) name openai api base compatible url"
    "llm_type": "google",

    # List of tools or plugins to use with the LLM, if any. Set to None if not used.
    "llm_tools": None,

    # Additional keyword arguments for LLM initialization.
    "llm_kwargs": {
        "temperature": 0.1,  # Sampling temperature for generation.
        "max_tokens": None,  # Maximum number of tokens to generate (None for default).
        "timeout": None,     # Timeout for API requests (None for default).
        "max_retries": 2,    # Maximum number of retries for failed requests.
        "api_key": llm_api_key,  # API key for authentication.
    },

    # Name of the embedding model to use.
    # For Google, use their embedding model names. For local/HuggingFace, use the model path or name.
    "embedding_model_name": "models/embedding-001",

    "embed_kwargs":{"google_api_key": embed_api_key}, #optional additional kwargs for embedding model initialization

    # Embedding backend: 'google' for Google, 'infinity_emb' for local/HuggingFace models.
    "embed_mode": "google",

    # Name of the cross-encoder model for reranking, typically a HuggingFace model.
    "cross_encoder_name": "BAAI/bge-reranker-base"
}


# NO CHANGE NEEDED UNLESS PROVIDER CHANGES THE BASE URLS, OR YOU WANT TO USE DIFFERENT PROVIDER UNDER "others"
openai_compatible = {
    'google': "https://generativelanguage.googleapis.com/v1beta/openai/",
    'local': "http://127.0.0.1:1234/v1",
    'groq': 'https://api.groq.com/openai/v1',
    'openai':'https://api.openai.com/v1',
    'others': 'https://openrouter.ai/api/v1' # for an example I have added here the openrouter api, since its openai compatible
}

#####IF YOU WANT TO GO ALL LOCAL 

# model_config = {
#     # Name of the LLM model to use. For local models, use the model name served by your local server.
#     "llm_model_name": "google/gemma-3-12b",

#     # LLM provider type: choose from 'google', 'local', 'groq', or 'openai' or 'others' 
#     # in case of 'others' (base url needs to be updated in openai_compatible given below accordingly).
#     # Make sure to update the api_key variable above to match the provider.
#     "llm_type": "local", 

#     # List of tools or plugins to use with the LLM, if any. Set to None if not used.
#     "llm_tools": None,

#     # Additional keyword arguments for LLM initialization.
#     "llm_kwargs": {
#         "temperature": 0.1,  # Sampling temperature for generation.
#         "max_tokens": None,  # Maximum number of tokens to generate (None for default).
#         "timeout": None,     # Timeout for API requests (None for default).
#         "max_retries": 2,    # Maximum number of retries for failed requests.
#         "api_key": llm_api_key,  # API key for authentication.
#     },

#     # Name of the embedding model to use.
#     # For Google, use their embedding model names. For local/HuggingFace, use the model path or name.
#     "embedding_model_name": "nomic-ai/nomic-embed-text-v1",

#     "embed_kwargs":{}, #additional kwargs for embedding model initialization

#     # Embedding backend: 'google' for Google, 'infinity_emb' for local/HuggingFace models.
#     "embed_mode": "infinity_emb",

#     # Name of the cross-encoder model for reranking, typically a HuggingFace model.
#     "cross_encoder_name": "BAAI/bge-reranker-base"
# }