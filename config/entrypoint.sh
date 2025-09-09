#!/bin/bash
set -e

# Wait for dependent services if needed
if [ -n "$WAIT_FOR_SERVICES" ]; then
    echo "Waiting for dependent services..."
    IFS=","; for service in $WAIT_FOR_SERVICES; do
        echo "Waiting for $service..."
        host=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        timeout 30 bash -c "until curl -f http://$host:$port/health 2>/dev/null; do sleep 1; done" || {
            echo "Service $service not available after 30 seconds"
            exit 1
        }
    done
fi

# Generate dynamic config from environment variables
python -c "
import os
import json

# Parse LLM tools from environment variable
llm_tools_str = os.getenv(\"LLM_TOOLS\", \"None\")
if llm_tools_str != \"None\" and llm_tools_str.strip():
    try:
        llm_tools = json.loads(llm_tools_str)
    except:
        llm_tools = None
else:
    llm_tools = None

# Parse embed kwargs based on provider
embed_mode = os.getenv(\"EMBED_MODE\", \"google\")
if embed_mode == \"google\":
    embed_kwargs = {\"google_api_key\": os.getenv(\"EMBED_API_KEY\", \"DUMMY\")}
elif embed_mode == \"infinity_emb\":
    # For infinity_emb, parse additional kwargs if provided
    infinity_kwargs_str = os.getenv(\"INFINITY_EMB_KWARGS\", \"{}\")
    try:
        embed_kwargs = json.loads(infinity_kwargs_str) if infinity_kwargs_str.strip() else {}
    except:
        embed_kwargs = {}
else:
    # For other providers, use generic embed_kwargs
    embed_kwargs_str = os.getenv(\"EMBED_KWARGS\", \"{}\")
    try:
        embed_kwargs = json.loads(embed_kwargs_str) if embed_kwargs_str.strip() else {}
    except:
        embed_kwargs = {}

# Determine LLM API key with proper fallbacks
llm_api_key = os.getenv(\"LLM_API_KEY\", \"DUMMY\")
if llm_api_key == \"DUMMY\":
    # Try provider-specific keys as fallbacks
    llm_type = os.getenv(\"LLM_TYPE\", \"google\")
    if llm_type == \"google\":
        llm_api_key = os.getenv(\"GOOGLE_API_KEY\", \"DUMMY\")
    elif llm_type == \"openai\":
        llm_api_key = os.getenv(\"OPENAI_API_KEY\", \"DUMMY\")
    elif llm_type == \"groq\":
        llm_api_key = os.getenv(\"GROQ_API_KEY\", \"DUMMY\")

# Determine embedding API key with proper fallbacks
embed_api_key = os.getenv(\"EMBED_API_KEY\", \"DUMMY\")
if embed_api_key == \"DUMMY\" and embed_mode == \"google\":
    embed_api_key = os.getenv(\"GOOGLE_API_KEY\", \"DUMMY\")

# Create comprehensive config from environment
config = {
    \"llm_model_name\": os.getenv(\"LLM_MODEL_NAME\", \"gemini-2.0-flash\"),
    \"llm_type\": os.getenv(\"LLM_TYPE\", \"google\"),
    \"llm_tools\": llm_tools,
    \"llm_kwargs\": {
        \"temperature\": float(os.getenv(\"LLM_TEMPERATURE\", \"0.1\")),
        \"max_tokens\": int(os.getenv(\"LLM_MAX_TOKENS\")) if os.getenv(\"LLM_MAX_TOKENS\") else None,
        \"timeout\": int(os.getenv(\"LLM_TIMEOUT\")) if os.getenv(\"LLM_TIMEOUT\") else None,
        \"max_retries\": int(os.getenv(\"LLM_MAX_RETRIES\", \"2\")),
        \"api_key\": llm_api_key,
    },
    \"embedding_model_name\": os.getenv(\"EMBEDDING_MODEL_NAME\", \"models/embedding-001\"),
    \"embed_kwargs\": embed_kwargs,
    \"embed_mode\": embed_mode,
    \"cross_encoder_name\": os.getenv(\"CROSS_ENCODER_NAME\", \"BAAI/bge-reranker-base\")
}

# Convert JSON null to Python None for proper Python syntax
config_str = json.dumps(config, indent=4)
config_str = config_str.replace('null', 'None')

# Build openai_compatible dictionary with all provider-specific base URLs
openai_compatible = {
    \"google\": os.getenv(\"GOOGLE_API_BASE\", \"https://generativelanguage.googleapis.com/v1beta/openai/\"),
    \"local\": os.getenv(\"LOCAL_API_BASE\", \"http://127.0.0.1:1234/v1\"),
    \"groq\": os.getenv(\"GROQ_API_BASE\", \"https://api.groq.com/openai/v1\"),
    \"openai\": os.getenv(\"OPENAI_API_BASE\", \"https://api.openai.com/v1\"),
    \"others\": os.getenv(\"OTHERS_API_BASE\", \"https://openrouter.ai/api/v1\")
}

# Write comprehensive config
with open(\"/tmp/model_config_dynamic.py\", \"w\") as f:
    f.write(f\"\"\"
import os

# Dynamic configuration from environment variables - Full 12-Factor Compliance
PORT_NUM_SEARXNG = int(os.getenv(\"SEARXNG_PORT\", \"8085\"))
PORT_NUM_APP = int(os.getenv(\"PORT\", \"8000\"))
HOST_APP = os.getenv(\"HOST\", \"0.0.0.0\")
HOST_SEARXNG = os.getenv(\"SEARXNG_HOST\", \"localhost\")

# API Keys with provider-specific fallbacks
llm_api_key = os.getenv(\"LLM_API_KEY\", os.getenv(\"GOOGLE_API_KEY\", \"DUMMY\"))
embed_api_key = os.getenv(\"EMBED_API_KEY\", os.getenv(\"GOOGLE_API_KEY\", \"DUMMY\"))

# Main model configuration
model_config = {config_str}

# Provider-specific API base URLs - fully configurable
openai_compatible = {json.dumps(openai_compatible, indent=4)}

# Additional environment-based configurations
SEARXNG_PROTOCOL = os.getenv(\"SEARXNG_PROTOCOL\", \"http\")
LOG_LEVEL = os.getenv(\"LOG_LEVEL\", \"INFO\")
PLACE = os.getenv(\"PLACE\", \"us\")
USER_AGENT = os.getenv(\"USER_AGENT\", \"CoexistAI/1.0\")

# Infinity embedding server configuration
INFINITY_EMB_URL = os.getenv(\"INFINITY_EMB_URL\", \"http://localhost:7997\")
INFINITY_DEVICE = os.getenv(\"INFINITY_DEVICE\", \"cpu\")
\"\"\")

print(\"Full 12-factor compliant dynamic configuration generated from environment variables\")
"

# Copy dynamic configuration to override static model_config.py
cp /tmp/model_config_dynamic.py /app/model_config.py
echo \"Dynamic configuration copied to override static model_config.py\"

# Start the application
echo "Starting CoexistAI server on port ${PORT:-8000}..."
exec uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --log-level ${LOG_LEVEL,,}