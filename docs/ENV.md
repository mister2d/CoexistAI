# CoexistAI Environment Variables Configuration

This document provides comprehensive documentation for all environment variables used in CoexistAI's 12-factor compliant Docker configuration.

## Table of Contents

- [Core Application Settings](#core-application-settings)
- [LLM Configuration](#llm-configuration)
- [Embedding Configuration](#embedding-configuration)
- [SearxNG Configuration](#searxng-configuration)
- [Provider-Specific API Configuration](#provider-specific-api-configuration)
- [Advanced Configuration](#advanced-configuration)
- [Usage Examples](#usage-examples)
- [Environment Variable Reference](#environment-variable-reference)

## Core Application Settings

### Basic Application Configuration
```bash
PORT=8000                    # Application port (default: 8000)
HOST=0.0.0.0                 # Application host (default: 0.0.0.0)
LOG_LEVEL=INFO               # Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
PLACE=us                     # Geographic location for personalized results (default: us)
USER_AGENT=CoexistAI/1.0     # User agent for web requests (default: CoexistAI/1.0)
```

### Service Dependencies
```bash
WAIT_FOR_SERVICES=           # Comma-separated list of services to wait for (e.g., "localhost:8085,localhost:7997")
```

## LLM Configuration

### Basic LLM Settings
```bash
LLM_TYPE=google              # Provider type: google, openai, local, groq, others (default: google)
LLM_MODEL_NAME=gemini-2.0-flash  # Model name (default: gemini-2.0-flash)
LLM_API_KEY=your-api-key     # API key for LLM authentication (default: DUMMY)
```

### LLM Generation Parameters
```bash
LLM_TEMPERATURE=0.1          # Sampling temperature for generation (default: 0.1)
LLM_MAX_TOKENS=              # Maximum tokens to generate (optional, default: None)
LLM_TIMEOUT=                 # Request timeout in seconds (optional, default: None)
LLM_MAX_RETRIES=2            # Maximum retry attempts (default: 2)
```

### Advanced LLM Configuration
```bash
LLM_TOOLS=None               # JSON array of tools/plugins for LLM (default: None)
# Example: LLM_TOOLS='["tool1", "tool2"]'
```

## Embedding Configuration

### Basic Embedding Settings
```bash
EMBEDDING_MODEL_NAME=models/embedding-001  # Embedding model name (default: models/embedding-001)
EMBED_MODE=google            # Backend: google, infinity_emb, huggingface (default: google)
EMBED_API_KEY=your-embed-key # API key for embeddings (default: DUMMY)
CROSS_ENCODER_NAME=BAAI/bge-reranker-base  # Reranking model (default: BAAI/bge-reranker-base)
```

### Provider-Specific Embedding Configuration
```bash
# For Google embeddings
EMBED_MODE=google

# For Infinity embeddings (local)
EMBED_MODE=infinity_emb
INFINITY_EMB_URL=http://localhost:7997    # Infinity server URL (default: http://localhost:7997)
INFINITY_DEVICE=cpu                       # Device: cpu, cuda (default: cpu)

# For custom embedding providers
EMBED_KWARGS={}            # JSON object with provider-specific kwargs (default: {})
INFINITY_EMB_KWARGS={}     # JSON object for infinity_emb specific kwargs (default: {})
```

## SearxNG Configuration

### Basic SearxNG Settings
```bash
SEARXNG_HOST=localhost       # SearxNG hostname (default: localhost)
SEARXNG_PORT=8085            # SearxNG port (default: 8085)
SEARXNG_PROTOCOL=http        # Protocol: http or https (default: http)
```

## Provider-Specific API Configuration

### Google API
```bash
GOOGLE_API_KEY=your-google-key           # Google API key
GOOGLE_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/  # Base URL
```

### OpenAI API
```bash
OPENAI_API_KEY=your-openai-key           # OpenAI API key
OPENAI_API_BASE=https://api.openai.com/v1  # Base URL
```

### Groq API
```bash
GROQ_API_KEY=your-groq-key               # Groq API key
GROQ_API_BASE=https://api.groq.com/openai/v1  # Base URL
```

### Local/Custom APIs
```bash
LOCAL_API_BASE=http://127.0.0.1:1234/v1  # Local API base URL
OTHERS_API_BASE=https://openrouter.ai/api/v1  # Custom provider base URL
```

## Advanced Configuration

### Fallback API Keys
The system supports provider-specific API key fallbacks:
- `GOOGLE_API_KEY` can be used as fallback for both LLM and embedding if specific keys aren't provided
- `LLM_API_KEY` and `EMBED_API_KEY` take precedence over provider-specific keys

## Usage Examples

### Example 1: Google Gemini Configuration
```bash
# Basic Google setup
export LLM_TYPE=google
export LLM_MODEL_NAME=gemini-2.0-flash
export GOOGLE_API_KEY=your-google-api-key
export EMBED_MODE=google
export EMBEDDING_MODEL_NAME=models/embedding-001

# Optional advanced settings
export LLM_TEMPERATURE=0.1
export LLM_MAX_RETRIES=3
export LOG_LEVEL=INFO
```

### Example 2: OpenAI Configuration
```bash
# OpenAI setup
export LLM_TYPE=openai
export LLM_MODEL_NAME=gpt-4
export OPENAI_API_KEY=your-openai-api-key
export EMBED_MODE=openai
export EMBEDDING_MODEL_NAME=text-embedding-ada-002

# Optional settings
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=2048
```

### Example 3: Local Ollama Configuration
```bash
# Local Ollama setup
export LLM_TYPE=local
export LLM_MODEL_NAME=llama3.2
export LOCAL_API_BASE=http://host.docker.internal:11434/v1
export LLM_API_KEY=DUMMY

# Local embeddings with Infinity
export EMBED_MODE=infinity_emb
export INFINITY_EMB_URL=http://localhost:7997
export EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1
export INFINITY_DEVICE=cpu
```

### Example 4: Groq Configuration
```bash
# Groq setup
export LLM_TYPE=groq
export LLM_MODEL_NAME=mixtral-8x7b-32768
export GROQ_API_KEY=your-groq-api-key
export EMBED_MODE=infinity_emb
export INFINITY_EMB_URL=http://localhost:7997
```

### Example 5: Custom Provider Configuration
```bash
# Custom OpenAI-compatible provider
export LLM_TYPE=others
export LLM_MODEL_NAME=custom-model
export OTHERS_API_BASE=https://api.custom-provider.com/v1
export LLM_API_KEY=your-custom-key

# Custom embedding
export EMBED_MODE=huggingface
export EMBED_KWARGS='{"model_name": "sentence-transformers/all-MiniLM-L6-v2"}'
```

### Example 6: Advanced Configuration with Tools
```bash
# Advanced setup with LLM tools
export LLM_TYPE=google
export LLM_MODEL_NAME=gemini-2.0-flash
export GOOGLE_API_KEY=your-key
export LLM_TOOLS='["web_search", "calculator"]'
export LLM_TEMPERATURE=0.2
export LLM_MAX_TOKENS=4096

# Custom cross-encoder
export CROSS_ENCODER_NAME=BAAI/bge-reranker-large

# SearxNG with HTTPS
export SEARXNG_HOST=search.example.com
export SEARXNG_PORT=443
export SEARXNG_PROTOCOL=https
```

### Example 7: Docker Compose Environment File (.env)
```bash
# Complete .env file for Google Gemini setup
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
PLACE=us

# LLM Configuration
LLM_TYPE=google
LLM_MODEL_NAME=gemini-2.0-flash
GOOGLE_API_KEY=your-google-api-key
LLM_TEMPERATURE=0.1
LLM_MAX_RETRIES=2

# Embedding Configuration
EMBED_MODE=google
EMBEDDING_MODEL_NAME=models/embedding-001
EMBED_API_KEY=your-google-api-key
CROSS_ENCODER_NAME=BAAI/bge-reranker-base

# SearxNG Configuration
SEARXNG_HOST=localhost
SEARXNG_PORT=8085
SEARXNG_PROTOCOL=http
```

## Environment Variable Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Application port | 8000 | No |
| `HOST` | Application host | 0.0.0.0 | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `PLACE` | Geographic location | us | No |
| `USER_AGENT` | User agent string | CoexistAI/1.0 | No |
| `LLM_TYPE` | LLM provider type | google | No |
| `LLM_MODEL_NAME` | LLM model name | gemini-2.0-flash | No |
| `LLM_API_KEY` | LLM API key | DUMMY | Yes (for cloud providers) |
| `LLM_TEMPERATURE` | Generation temperature | 0.1 | No |
| `LLM_MAX_TOKENS` | Max tokens | None | No |
| `LLM_TIMEOUT` | Request timeout | None | No |
| `LLM_MAX_RETRIES` | Max retries | 2 | No |
| `LLM_TOOLS` | LLM tools JSON | None | No |
| `EMBEDDING_MODEL_NAME` | Embedding model | models/embedding-001 | No |
| `EMBED_MODE` | Embedding backend | google | No |
| `EMBED_API_KEY` | Embedding API key | DUMMY | No |
| `CROSS_ENCODER_NAME` | Reranking model | BAAI/bge-reranker-base | No |
| `EMBED_KWARGS` | Embed kwargs JSON | {} | No |
| `INFINITY_EMB_KWARGS` | Infinity kwargs JSON | {} | No |
| `INFINITY_EMB_URL` | Infinity server URL | http://localhost:7997 | No |
| `INFINITY_DEVICE` | Infinity device | cpu | No |
| `SEARXNG_HOST` | SearxNG hostname | localhost | No |
| `SEARXNG_PORT` | SearxNG port | 8085 | No |
| `SEARXNG_PROTOCOL` | SearxNG protocol | http | No |
| `GOOGLE_API_KEY` | Google API key | None | Yes (for Google) |
| `OPENAI_API_KEY` | OpenAI API key | None | Yes (for OpenAI) |
| `GROQ_API_KEY` | Groq API key | None | Yes (for Groq) |
| `GOOGLE_API_BASE` | Google API base URL | https://generativelanguage.googleapis.com/v1beta/openai/ | No |
| `OPENAI_API_BASE` | OpenAI API base URL | https://api.openai.com/v1 | No |
| `GROQ_API_BASE` | Groq API base URL | https://api.groq.com/openai/v1 | No |
| `LOCAL_API_BASE` | Local API base URL | http://127.0.0.1:1234/v1 | No |
| `OTHERS_API_BASE` | Custom provider URL | https://openrouter.ai/api/v1 | No |

## Notes

- All environment variables are optional unless marked as "Required"
- The system uses intelligent fallbacks for missing configuration
- Provider-specific API keys can be used as fallbacks (e.g., `GOOGLE_API_KEY` works for both LLM and embedding if specific keys aren't provided)
- JSON-formatted variables (LLM_TOOLS, EMBED_KWARGS, INFINITY_EMB_KWARGS) must be valid JSON
- For production deployments, store sensitive API keys in secure secret management systems