# CoexistAI 12-Factor Docker Configuration

This document describes the 12-factor compliant Docker setup for CoexistAI, enabling flexible deployment with arbitrary SearxNG instances and OpenAI-compatible endpoints.

## Table of Contents

- [12-Factor Principles](#12-factor-principles)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Service Profiles](#service-profiles)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 12-Factor Principles

This Docker configuration follows the 12-factor app methodology:

1. **Codebase**: Single codebase tracked in version control
2. **Dependencies**: Explicitly declared and isolated via Docker
3. **Config**: Stored in environment variables, not code
4. **Backing Services**: Treat SearxNG, LLMs, and embedding services as attached resources
5. **Build, Release, Run**: Strictly separated build and run stages
6. **Processes**: Stateless, share-nothing architecture
7. **Port Binding**: Self-contained, exports services via port binding
8. **Concurrency**: Scale via process model
9. **Disposability**: Fast startup and graceful shutdown
10. **Dev/Prod Parity**: Keep development, staging, and production as similar as possible
11. **Logs**: Treat logs as event streams
12. **Admin Processes**: Run admin/management tasks as one-off processes

## Quick Start

### 1. Basic Setup (Google Gemini + SearxNG)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
nano .env
# Set: GOOGLE_API_KEY=your-api-key-here

# Start services
docker-compose up -d

# Access the application
curl http://localhost:8000/docs  # Swagger UI
```

### 2. With Local Infinity Embeddings

```bash
# Start with infinity embedding server
docker-compose --profile with-infinity up -d
```

### 3. OpenAI Configuration

```bash
# Set OpenAI API key in .env
echo "OPENAI_API_KEY=your-openai-key" >> .env

# Start OpenAI profile
docker-compose --profile openai up -d
```

## Configuration

### Environment Variables

The application is configured entirely through environment variables. Key categories:

#### Application Settings
- `PORT`: Application port (default: 8000)
- `HOST`: Application host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)
- `PLACE`: Geographic location for personalized results (see valid values below)

#### SearxNG Configuration
- `SEARXNG_HOST`: SearxNG service hostname
- `SEARXNG_PORT`: SearxNG service port
- `SEARXNG_PROTOCOL`: SearxNG protocol (http/https)

#### LLM Configuration
- `LLM_TYPE`: Provider type (google, openai, local, groq, others)
- `LLM_MODEL_NAME`: Model name
- `LLM_API_KEY`: API key for authentication
- `LLM_TEMPERATURE`: Generation temperature
- `LLM_MAX_TOKENS`: Maximum tokens (optional)
- `LLM_TIMEOUT`: Request timeout (optional)
- `LLM_MAX_RETRIES`: Maximum retry attempts

#### Embedding Configuration
- `EMBEDDING_MODEL_NAME`: Embedding model name
- `EMBED_MODE`: Backend (google, openai, infinity_emb, huggingface)
- `EMBED_API_KEY`: API key for embeddings
- `CROSS_ENCODER_NAME`: Reranking model name
- `INFINITY_EMB_URL`: Infinity embedding server URL

### Service Profiles

Docker Compose profiles allow different configurations:

#### Default Profile (Google Gemini)
```bash
docker-compose up -d
```

#### With Infinity Embeddings
```bash
docker-compose --profile with-infinity up -d
```

#### OpenAI Profile
```bash
docker-compose --profile openai up -d
```

#### Local LLM Profile
```bash
docker-compose --profile local up -d
```

## Examples

### 1. Google Gemini with Custom SearxNG

```bash
# Use external SearxNG instance
export SEARXNG_HOST=search.example.com
export SEARXNG_PORT=443
export SEARXNG_PROTOCOL=https
export GOOGLE_API_KEY=your-key

docker-compose up -d
```

### 2. OpenAI with Custom Endpoint

```bash
# Use custom OpenAI-compatible endpoint
export LLM_TYPE=others
export OTHERS_API_BASE=https://api.custom-provider.com/v1
export LLM_MODEL_NAME=gpt-4
export LLM_API_KEY=your-key

docker-compose up -d
```

### 3. Local Ollama Setup

```bash
# Local Ollama on host machine
export LLM_TYPE=local
export LOCAL_API_BASE=http://host.docker.internal:11434/v1
export LLM_MODEL_NAME=llama3.2
export LLM_API_KEY=DUMMY

# Local embeddings
export EMBED_MODE=infinity_emb
export INFINITY_EMB_URL=http://localhost:7997

docker-compose --profile with-infinity up -d
```

### 4. Multiple LLM Providers

```bash
# Start multiple configurations simultaneously
docker-compose up -d                    # Google Gemini (port 8000)
docker-compose --profile openai up -d   # OpenAI (port 8001)
docker-compose --profile local up -d    # Local LLM (port 8002)
```

## Deployment

### Production Deployment

1. **External SearxNG**: Use managed SearxNG service or deploy separately
2. **API Keys**: Store in secure secret management system
3. **SSL/TLS**: Use reverse proxy (nginx, traefik) for HTTPS
4. **Monitoring**: Add health checks and monitoring
5. **Scaling**: Use container orchestration (Kubernetes, Docker Swarm)

### Kubernetes Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coexistai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coexistai
  template:
    metadata:
      labels:
        app: coexistai
    spec:
      containers:
      - name: coexistai
        image: coexistai:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_TYPE
          value: "google"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: google-api-key
        - name: SEARXNG_HOST
          value: "searxng-service"
        - name: SEARXNG_PORT
          value: "8080"
```

### Docker Swarm Example

```bash
# Create secrets
echo "your-google-api-key" | docker secret create google-api-key -

# Deploy stack
docker stack deploy -c docker-compose.yml coexistai
```

## Environment Variables Reference

### Core Application
| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Application port | 8000 |
| `HOST` | Application host | 0.0.0.0 |
| `LOG_LEVEL` | Logging level | INFO |
| `PLACE` | Geographic location | UTC |

### SearxNG
| Variable | Description | Default |
|----------|-------------|---------|
| `SEARXNG_HOST` | SearxNG hostname | localhost |
| `SEARXNG_PORT` | SearxNG port | 8085 |
| `SEARXNG_PROTOCOL` | SearxNG protocol | http |

### LLM Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_TYPE` | Provider type | google |
| `LLM_MODEL_NAME` | Model name | gemini-2.0-flash |
| `LLM_API_KEY` | API key | DUMMY |
| `LLM_TEMPERATURE` | Temperature | 0.1 |
| `LLM_MAX_TOKENS` | Max tokens | (empty) |
| `LLM_TIMEOUT` | Timeout | (empty) |
| `LLM_MAX_RETRIES` | Max retries | 2 |

### API Base URLs
| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_BASE` | Google API base | https://generativelanguage.googleapis.com/v1beta/openai/ |
| `OPENAI_API_BASE` | OpenAI API base | https://api.openai.com/v1 |
| `GROQ_API_BASE` | Groq API base | https://api.groq.com/openai/v1 |
| `LOCAL_API_BASE` | Local API base | http://127.0.0.1:1234/v1 |
| `OTHERS_API_BASE` | Custom API base | https://openrouter.ai/api/v1 |

### Embedding
| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL_NAME` | Embedding model | models/embedding-001 |
| `EMBED_MODE` | Embedding backend | google |
| `EMBED_API_KEY` | Embedding API key | DUMMY |
| `CROSS_ENCODER_NAME` | Reranking model | BAAI/bge-reranker-base |
| `INFINITY_EMB_URL` | Infinity server URL | http://localhost:7997 |

### PLACE Variable Configuration

The `PLACE` environment variable provides geographic context for time-sensitive and location-specific queries. This variable is used in AI prompts to enhance relevance when users ask about local events, weather, business hours, or location-specific information.

#### Valid Values Format

The `PLACE` variable accepts any of the following formats:

**City/Location Names:**
- `UTC` (default - Universal Time Coordinate)
- `New York, USA`
- `London, UK`
- `Tokyo, Japan`
- `Sydney, Australia`
- `Paris, France`
- `Berlin, Germany`
- `Mumbai, India`
- `Singapore`
- `São Paulo, Brazil`

**Time Zones:**
- `UTC` (Universal Time Coordinate)
- `UTC+5:30` (Indian Standard Time)
- `UTC-5:00` (Eastern Standard Time)
- `UTC+1:00` (Central European Time)
- `UTC+9:00` (Japan Standard Time)

**Coordinates (for precise locations):**
- `40.7128,-74.0060` (New York City coordinates)
- `51.5074,-0.1278` (London coordinates)

**Regional Descriptors:**
- `West Coast, USA`
- `East Coast, USA`
- `Central Europe`
- `Southeast Asia`
- `Middle East`

#### Usage Examples

```bash
# Set to UTC (default, globally neutral)
export PLACE=UTC

# Set to specific city
export PLACE="San Francisco, USA"

# Set to time zone offset
export PLACE="UTC-8:00"

# Set to coordinates for precise location
export PLACE="37.7749,-122.4194"
```

#### Best Practices

1. **Use UTC for global deployments** - When your application serves users worldwide, UTC provides a neutral baseline
2. **Use specific cities for regional services** - When targeting specific geographic markets
3. **Consider user preferences** - Allow users to override location settings when possible
4. **Be consistent** - Use the same location format throughout your deployment

For a comprehensive list of valid city names and time zones, refer to the [IANA Time Zone Database](https://www.iana.org/time-zones) or [GeoNames geographical database](https://www.geonames.org/).

## Troubleshooting

### Common Issues

1. **SearxNG Connection Failed**
   ```bash
   # Check SearxNG health
   curl http://localhost:8085/health
   
   # Check logs
   docker-compose logs searxng
   ```

2. **API Key Issues**
   ```bash
   # Verify API key is set
   docker-compose exec coexistai env | grep API_KEY
   
   # Test LLM connection
   curl -X POST http://localhost:8000/web-search \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "rerank": false}'
   ```

3. **Infinity Embedding Server**
   ```bash
   # Check if infinity is running
   curl http://localhost:7997/health
   
   # Start with infinity profile
   docker-compose --profile with-infinity up -d
   ```

### Health Checks

All services include health checks:
- **CoexistAI**: `http://localhost:8000/health`
- **SearxNG**: `http://localhost:8085/health`
- **Infinity**: `http://localhost:7997/health`

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f coexistai
docker-compose logs -f searxng
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Network**: Use Docker networks to isolate services
3. **Volumes**: Mount only necessary directories
4. **User**: Application runs as non-root user
5. **Secrets**: Use Docker secrets or external secret management

## Performance Tuning

1. **Resource Limits**: Set appropriate CPU/memory limits
2. **Connection Pooling**: Configure connection pools for external services
3. **Caching**: Implement caching where appropriate
4. **Scaling**: Use multiple container instances for high availability