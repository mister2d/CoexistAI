#!/bin/bash
set -e

# CoexistAI Docker Entrypoint Script
# 12-Factor App compliant startup script

echo "🚀 Starting CoexistAI..."

# Validate required environment variables
if [ -z "$PORT" ]; then
    echo "⚠️  PORT not set, defaulting to 8000"
    export PORT=8000
fi

if [ -z "$HOST" ]; then
    echo "⚠️  HOST not set, defaulting to 0.0.0.0"
    export HOST=0.0.0.0
fi

# Create necessary directories if they don't exist
mkdir -p /app/output/basic_tts /app/output/podcasts /app/documents /app/.cache/huggingface

# Set proper permissions (should already be set, but just in case)
chown -R coexistai:coexistai /app 2>/dev/null || true

# Health check endpoint setup
echo "🩺 Setting up health check endpoint..."
cat > /app/health_check.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import urllib.request
import urllib.error

def health_check():
    """Simple health check for the FastAPI application"""
    try:
        port = os.environ.get('PORT', '8000')
        host = os.environ.get('HOST', '0.0.0.0')
        
        # If host is 0.0.0.0, use localhost for health check
        check_host = 'localhost' if host == '0.0.0.0' else host
        
        url = f"http://{check_host}:{port}/"
        
        # Try to connect to the root endpoint
        response = urllib.request.urlopen(url, timeout=5)
        
        if response.status == 200:
            print("✅ Application is healthy")
            return 0
        else:
            print(f"❌ Application returned status {response.status}")
            return 1
            
    except urllib.error.URLError as e:
        print(f"❌ Application is not responding: {e}")
        return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())
EOF

chmod +x /app/health_check.py

# Environment validation
echo "🔍 Validating environment configuration..."
echo "📍 HOST: ${HOST}"
echo "📍 PORT: ${PORT}"
echo "🔍 SEARXNG_HOST: ${SEARXNG_HOST:-not set}"
echo "🔍 LLM_TYPE: ${LLM_TYPE:-not set}"
echo "🔍 LLM_MODEL_NAME: ${LLM_MODEL_NAME:-not set}"

# Wait for dependent services (if configured)
if [ -n "$SEARXNG_HOST" ] && [ "$SEARXNG_HOST" != "localhost" ]; then
    echo "⏳ Waiting for SearxNG service at ${SEARXNG_PROTOCOL:-http}://${SEARXNG_HOST}:${SEARXNG_PORT:-8085}..."
    timeout=30
    while ! curl -f -s "${SEARXNG_PROTOCOL:-http}://${SEARXNG_HOST}:${SEARXNG_PORT:-8085}" > /dev/null 2>&1; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            echo "⚠️  SearxNG service not responding after 30 seconds, continuing anyway..."
            break
        fi
    done
fi

# Start the application
echo "🎯 Starting FastAPI application on ${HOST}:${PORT}..."
exec uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload-dir /app \
    --log-level "${LOG_LEVEL:-info}" \
    --access-log \
    --use-colors