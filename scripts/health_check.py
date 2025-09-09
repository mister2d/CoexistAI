#!/usr/bin/env python3
"""
Health check script for CoexistAI Docker container
Compatible with Docker HEALTHCHECK instruction
"""

import os
import sys
import urllib.request
import urllib.error
import time

def health_check():
    """Simple health check for the FastAPI application"""
    try:
        port = os.environ.get('PORT', '8000')
        host = os.environ.get('HOST', '0.0.0.0')
        
        # If host is 0.0.0.0, use localhost for health check
        check_host = 'localhost' if host == '0.0.0.0' else host
        
        url = f"http://{check_host}:{port}/"
        
        # Try to connect to the root endpoint with timeout
        response = urllib.request.urlopen(url, timeout=10)
        
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

def wait_for_service(max_retries=30, delay=1):
    """Wait for the service to become healthy"""
    for attempt in range(max_retries):
        if health_check() == 0:
            print(f"✅ Service became healthy after {attempt + 1} attempts")
            return True
        time.sleep(delay)
    
    print(f"❌ Service did not become healthy after {max_retries} attempts")
    return False

if __name__ == "__main__":
    # If called with --wait flag, wait for service to become healthy
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        success = wait_for_service()
        sys.exit(0 if success else 1)
    else:
        # Standard health check for Docker
        sys.exit(health_check())