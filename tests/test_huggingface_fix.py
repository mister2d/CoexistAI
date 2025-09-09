#!/usr/bin/env python3
"""
Test script to validate HuggingFaceCrossEncoder permission fix
"""

import os
import sys
import getpass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_huggingface_setup():
    """Test HuggingFace cache directory setup and permissions"""
    logger.info("=== Testing HuggingFace Setup ===")
    
    # Check current user
    current_user = getpass.getuser()
    logger.info(f"Current user: {current_user}")
    
    # Check environment variables
    hf_home = os.environ.get("HF_HOME", "Not set")
    
    logger.info(f"HF_HOME: {hf_home}")
    # Note: TRANSFORMERS_CACHE and HF_DATASETS_CACHE are deprecated in favor of HF_HOME
    
    # Check home directory
    home_dir = os.path.expanduser("~")
    logger.info(f"Home directory: {home_dir}")
    logger.info(f"Home directory exists: {os.path.exists(home_dir)}")
    
    # Check cache directories
    cache_dirs = [
        "/app/.cache/huggingface",
        os.path.join(home_dir, ".cache", "huggingface"),
        hf_home if hf_home != "Not set" else None
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir:
            logger.info(f"Checking cache directory: {cache_dir}")
            exists = os.path.exists(cache_dir)
            logger.info(f"  Exists: {exists}")
            if exists:
                stat_info = os.stat(cache_dir)
                logger.info(f"  Permissions: {oct(stat_info.st_mode)}")
                logger.info(f"  Owner: {stat_info.st_uid}:{stat_info.st_gid}")
                logger.info(f"  Writable: {os.access(cache_dir, os.W_OK)}")
    
    # Test creating a test file in the cache directory
    test_cache_dir = "/app/.cache/huggingface"
    if os.path.exists(test_cache_dir):
        test_file = os.path.join(test_cache_dir, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"✓ Successfully wrote to cache directory: {test_cache_dir}")
        except Exception as e:
            logger.error(f"✗ Failed to write to cache directory: {e}")
            return False
    else:
        logger.error(f"✗ Cache directory does not exist: {test_cache_dir}")
        return False
    
    return True

def test_huggingface_cross_encoder():
    """Test loading HuggingFaceCrossEncoder"""
    logger.info("=== Testing HuggingFaceCrossEncoder Loading ===")
    
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        logger.info("✓ Successfully imported HuggingFaceCrossEncoder")
        
        # Try to load the cross encoder
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        logger.info("✓ Successfully loaded HuggingFaceCrossEncoder")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load HuggingFaceCrossEncoder: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

def main():
    """Main test function"""
    logger.info("Starting HuggingFace permission fix validation...")
    
    # Test setup
    setup_ok = test_huggingface_setup()
    
    # Test cross encoder loading
    encoder_ok = test_huggingface_cross_encoder()
    
    if setup_ok and encoder_ok:
        logger.info("✓ All tests passed! HuggingFace permission fix is working correctly.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())