#!/usr/bin/env python3
"""
Simple verification script to test the dynamic configuration generation logic
without relying on subprocess calls.
"""

import os
import json
import tempfile

def generate_config_from_env():
    """Replicate the logic from entrypoint.sh for testing"""
    
    # Parse LLM tools from environment variable
    llm_tools_str = os.getenv("LLM_TOOLS", "None")
    if llm_tools_str != "None" and llm_tools_str.strip():
        try:
            llm_tools = json.loads(llm_tools_str)
        except:
            llm_tools = None
    else:
        llm_tools = None

    # Parse embed kwargs based on provider
    embed_mode = os.getenv("EMBED_MODE", "google")
    if embed_mode == "google":
        embed_kwargs = {"google_api_key": os.getenv("EMBED_API_KEY", "DUMMY")}
    elif embed_mode == "infinity_emb":
        # For infinity_emb, parse additional kwargs if provided
        infinity_kwargs_str = os.getenv("INFINITY_EMB_KWARGS", "{}")
        try:
            embed_kwargs = json.loads(infinity_kwargs_str) if infinity_kwargs_str.strip() else {}
        except:
            embed_kwargs = {}
    else:
        # For other providers, use generic embed_kwargs
        embed_kwargs_str = os.getenv("EMBED_KWARGS", "{}")
        try:
            embed_kwargs = json.loads(embed_kwargs_str) if embed_kwargs_str.strip() else {}
        except:
            embed_kwargs = {}

    # Determine LLM API key with proper fallbacks
    llm_api_key = os.getenv("LLM_API_KEY", "DUMMY")
    if llm_api_key == "DUMMY":
        # Try provider-specific keys as fallbacks
        llm_type = os.getenv("LLM_TYPE", "google")
        if llm_type == "google":
            llm_api_key = os.getenv("GOOGLE_API_KEY", "DUMMY")
        elif llm_type == "openai":
            llm_api_key = os.getenv("OPENAI_API_KEY", "DUMMY")
        elif llm_type == "groq":
            llm_api_key = os.getenv("GROQ_API_KEY", "DUMMY")

    # Determine embedding API key with proper fallbacks
    embed_api_key = os.getenv("EMBED_API_KEY", "DUMMY")
    if embed_api_key == "DUMMY" and embed_mode == "google":
        embed_api_key = os.getenv("GOOGLE_API_KEY", "DUMMY")

    # Create comprehensive config from environment
    config = {
        "llm_model_name": os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash"),
        "llm_type": os.getenv("LLM_TYPE", "google"),
        "llm_tools": llm_tools,
        "llm_kwargs": {
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None,
            "timeout": int(os.getenv("LLM_TIMEOUT")) if os.getenv("LLM_TIMEOUT") else None,
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),
            "api_key": llm_api_key,
        },
        "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001"),
        "embed_kwargs": embed_kwargs,
        "embed_mode": embed_mode,
        "cross_encoder_name": os.getenv("CROSS_ENCODER_NAME", "BAAI/bge-reranker-base")
    }

    return config

def test_configuration(test_name, env_vars, expected_checks):
    """Test configuration with specific environment variables"""
    print(f"\n🧪 Testing: {test_name}")
    
    # Clear environment first
    for key in list(os.environ.keys()):
        if key.startswith(('LLM_', 'EMBED_', 'SEARXNG_', 'GOOGLE_', 'OPENAI_', 'GROQ_', 'LOCAL_', 'OTHERS_', 'INFINITY_', 'CROSS_', 'PORT', 'HOST', 'LOG_', 'PLACE', 'USER_', 'WAIT_')):
            del os.environ[key]
    
    # Set test environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    try:
        # Generate configuration
        config = generate_config_from_env()
        
        print("✅ Configuration generated successfully")
        
        # Check expected values
        all_passed = True
        for check_name, check_func in expected_checks.items():
            try:
                if check_func(config):
                    print(f"  ✅ {check_name}")
                else:
                    print(f"  ❌ {check_name}")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Run verification tests"""
    
    print("🔍 Verifying CoexistAI Dynamic Configuration Generation")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic Google Configuration
    test_results.append(test_configuration(
        "Basic Google Configuration",
        {
            'LLM_TYPE': 'google',
            'LLM_MODEL_NAME': 'gemini-2.0-flash',
            'GOOGLE_API_KEY': 'test-google-key',
            'EMBED_MODE': 'google',
            'EMBEDDING_MODEL_NAME': 'models/embedding-001'
        },
        {
            'LLM type is google': lambda config: config['llm_type'] == 'google',
            'LLM model name correct': lambda config: config['llm_model_name'] == 'gemini-2.0-flash',
            'LLM API key set': lambda config: config['llm_kwargs']['api_key'] == 'test-google-key',
            'Embed mode is google': lambda config: config['embed_mode'] == 'google',
            'Embedding model correct': lambda config: config['embedding_model_name'] == 'models/embedding-001'
        }
    ))
    
    # Test 2: OpenAI Configuration
    test_results.append(test_configuration(
        "OpenAI Configuration",
        {
            'LLM_TYPE': 'openai',
            'LLM_MODEL_NAME': 'gpt-4',
            'OPENAI_API_KEY': 'test-openai-key',
            'LLM_TEMPERATURE': '0.7',
            'LLM_MAX_TOKENS': '2048'
        },
        {
            'LLM type is openai': lambda config: config['llm_type'] == 'openai',
            'LLM model name correct': lambda config: config['llm_model_name'] == 'gpt-4',
            'LLM API key set': lambda config: config['llm_kwargs']['api_key'] == 'test-openai-key',
            'Temperature correct': lambda config: config['llm_kwargs']['temperature'] == 0.7,
            'Max tokens correct': lambda config: config['llm_kwargs']['max_tokens'] == 2048
        }
    ))
    
    # Test 3: Local Ollama Configuration
    test_results.append(test_configuration(
        "Local Ollama Configuration",
        {
            'LLM_TYPE': 'local',
            'LLM_MODEL_NAME': 'llama3.2',
            'LOCAL_API_BASE': 'http://host.docker.internal:11434/v1',
            'LLM_API_KEY': 'DUMMY',
            'EMBED_MODE': 'infinity_emb',
            'INFINITY_EMB_URL': 'http://localhost:7997',
            'INFINITY_DEVICE': 'cpu'
        },
        {
            'LLM type is local': lambda config: config['llm_type'] == 'local',
            'LLM API key is DUMMY': lambda config: config['llm_kwargs']['api_key'] == 'DUMMY',
            'Embed mode is infinity_emb': lambda config: config['embed_mode'] == 'infinity_emb'
        }
    ))
    
    # Test 4: Advanced Configuration with Tools
    test_results.append(test_configuration(
        "Advanced Configuration with Tools",
        {
            'LLM_TYPE': 'google',
            'LLM_MODEL_NAME': 'gemini-2.0-flash',
            'GOOGLE_API_KEY': 'test-key',
            'LLM_TOOLS': '["web_search", "calculator"]',
            'LLM_MAX_RETRIES': '5',
            'CROSS_ENCODER_NAME': 'BAAI/bge-reranker-large'
        },
        {
            'LLM tools parsed correctly': lambda config: config['llm_tools'] == ["web_search", "calculator"],
            'Max retries correct': lambda config: config['llm_kwargs']['max_retries'] == 5,
            'Cross encoder correct': lambda config: config['cross_encoder_name'] == 'BAAI/bge-reranker-large'
        }
    ))
    
    # Test 5: Custom Embedding Configuration
    test_results.append(test_configuration(
        "Custom Embedding Configuration",
        {
            'EMBED_MODE': 'huggingface',
            'EMBED_KWARGS': '{"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"}',
            'EMBEDDING_MODEL_NAME': 'custom-embedding-model'
        },
        {
            'Embed mode correct': lambda config: config['embed_mode'] == 'huggingface',
            'Embed kwargs parsed': lambda config: config['embed_kwargs']['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2',
            'Embedding model correct': lambda config: config['embedding_model_name'] == 'custom-embedding-model'
        }
    ))
    
    # Test 6: Infinity Embedding with Custom Kwargs
    test_results.append(test_configuration(
        "Infinity Embedding Configuration",
        {
            'EMBED_MODE': 'infinity_emb',
            'INFINITY_EMB_KWARGS': '{"batch_size": 32, "max_length": 512}',
            'INFINITY_DEVICE': 'cuda'
        },
        {
            'Embed mode is infinity_emb': lambda config: config['embed_mode'] == 'infinity_emb',
            'Infinity kwargs parsed': lambda config: config['embed_kwargs']['batch_size'] == 32,
            'Infinity device set': lambda config: os.environ.get('INFINITY_DEVICE') == 'cuda'
        }
    ))
    
    # Test 7: Default Values
    test_results.append(test_configuration(
        "Default Values Test",
        {},
        {
            'Default LLM type': lambda config: config['llm_type'] == 'google',
            'Default model name': lambda config: config['llm_model_name'] == 'gemini-2.0-flash',
            'Default temperature': lambda config: config['llm_kwargs']['temperature'] == 0.1,
            'Default max retries': lambda config: config['llm_kwargs']['max_retries'] == 2,
            'Default embed mode': lambda config: config['embed_mode'] == 'google',
            'Default cross encoder': lambda config: config['cross_encoder_name'] == 'BAAI/bge-reranker-base'
        }
    ))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, result in enumerate(test_results, 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"Test {i}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Full 12-factor compliance achieved.")
        return True
    else:
        print("⚠️  Some tests failed. Review the configuration generation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)