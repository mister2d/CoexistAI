#!/usr/bin/env python3
"""
Test script to verify the dynamic configuration generation works correctly
with all the new 12-factor compliant environment variables.
"""

import os
import json
import tempfile
import subprocess

def test_environment_config(test_name, env_vars, expected_checks):
    """Test configuration generation with specific environment variables"""
    print(f"\n🧪 Testing: {test_name}")
    
    # Set environment variables
    test_env = os.environ.copy()
    for key, value in env_vars.items():
        test_env[key] = value
    
    # Run the dynamic config generation script
    try:
        # Extract the Python code from entrypoint.sh
        with open('config/entrypoint.sh', 'r') as f:
            entrypoint_content = f.read()
        
        # Extract the Python configuration generation code
        start_marker = 'python -c "'
        end_marker = '"'
        start_idx = entrypoint_content.find(start_marker) + len(start_marker)
        end_idx = entrypoint_content.find(end_marker, start_idx)
        
        if start_idx == -1 or end_idx == -1:
            print("❌ Could not extract Python code from entrypoint.sh")
            return False
            
        python_code = entrypoint_content[start_idx:end_idx]
        
        # Create a temporary file to write the generated config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_config_path = temp_file.name
        
        # Modify the code to write to our temp file instead of /tmp/model_config_dynamic.py
        python_code = python_code.replace('/tmp/model_config_dynamic.py', temp_config_path)
        
        # Execute the Python code
        result = subprocess.run(['python3', '-c', python_code],
                              env=test_env,
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            print(f"❌ Python execution failed: {result.stderr}")
            return False
            
        # Read the generated configuration
        with open(temp_config_path, 'r') as f:
            generated_config = f.read()
        
        print("✅ Configuration generated successfully")
        
        # Check expected values
        all_passed = True
        for check_name, check_func in expected_checks.items():
            try:
                if check_func(generated_config):
                    print(f"  ✅ {check_name}")
                else:
                    print(f"  ❌ {check_name}")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
                all_passed = False
        
        # Cleanup
        os.unlink(temp_config_path)
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Run comprehensive tests for dynamic configuration generation"""
    
    print("🔍 Testing CoexistAI Dynamic Configuration Generation")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic Google Configuration
    test_results.append(test_environment_config(
        "Basic Google Configuration",
        {
            'LLM_TYPE': 'google',
            'LLM_MODEL_NAME': 'gemini-2.0-flash',
            'GOOGLE_API_KEY': 'test-google-key',
            'EMBED_MODE': 'google',
            'EMBEDDING_MODEL_NAME': 'models/embedding-001'
        },
        {
            'LLM type is google': lambda config: '"llm_type": "google"' in config,
            'LLM model name correct': lambda config: '"llm_model_name": "gemini-2.0-flash"' in config,
            'Google API key set': lambda config: 'test-google-key' in config,
            'Embed mode is google': lambda config: '"embed_mode": "google"' in config,
            'Embedding model correct': lambda config: '"embedding_model_name": "models/embedding-001"' in config
        }
    ))
    
    # Test 2: OpenAI Configuration
    test_results.append(test_environment_config(
        "OpenAI Configuration",
        {
            'LLM_TYPE': 'openai',
            'LLM_MODEL_NAME': 'gpt-4',
            'OPENAI_API_KEY': 'test-openai-key',
            'LLM_TEMPERATURE': '0.7',
            'LLM_MAX_TOKENS': '2048'
        },
        {
            'LLM type is openai': lambda config: '"llm_type": "openai"' in config,
            'LLM model name correct': lambda config: '"llm_model_name": "gpt-4"' in config,
            'OpenAI API key set': lambda config: 'test-openai-key' in config,
            'Temperature correct': lambda config: '"temperature": 0.7' in config,
            'Max tokens correct': lambda config: '"max_tokens": 2048' in config
        }
    ))
    
    # Test 3: Local Ollama Configuration
    test_results.append(test_environment_config(
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
            'LLM type is local': lambda config: '"llm_type": "local"' in config,
            'Local API base correct': lambda config: 'http://host.docker.internal:11434/v1' in config,
            'Embed mode is infinity_emb': lambda config: '"embed_mode": "infinity_emb"' in config,
            'Infinity URL correct': lambda config: 'http://localhost:7997' in config,
            'Infinity device correct': lambda config: 'INFINITY_DEVICE = os.getenv("INFINITY_DEVICE", "cpu")' in config
        }
    ))
    
    # Test 4: Advanced Configuration with Tools
    test_results.append(test_environment_config(
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
            'LLM tools parsed correctly': lambda config: '"llm_tools": ["web_search", "calculator"]' in config,
            'Max retries correct': lambda config: '"max_retries": 5' in config,
            'Cross encoder correct': lambda config: '"cross_encoder_name": "BAAI/bge-reranker-large"' in config
        }
    ))
    
    # Test 5: Custom Embedding Configuration
    test_results.append(test_environment_config(
        "Custom Embedding Configuration",
        {
            'EMBED_MODE': 'huggingface',
            'EMBED_KWARGS': '{"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"}',
            'EMBEDDING_MODEL_NAME': 'custom-embedding-model'
        },
        {
            'Embed mode correct': lambda config: '"embed_mode": "huggingface"' in config,
            'Embed kwargs parsed': lambda config: '"model_name": "sentence-transformers/all-MiniLM-L6-v2"' in config,
            'Embedding model correct': lambda config: '"embedding_model_name": "custom-embedding-model"' in config
        }
    ))
    
    # Test 6: Infinity Embedding with Custom Kwargs
    test_results.append(test_environment_config(
        "Infinity Embedding Configuration",
        {
            'EMBED_MODE': 'infinity_emb',
            'INFINITY_EMB_KWARGS': '{"batch_size": 32, "max_length": 512}',
            'INFINITY_DEVICE': 'cuda'
        },
        {
            'Embed mode is infinity_emb': lambda config: '"embed_mode": "infinity_emb"' in config,
            'Infinity kwargs parsed': lambda config: '"batch_size": 32' in config,
            'Infinity device set': lambda config: 'INFINITY_DEVICE = os.getenv("INFINITY_DEVICE", "cuda")' in config
        }
    ))
    
    # Test 7: Provider API Base URLs
    test_results.append(test_environment_config(
        "Custom API Base URLs",
        {
            'LLM_TYPE': 'others',
            'OTHERS_API_BASE': 'https://api.custom-provider.com/v1',
            'GROQ_API_BASE': 'https://custom-groq.example.com/v1',
            'LOCAL_API_BASE': 'http://custom-local:8080/v1'
        },
        {
            'Others API base correct': lambda config: 'https://api.custom-provider.com/v1' in config,
            'Groq API base correct': lambda config: 'https://custom-groq.example.com/v1' in config,
            'Local API base correct': lambda config: 'http://custom-local:8080/v1' in config
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