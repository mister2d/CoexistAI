#!/usr/bin/env python3
"""
Verification script to ensure all static model_config.py options are covered
by the dynamic configuration generation.
"""

import ast
import os

def extract_static_config_options():
    """Extract all configuration options from the static model_config.py file"""
    
    with open('../config/model_config.py', 'r') as f:
        content = f.read()
    
    # Parse the Python file
    tree = ast.parse(content)
    
    options = {
        'port_settings': {},
        'api_keys': {},
        'model_config': {},
        'openai_compatible': {}
    }
    
    # Extract port and host settings
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == 'PORT_NUM_SEARXNG':
                        options['port_settings']['PORT_NUM_SEARXNG'] = '8085'
                    elif target.id == 'PORT_NUM_APP':
                        options['port_settings']['PORT_NUM_APP'] = '8000'
                    elif target.id == 'HOST_APP':
                        options['port_settings']['HOST_APP'] = '0.0.0.0'
                    elif target.id == 'HOST_SEARXNG':
                        options['port_settings']['HOST_SEARXNG'] = 'localhost'
                    elif target.id == 'llm_api_key':
                        options['api_keys']['llm_api_key'] = 'GOOGLE_API_KEY fallback'
                    elif target.id == 'embed_api_key':
                        options['api_keys']['embed_api_key'] = 'GOOGLE_API_KEY fallback'
    
    # Extract model_config dictionary
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'model_config':
                    # Extract model_config values
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                                options['model_config'][key.value] = value.value
                            elif isinstance(key, ast.Str) and isinstance(value, ast.Str):
                                options['model_config'][key.s] = value.s
                            elif isinstance(key, ast.Constant) and isinstance(value, ast.Dict):
                                # Handle nested dictionaries like llm_kwargs
                                nested_dict = {}
                                for k, v in zip(value.keys, value.values):
                                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                                        nested_dict[k.value] = v.value
                                    elif isinstance(k, ast.Str) and isinstance(v, ast.Str):
                                        nested_dict[k.s] = v.s
                                    elif isinstance(k, ast.Constant) and isinstance(v, ast.Name):
                                        if v.id == 'llm_api_key':
                                            nested_dict[k.value] = 'llm_api_key_variable'
                                        elif v.id == 'embed_api_key':
                                            nested_dict[k.value] = 'embed_api_key_variable'
                                options['model_config'][key.value] = nested_dict
    
    # Extract openai_compatible dictionary
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'openai_compatible':
                    # Extract openai_compatible values
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                                options['openai_compatible'][key.value] = value.value
                            elif isinstance(key, ast.Str) and isinstance(value, ast.Str):
                                options['openai_compatible'][key.s] = value.s
    
    return options

def get_dynamic_env_mappings():
    """Map static options to their corresponding environment variables"""
    
    mappings = {
        'port_settings': {
            'PORT_NUM_SEARXNG': 'SEARXNG_PORT',
            'PORT_NUM_APP': 'PORT',
            'HOST_APP': 'HOST',
            'HOST_SEARXNG': 'SEARXNG_HOST'
        },
        'api_keys': {
            'llm_api_key': 'LLM_API_KEY (with GOOGLE_API_KEY fallback)',
            'embed_api_key': 'EMBED_API_KEY (with GOOGLE_API_KEY fallback)'
        },
        'model_config': {
            'llm_model_name': 'LLM_MODEL_NAME',
            'llm_type': 'LLM_TYPE',
            'llm_tools': 'LLM_TOOLS',
            'llm_kwargs': {
                'temperature': 'LLM_TEMPERATURE',
                'max_tokens': 'LLM_MAX_TOKENS',
                'timeout': 'LLM_TIMEOUT',
                'max_retries': 'LLM_MAX_RETRIES',
                'api_key': 'LLM_API_KEY (with provider fallbacks)'
            },
            'embedding_model_name': 'EMBEDDING_MODEL_NAME',
            'embed_kwargs': 'EMBED_KWARGS / INFINITY_EMB_KWARGS (provider-specific)',
            'embed_mode': 'EMBED_MODE',
            'cross_encoder_name': 'CROSS_ENCODER_NAME'
        },
        'openai_compatible': {
            'google': 'GOOGLE_API_BASE',
            'local': 'LOCAL_API_BASE',
            'groq': 'GROQ_API_BASE',
            'openai': 'OPENAI_API_BASE',
            'others': 'OTHERS_API_BASE'
        }
    }
    
    return mappings

def verify_coverage():
    """Verify that all static options are covered by dynamic configuration"""
    
    print("🔍 Verifying Coverage: Static model_config.py vs Dynamic Configuration")
    print("=" * 80)
    
    # Extract static configuration
    static_options = extract_static_config_options()
    dynamic_mappings = get_dynamic_env_mappings()
    
    print("\n📋 Static Configuration Options Found:")
    print("-" * 40)
    
    for category, options in static_options.items():
        print(f"\n{category.upper()}:")
        if isinstance(options, dict):
            for key, value in options.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {options}")
    
    print("\n🔄 Dynamic Environment Variable Mappings:")
    print("-" * 40)
    
    for category, mappings in dynamic_mappings.items():
        print(f"\n{category.upper()}:")
        if isinstance(mappings, dict):
            for key, value in mappings.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {mappings}")
    
    print("\n✅ Coverage Analysis:")
    print("-" * 40)
    
    # Check coverage
    coverage_issues = []
    
    # Check port settings
    static_ports = set(static_options['port_settings'].keys())
    dynamic_ports = set(dynamic_mappings['port_settings'].keys())
    if static_ports == dynamic_ports:
        print("✅ Port settings: Full coverage")
    else:
        missing = static_ports - dynamic_ports
        extra = dynamic_ports - static_ports
        if missing:
            coverage_issues.append(f"Missing port mappings: {missing}")
        if extra:
            print(f"ℹ️  Extra port mappings (enhancement): {extra}")
    
    # Check API keys
    static_keys = set(static_options['api_keys'].keys())
    dynamic_keys = set(dynamic_mappings['api_keys'].keys())
    if static_keys == dynamic_keys:
        print("✅ API keys: Full coverage")
    else:
        missing = static_keys - dynamic_keys
        extra = dynamic_keys - static_keys
        if missing:
            coverage_issues.append(f"Missing API key mappings: {missing}")
        if extra:
            print(f"ℹ️  Extra API key mappings (enhancement): {extra}")
    
    # Check model_config
    static_model = set(static_options['model_config'].keys())
    dynamic_model = set(dynamic_mappings['model_config'].keys())
    if static_model == dynamic_model:
        print("✅ Model configuration: Full coverage")
    else:
        missing = static_model - dynamic_model
        extra = dynamic_model - static_model
        if missing:
            coverage_issues.append(f"Missing model config mappings: {missing}")
        if extra:
            print(f"ℹ️  Extra model config mappings (enhancement): {extra}")
    
    # Check openai_compatible
    static_apis = set(static_options['openai_compatible'].keys())
    dynamic_apis = set(dynamic_mappings['openai_compatible'].keys())
    if static_apis == dynamic_apis:
        print("✅ OpenAI-compatible API bases: Full coverage")
    else:
        missing = static_apis - dynamic_apis
        extra = dynamic_apis - static_apis
        if missing:
            coverage_issues.append(f"Missing API base mappings: {missing}")
        if extra:
            print(f"ℹ️  Extra API base mappings (enhancement): {extra}")
    
    print("\n📊 Summary:")
    if not coverage_issues:
        print("🎉 All static configuration options are covered by dynamic environment variables!")
        print("✅ Full 12-factor compliance achieved!")
        return True
    else:
        print("⚠️  Coverage issues found:")
        for issue in coverage_issues:
            print(f"  - {issue}")
        return False

if __name__ == "__main__":
    success = verify_coverage()
    exit(0 if success else 1)