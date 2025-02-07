import os
import yaml
import json
from typing import List, Dict

def escape_string(s: str) -> str:
    """Escape special characters in strings for Modelfile."""
    # First escape any existing quotes
    s = s.replace('"', '\\"')
    # Remove any newlines and extra whitespace
    s = ' '.join(s.split())
    return s

def create_modelfile(config_path: str = "config.yaml") -> None:
    """Create a properly formatted Modelfile for an Ollama custom model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    modelfile_path = os.path.join(output_dir, config['output']['files']['modelfile'])

    # Build base model and parameters section
    modelfile_content = [
        f"FROM {config['model']['base_model']}",
    ]
    
    # Add all parameters
    for param_name, param_value in config['model']['parameters'].items():
        # Skip num_thread as it's not a valid Ollama parameter
        if param_name != 'num_thread':
            modelfile_content.append(f"PARAMETER {param_name} {param_value}")
    
    # Add template if provided
    if config['model'].get('template'):
        modelfile_content.append(f'\nTEMPLATE """{config["model"]["template"]}"""')
    
    # Load training examples
    training_file = os.path.join(output_dir, config['output']['files']['training'])
    try:
        with open(training_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
            
            # Get unique system messages
            system_messages = set(item['system'] for item in training_data if 'system' in item)
            
            # Use the first system message as the main SYSTEM instruction
            if system_messages:
                system_msg = next(iter(system_messages))
                modelfile_content.append(f'\nSYSTEM """{system_msg}"""')
            
            # Add conversation examples using MESSAGE
            for example in training_data:
                # Clean and escape the strings
                instruction = escape_string(example['instruction'])
                response = escape_string(example['response'])
                
                modelfile_content.extend([
                    f'\nMESSAGE user "{instruction}"',
                    f'MESSAGE assistant "{response}"'
                ])
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")
        return
    
    # Write the Modelfile
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modelfile_content))
    
    print(f"Created Modelfile at: {modelfile_path}")
    
    # Print a sample of the generated content for verification
    print("\nFirst few lines of the generated Modelfile:")
    with open(modelfile_path, 'r') as f:
        print(f.read()[:500] + "...")

def main():
    config_path = "config.yaml"
    
    print("Creating Modelfile...")
    create_modelfile(config_path)
    
    # Print configuration summary
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n=== Training Configuration ===")
    print(f"Base Model: {config['model']['base_model']}")
    print(f"Parameters: {json.dumps(config['model']['parameters'], indent=2)}")
    print("===============================")

if __name__ == "__main__":
    main()