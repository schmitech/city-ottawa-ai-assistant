import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import pandas as pd
from tqdm import tqdm
import re
import concurrent.futures
import os
from typing import List, Dict, Set
import tiktoken
import hashlib
import urllib3
import time
import yaml
from queue import PriorityQueue
import random
from urllib.robotparser import RobotFileParser

def create_modelfile(config_path="config.yaml") -> None:
    """Create a Modelfile for an Ollama custom model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    modelfile_path = os.path.join(output_dir, config['output']['files']['modelfile'])

    # Build base configuration from config
    base_config = f'''FROM {config['model']['base_model']}
PARAMETER num_ctx {config['model']['parameters']['num_ctx']}
ADAPTER ./{config['output']['directory']}/{config['output']['files']['training']}\n\n'''

    # Load model instructions from file
    instruction_file = config['model']['instruction_file']
    try:
        with open(instruction_file, 'r', encoding='utf-8') as f:
            model_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Model instruction file not found: {instruction_file}")
        return

    # Load and process training data
    training_file = os.path.join(output_dir, config['output']['files']['training'])
    training_examples = []
    try:
        with open(training_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
            training_examples = [
                f'TEMPLATE "{example["instruction"]}"\nMESSAGE "{example["response"]}"'
                for example in training_data
            ]
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")

    # Build the complete Modelfile
    modelfile_content = f'''{base_config}
# System instructions
{model_instructions}

# Training examples
{"\n\n".join(training_examples)}
'''
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

def main():
    config_path = "config.yaml"
    
    print("Creating Modelfile...")
    create_modelfile(config_path)
    
    # Add final model confirmation with error handling
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n=== Training Configuration ===")
    print(f"Base Model: {config['model']['base_model']}")
    print(f"Instruction File: {config['model']['instruction_file']}")
    print(f"Parameters: {json.dumps(config['model']['parameters'], indent=2)}")
    print("===============================")

if __name__ == "__main__":
    main()