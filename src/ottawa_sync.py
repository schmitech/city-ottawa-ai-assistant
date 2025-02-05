import json
import os
import re

def clean_training_data(input_file: str, output_file: str = None) -> None:
    """
    Clean training data by removing 'User:' and 'Assistant:' prefixes and reorganizing the format.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file. If None, overwrites the input file.
    """
    if output_file is None:
        output_file = input_file
        
    # Read the existing training data
    with open(input_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    # Clean the training data
    cleaned_data = []
    for item in training_data:
        # Remove 'User:' and 'Assistant:' prefixes from the output
        output = item['output']
        output = re.sub(r'\n\s*User:\s*.*?\n\s*Assistant:\s*', '\n', output)
        output = re.sub(r'^User:\s*.*?\n\s*Assistant:\s*', '', output)
        
        # Create cleaned item
        cleaned_item = {
            'input': item['input'],
            'output': output.strip()
        }
        cleaned_data.append(cleaned_item)
    
    # Save the cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

def main():
    # Path to the training data
    data_dir = 'ottawa_data'
    training_data_path = os.path.join(data_dir, 'training_data.json')
    
    print("Cleaning training data...")
    clean_training_data(training_data_path)
    print("Training data cleaned successfully!")
    
    print("""
Next steps:
1. Rebuild the custom Mistral model:
   ollama create ottawa-assistant -f ottawa_data/Modelfile
    """)

if __name__ == "__main__":
    main()
