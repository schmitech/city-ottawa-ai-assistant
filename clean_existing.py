import json
import os

def clean_training_data(input_file: str, output_file: str):
    """Clean training data by removing low-quality entries."""
    cleaned_examples = []
    
    def is_valid_example(example: Dict[str, str]) -> bool:
        """Check if an example meets quality criteria."""
        # Check if response is too short
        if len(example['response'].strip()) < 20:
            return False
            
        # Check if response contains incomplete sentences or lists
        if example['response'].strip().endswith(('...', ',')):
            return False
            
        # Check if response contains too many links
        if example['response'].count('(link is external)') > 3:
            return False
            
        # Check if response is mostly repeated content
        lines = example['response'].split('\n')
        unique_lines = set(lines)
        if len(lines) > 10 and len(unique_lines) < len(lines) * 0.5:
            return False
            
        # Check if response contains actual content (not just navigation elements)
        if all(line.startswith(('Browse ', 'View ', 'Register ')) for line in lines if line.strip()):
            return False
            
        return True

    try:
        # Read existing examples
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if is_valid_example(example):
                        cleaned_examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # Write cleaned examples
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in cleaned_examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Cleaned training data: {len(cleaned_examples)} examples retained")
        
    except Exception as e:
        print(f"Error cleaning training data: {str(e)}")

if __name__ == "__main__":
    input_file = "./rawdata/training_data.jsonl"
    output_file = "./rawdata/training_data_cleaned.jsonl"
    clean_training_data(input_file, output_file) 