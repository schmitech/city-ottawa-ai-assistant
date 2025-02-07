import os
import yaml
import json
from typing import List, Dict, Any
from collections import defaultdict

def escape_string(s: str) -> str:
    """Escape special characters in strings for Modelfile."""
    # First escape any existing quotes and backslashes
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    # Remove any newlines and extra whitespace while preserving list formatting
    lines = s.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    return ' '.join(cleaned_lines)

def structure_content(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Structure content into Q&A pairs with context."""
    qa_pairs = []
    
    # Add title-based general questions only if we have content
    title = data.get('title', '')
    all_content = []
    
    # Process each section
    for section in data.get('sections', []):
        section_title = section.get('title', '')
        section_content = []
        fee_content = []
        
        for content in section.get('content', []):
            # Add description
            if content.get('description'):
                desc = content['description']
                section_content.append(desc)
                all_content.append(desc)
                # Check for fee-related content
                if any(term in desc.lower() for term in ['fee', 'cost', 'price', 'rate', '$', 'dollar', 'payment']):
                    fee_content.append(desc)
            
            # Add details with proper formatting
            if content.get('details'):
                for detail in content['details']:
                    if detail.startswith('Heading:'):
                        current_heading = detail[8:]  # Remove 'Heading:' prefix
                        section_content.append(f"\n{current_heading}")
                        all_content.append(current_heading)
                        # Check if heading is fee-related
                        if any(term in current_heading.lower() for term in ['fee', 'cost', 'price', 'rate', 'payment']):
                            fee_content.append(current_heading)
                    else:
                        section_content.append(f"- {detail}")
                        all_content.append(detail)
                        # Check for fee-related content
                        if any(term in detail.lower() for term in ['fee', 'cost', 'price', 'rate', '$', 'dollar', 'payment']):
                            fee_content.append(detail)
        
        # Only create Q&A pairs if we have actual content
        if section_content:
            # Create section overview Q&A
            section_text = "\n".join(section_content)
            if section_text.strip():  # Only add if we have non-empty content
                qa_pairs.append({
                    'instruction': f"Explain {section_title}",
                    'response': f"{section_title}:\n{section_text}",
                    'context': f"{title} - {section_title}"
                })
                
                qa_pairs.append({
                    'instruction': f"What are the rules for {section_title.lower()}?",
                    'response': section_text,
                    'context': f"{title} - {section_title}"
                })
            
            # Create fee-specific questions only if we have fee content
            if fee_content:
                fee_text = "\n".join(fee_content)
                if fee_text.strip():  # Only add if we have non-empty fee content
                    fee_questions = [
                        f"What are the fees for {section_title.lower()}?",
                        f"How much does {section_title.lower()} cost?",
                        f"What are the rates for {section_title.lower()}?",
                    ]
                    for question in fee_questions:
                        qa_pairs.append({
                            'instruction': question,
                            'response': f"Here are the fees and rates:\n{fee_text}",
                            'context': f"{title} - {section_title} Fees"
                        })
                    
                    # Add senior-specific fee questions if relevant
                    if any(term in title.lower() or term in section_title.lower() 
                          for term in ['senior', 'older adult', 'elderly']):
                        qa_pairs.append({
                            'instruction': "What are the fees for seniors?",
                            'response': f"Here are the senior rates and fees:\n{fee_text}",
                            'context': f"{title} - Senior Rates"
                        })
            
            # Add user group specific questions only if we have relevant content
            user_groups = ['seniors', 'students', 'adults', 'children', 'families']
            for group in user_groups:
                group_content = [c for c in section_content if group in c.lower()]
                if group_content:  # Only add if we have content for this group
                    group_text = "\n".join(group_content)
                    if group_text.strip():  # Double check for non-empty content
                        qa_pairs.append({
                            'instruction': f"What are the details for {group}?",
                            'response': f"Here are the details for {group}:\n{group_text}",
                            'context': f"{title} - {group.capitalize()} Information"
                        })
    
    # Add title-based overview only if we have content
    if all_content:
        overview_text = "\n".join(all_content)
        if overview_text.strip():
            qa_pairs.append({
                'instruction': f"What are the key points about {title}?",
                'response': f"Here are the key points about {title}:\n{overview_text}",
                'context': title
            })

    return qa_pairs

def load_training_data(rawdata_dir: str = "rawdata") -> List[Dict[str, str]]:
    """Load and structure training data from JSON files"""
    all_qa_pairs = []
    
    for filename in os.listdir(rawdata_dir):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(rawdata_dir, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                qa_pairs = structure_content(data)
                all_qa_pairs.extend(qa_pairs)
            except json.JSONDecodeError as e:
                print(f"Error parsing {filename}: {e}")
                continue
    
    return all_qa_pairs

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
        "",
        "# Set parameters for model behavior",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER top_k 40",
        "PARAMETER num_ctx 4096",
        "PARAMETER repeat_penalty 1.1",
        "PARAMETER stop \"<|im_start|>\"",
        "PARAMETER stop \"<|im_end|>\"",
        "",
        "# Set template for chat interactions",
        'TEMPLATE """{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{- end }}'
        '{{- if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{- end }}'
        '<|im_start|>assistant\n{{ .Response }}<|im_end|>"""',
        ""
    ]
    
    # Read system prompt from gemma2.txt
    try:
        with open('model-files/gemma2.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read()
            # Extract the actual SYSTEM prompt content between the triple quotes
            if '"""' in system_prompt:
                system_prompt = system_prompt.split('"""')[1].strip()
            modelfile_content.extend([
                "# Set system prompt",
                f'SYSTEM """{system_prompt}"""',
                ""
            ])
    except FileNotFoundError:
        print("Warning: gemma2.txt not found, using default system prompt")
        modelfile_content.append('SYSTEM """You are an AI expert on Ottawa municipal services..."""\n')
    
    # Load and add training examples
    try:
        training_data = load_training_data("rawdata")
        
        modelfile_content.append("# Training examples")
        # First add some basic context-setting examples
        modelfile_content.extend([
            'MESSAGE user "Do you have access to information about City of Ottawa services?"',
            'MESSAGE assistant "Yes, I have access to official City of Ottawa information about municipal services, including programs, fees, and regulations. I can provide specific details from ottawa.ca about services, rates, and requirements. How can I help you?"',
            "",
            'MESSAGE user "Should I verify the information you provide?"',
            'MESSAGE assistant "Yes, while I provide information from official City of Ottawa sources, it\'s always good practice to verify current details, especially for time-sensitive matters like fees or schedules, at ottawa.ca or by contacting 3-1-1."',
            ""
        ])
        
        # Add conversation examples with proper formatting
        for example in training_data:
            instruction = escape_string(example["instruction"])
            response = escape_string(example["response"])
            context = example.get("context", "City of Ottawa regulations")
            
            # Format response to acknowledge the source but be more natural
            formatted_response = f"According to {context}: {response}"
            
            modelfile_content.extend([
                f'MESSAGE user "{instruction}"',
                f'MESSAGE assistant "{formatted_response}"',
                ""
            ])
    
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")
        return
    
    # Write the Modelfile
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modelfile_content))
    
    print(f"Created Modelfile at: {modelfile_path}")
    print(f"Total training examples: {len(training_data)}")

def main():
    config_path = "config.yaml"
    create_modelfile(config_path)

if __name__ == "__main__":
    main()