import json
import os
import re
from typing import List, Dict

class JSONLConverter:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.data = []
        # Add validation for directory existence
        if not os.path.exists(raw_data_dir):
            raise ValueError(f"Raw data directory {raw_data_dir} does not exist")

    def load_json_files(self) -> None:
        """Load all JSON files from the raw data directory."""
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.raw_data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.data.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Improved cleaning with regex patterns"""
        if not text:
            return ""
            
        # Remove markdown-style links and special markers
        clean_patterns = [
            r'\(link.*?\)',  # Remove (link...) patterns
            r'\*+',          # Remove asterisks
            r'#+\s*',         # Remove heading markers
            r'Opens in.*$',  # Remove "Opens in..." text
            r'_{2,}',        # Remove underscores
            r'\s+',          # Replace multiple spaces with single
        ]
        
        for pattern in clean_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
            
        return text.strip()

    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Improved with better fee detection and response formatting"""
        qa_pairs = []
        fee_keywords = {'fee', 'cost', 'price', '$', 'charge', 'rate'}

        for data in self.data:
            for section in data.get('sections', []):
                section_title = self._clean_text(section.get('title', ''))
                if not section_title:
                    continue

                for item in section.get('content', []):
                    # Improved content extraction
                    description = self._clean_text(item.get('description', ''))
                    details = [self._clean_text(d) for d in item.get('details', [])]
                    
                    # Filter out empty/non-essential details
                    valid_details = [
                        d for d in details 
                        if d and len(d) > 10 and not d.startswith(('http', 'www.'))
                    ]

                    # Create base response
                    response_parts = []
                    if description:
                        response_parts.append(description)
                    if valid_details:
                        response_parts.extend(valid_details)
                    
                    if not response_parts:
                        continue
                        
                    # Format response with bullet points
                    formatted_response = "According to official City of Ottawa documentation:\n- " + "\n- ".join(response_parts)

                    # Generate question variations
                    base_questions = [
                        f"What is {section_title}?",
                        f"Tell me about {section_title}",
                        f"What should I know about {section_title}?",
                        f"Explain {section_title}",
                        f"Details about {section_title}"
                    ]
                    
                    # Add context-specific questions
                    if any(kw in formatted_response.lower() for kw in fee_keywords):
                        base_questions.extend([
                            f"What are the fees for {section_title}?",
                            f"How much does {section_title} cost?",
                            f"What is the pricing for {section_title}?",
                            f"Cost details for {section_title}"
                        ])

                    # Create QA pairs
                    for question in base_questions:
                        qa_pairs.append({
                            "prompt": question,
                            "response": formatted_response
                        })

        return qa_pairs

    def convert_to_jsonl(self, output_path: str) -> None:
        """Convert the data to JSONL format and save to file."""
        self.load_json_files()
        qa_pairs = self.generate_qa_pairs()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write each QA pair as a JSON line
                for pair in qa_pairs:
                    json_line = json.dumps(pair, ensure_ascii=False)
                    f.write(json_line + '\n')
                    
            print(f"Successfully generated {len(qa_pairs)} training examples at {output_path}")
            
        except Exception as e:
            print(f"Error writing JSONL file: {str(e)}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, 'rawdata')
    output_path = os.path.join(script_dir, 'training_data.jsonl')
    
    converter = JSONLConverter(raw_data_dir)
    converter.convert_to_jsonl(output_path)

if __name__ == "__main__":
    main()