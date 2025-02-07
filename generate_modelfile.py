import json
import argparse
import os

def generate_modelfile(jsonl_path, output_path):
    """
    Reads a JSONL file with training data and generates a Modelfile for Ollama.
    """
    modelfile_content = """FROM gemma2:2b

# Define system instructions (Optimized for better response handling)
SYSTEM "You are an AI assistant trained to answer questions based on official City of Ottawa documentation. Provide accurate and concise responses. Use the provided training data to infer answers even if they are phrased slightly differently. If a question is entirely outside the scope of the training data, respond with 'I don't have that information.'"

"""
    
    with open(jsonl_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                entry = json.loads(line)
                if "prompt" in entry and "response" in entry:
                    modelfile_content += f'MESSAGE user {json.dumps(entry["prompt"])}\n'
                    modelfile_content += f'MESSAGE assistant {json.dumps(entry["response"])}\n'
                else:
                    print(f"Skipping line {line_number}: Missing 'prompt' or 'response'")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")

    with open(output_path, "w") as output_file:
        output_file.write(modelfile_content)

    print(f"Modelfile generated successfully at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Modelfile from a JSONL file.")
    parser.add_argument("jsonl_path", help="Path to the JSONL file containing training data.")
    args = parser.parse_args()

    output_path = os.path.join(os.getcwd(), "Modelfile")
    generate_modelfile(args.jsonl_path, output_path)

# Example usage:
# generate_modelfile("training_data.jsonl", "Modelfile")
