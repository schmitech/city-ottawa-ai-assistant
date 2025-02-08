from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llama_cpp import Llama
import os

def convert_to_gguf():
    print("Starting conversion process...")
    
    # Check if merged model exists
    if not os.path.exists("./merged_model"):
        print("Error: merged_model directory not found. Run merge_model.py first.")
        return
    
    try:
        # Load the merged model
        print("Loading merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            "./merged_model",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("./merged_model")
        
        # Save in format compatible with llama.cpp
        print("Saving model in llama.cpp format...")
        model.save_pretrained(
            "./llama_model",
            max_shard_size="2GB",
            safe_serialization=True
        )
        tokenizer.save_pretrained("./llama_model")
        
        print("Model saved. You can now use llama.cpp tools to convert to GGUF format.")
        print("Next steps:")
        print("1. Install llama.cpp")
        print("2. Run: python -m llama_cpp.convert_hf_to_gguf ./llama_model ./ottawa_services.gguf --model-type opt")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    convert_to_gguf() 