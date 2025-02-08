from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def convert_to_gguf():
    print("Starting conversion process...")
    
    # Check if merged model exists
    if not os.path.exists("./merged_model"):
        print("Error: merged_model directory not found. Run merge_model.py first.")
        return
    
    try:
        print("Converting model to GGUF format...")
        os.system("python /Users/remsyschmilinsky/Downloads/llama.cpp/convert_hf_to_gguf.py --outfile ottawa_services.gguf --outtype q8_0 ./merged_model")
        
        print("Model converted successfully!")
        print("Model saved as: ottawa_services.gguf")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    convert_to_gguf() 