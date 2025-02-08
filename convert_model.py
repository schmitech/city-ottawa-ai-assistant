import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import os
import shutil

def convert_model():
    print("Starting conversion process...")
    
    if not os.path.exists("./merged_model"):
        print("Error: merged_model directory not found. Run merge_model.py first.")
        return
        
    try:
        print("Loading OPT model...")
        opt_model = AutoModelForCausalLM.from_pretrained(
            "./merged_model",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        opt_tokenizer = AutoTokenizer.from_pretrained("./merged_model")
        
        print("Preparing for conversion...")
        # Create export directory
        export_path = Path("./export_model")
        export_path.mkdir(exist_ok=True)
        
        # Update the config for compatibility
        config = opt_model.config
        config.architectures = ["OPTForCausalLM"]
        config.model_type = "opt"
        
        # Add required attributes if they don't exist
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = config.num_attention_heads
        
        print("Saving model...")
        opt_model.save_pretrained(
            export_path,
            max_shard_size="2GB",
            safe_serialization=True
        )
        opt_tokenizer.save_pretrained(export_path)
        
        print("Model exported successfully.")
        print("Now you can convert to GGUF using:")
        print(f"python llama.cpp/convert_hf_to_gguf.py --outfile ottawa_services.gguf --outtype q8_0 {export_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_model() 