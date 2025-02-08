import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def merge_lora_with_base():
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load the LoRA configuration
    peft_config = PeftConfig.from_pretrained("./final_model")
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, "./final_model")
    
    # Merge LoRA weights with base model
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    merged_model.save_pretrained("./merged_model")
    
    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.save_pretrained("./merged_model")
    
    print("Model successfully merged and saved to ./merged_model")

if __name__ == "__main__":
    merge_lora_with_base() 