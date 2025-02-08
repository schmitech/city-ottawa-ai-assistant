import torch
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def convert_model():
    print("Starting conversion process...")
    
    if not Path("./merged_model").exists():
        print("Error: merged_model directory not found. Run merge_model.py first.")
        return
        
    try:
        print("Converting model to ONNX format...")
        # Load and convert model
        ort_model = ORTModelForCausalLM.from_pretrained(
            "./merged_model",
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./merged_model")
        
        # Save converted model
        output_path = Path("./onnx_model")
        ort_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("Model converted successfully!")
        print("Model saved to:", output_path)
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_model() 