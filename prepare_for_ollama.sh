#!/bin/bash

# Install required packages
pip install 'transformers[sentencepiece]' optimum[onnxruntime]

# Merge LoRA with base model
echo "Merging LoRA weights with base model..."
python merge_model.py

# Convert model to ONNX format
echo "Converting model to ONNX format..."
python convert_to_onnx.py

# Create Ollama model
echo "Creating Ollama model..."
ollama create ottawa-services -f Modelfile

echo "Done! You can now use the model with: ollama run ottawa-services" 