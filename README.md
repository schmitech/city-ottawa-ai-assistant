# City of Ottawa AI Assistant

This project creates an AI-powered assistant for the City of Ottawa services using Mistral and fine-tuning. The assistant helps citizens quickly find information about city services, business licenses, and recreational programs.

## Architecture Overview

The project uses:
- Mistral as the base model
- LoRA for efficient fine-tuning
- Ollama for model serving
- Training data focused on city services

## Features

- Fine-tuned model for Ottawa city services
- Accurate responses about:
  - Business licenses and fees
  - Sports and fitness programs
  - Older adult activities
- Professional response formatting
- Source references in responses

## Prerequisites

- Python 3.11 or higher
- Ollama installed ([Ollama Installation Guide](https://github.com/jmorganca/ollama))
- 16GB RAM minimum

## Project Structure

```
city-ottawa-ai-assistant/
├── train_model.py        # Training script
├── merge_model.py        # Merge LoRA weights
├── convert_to_onnx.py    # Model conversion
├── Modelfile            # Ollama configuration
├── requirements.txt     # Project dependencies
└── README.md           # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/city-ottawa-ai-assistant.git
cd city-ottawa-ai-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Training and Deployment

1. Train the model:
```bash
python train_model.py
```

2. Merge the LoRA weights with base model:
```bash
python merge_model.py
```

3. Convert the model:
```bash
python convert_to_onnx.py
```

4. Create Ollama model:
```bash
ollama create ottawa-services -f Modelfile
```

## Using the Model

1. Start a conversation:
```bash
ollama run ottawa-services "What is Business licences?"
```

2. Run in chat mode:
```bash
ollama run ottawa-services
```

## Model Configuration

The Modelfile contains:
- Base model configuration (using Mistral)
- Temperature and other generation parameters
- Prompt templates
- System instructions
- Example conversations for few-shot learning

Example Modelfile structure:
```modelfile
FROM mistral:latest

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Prompt template
TEMPLATE """### Instruction: {{ .Prompt }}

### Response: """

# System instructions and example conversations
SYSTEM """You are an AI assistant trained to help with information about City of Ottawa services..."""
```

## Sample Queries

Try these example queries:

```
# Business Licenses
"How do I apply for a business licence in Ottawa?"
"What is the cost of a kennel licence?"
"What businesses require municipal licenses?"

# Recreation Programs
"What is the Heart Wise Exercise program?"
"How much do tennis courts cost?"
"How can I get a free fitness pass?"
```

## Limitations

- Limited to trained topics (business licenses, recreation programs)
- Requires Ollama to be installed
- CPU-only training and inference
- Response time varies based on hardware

## Troubleshooting

Common issues and solutions:
1. Memory errors during training:
   - Reduce batch size in train_model.py
   - Close other applications
   - Use CPU-only training mode

2. Conversion issues:
   - Ensure all dependencies are installed
   - Check available disk space
   - Verify model files are complete

3. Ollama integration:
   - Verify Ollama is running
   - Check Modelfile syntax
   - Ensure base model is downloaded

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://github.com/jmorganca/ollama) for the model serving infrastructure
- Mistral AI for the base model

## Support

For support, please open an issue in the GitHub repository.