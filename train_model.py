import json
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer
import torch
from peft import LoraConfig, get_peft_model

class PrintCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            try:
                if state.log_history:
                    loss = state.log_history[-1].get('loss', 'N/A')
                    print(f"Step {state.global_step}: Loss = {loss}")
                else:
                    print(f"Step {state.global_step}: No loss data available yet")
            except Exception as e:
                print(f"Step {state.global_step}: Unable to get loss")

def load_jsonl_data(file_path):
    """Load and format data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            formatted_text = f"""### Instruction: {example['prompt']}

### Response: {example['response']}

### End"""
            data.append({
                "text": formatted_text
            })
    return data

def prepare_dataset():
    """Prepare the dataset for training."""
    data = load_jsonl_data("./rawdata/training_data_cleaned.jsonl")
    dataset = Dataset.from_list(data)
    return dataset

def train():
    """Initialize and train the model."""
    model_name = "facebook/opt-350m"
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    dataset = prepare_dataset()
    
    training_args = TrainingArguments(
        output_dir="./model_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=100,
        report_to="none",
        log_level="info",
        logging_first_step=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    
    trainer.add_callback(PrintCallback())
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("./final_model")

if __name__ == "__main__":
    print("Running on CPU - training will be slow")
    train() 