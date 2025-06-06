'''
note:
this is for learning purpose. we didn't use the code in this file to finetune
the model. we used the services offered by the replicate website/api instead.
'''

# Prepare dataset
import json
import pandas as pd
from datasets import Dataset

# Load JSONL data
data = []
with open('data/csvjson.json', 'r',encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
token = "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"

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/gpt2-vietnamese", token=token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("NlpHUST/gpt2-vietnamese", token=token)

# Add a padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=225)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset into train and validation sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']





# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,            # LoRA rank
    lora_alpha=16,  # LoRA alpha
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced batch size
    gradient_accumulation_steps=8,  # Simulate larger batch size
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True, # Enable mixed precision training
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()



# Save the fine-tuned model
model.save_pretrained("model/lora-finetuned-llama2-7b")
tokenizer.save_pretrained("token/lora-finetuned-llama2-7b")
