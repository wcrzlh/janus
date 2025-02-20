from datasets import load_dataset
from janus.models import MultiModalityCausalLM
from mindone.transformers import AutoTokenizer, Trainer, TrainingArguments

# Load dataset (this is the PubMedQA dataset)
dataset = load_dataset("pubmed_qa", "pqa_m")

# Load a pretrained transformer model
model_name = "janus-pro-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MultiModalityCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_data(example):
    return tokenizer(example['question'], example['abstract'], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Split the data into train and validation
train_dataset = tokenized_dataset['train']
val_dataset = tokenized_dataset['validation']

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start the training process
trainer.train()

# Save the model after training
trainer.save_model("pubmed_qa_model")