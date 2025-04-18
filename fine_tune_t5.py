import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
import nltk

# --- Load Dataset ---
def preprocess(tweet_series):
    stopwords = nltk.corpus.stopwords.words('english')
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()
    tweet_series = tweet_series.astype(str)

    tweet_series = tweet_series.str.replace(r'\s+', ' ', regex=True)
    tweet_series = tweet_series.str.replace(r'@[\w\-]+', '', regex=True)
    tweet_series = tweet_series.str.replace(r'http[s]?://\S+', '', regex=True)

    tweet_series = tweet_series.str.replace(r'[^a-zA-Z]', ' ', regex=True)
    tweet_series = tweet_series.str.replace(r'\s+', ' ', regex=True)
    tweet_series = tweet_series.str.replace(r'^\s+|\s+?$', '', regex=True)
    tweet_series = tweet_series.str.replace(r'\d+(\.\d+)?', 'numbr', regex=True)
    tweet_series = tweet_series.str.lower()

    tokenized = tweet_series.apply(lambda x: x.split())
    # tokenized = tokenized.apply(lambda x: [w for w in x if w not in stopwords])
    # tokenized = tokenized.apply(lambda x: [stemmer.stem(w) for w in x])
    cleaned = tokenized.apply(lambda x: ' '.join(x))
    return cleaned

dataset = pd.read_csv('data.csv')
dataset["original_comment"] = preprocess(dataset["original_comment"])
dataset["non_offensive_comment"] = preprocess(dataset["non_offensive_comment"])

hate_neutral, hate_neutral_eval = train_test_split(dataset, test_size=0.2, random_state=42)



dataset = hate_neutral.apply(lambda row: {
    "original": row["original_comment"],
    "neutral": row["non_offensive_comment"]
}, axis=1).tolist()

dataset_eval = hate_neutral_eval.apply(lambda row: {
    "original": row["original_comment"],
    "neutral": row["non_offensive_comment"]
}, axis=1).tolist()

def format_example(example):
    previous_tweet = example["original"]
    return {
        "input_text": f"neutralize: {previous_tweet}",
        "label_text": example["neutral"]
    }
dataset = [format_example(example) for example in dataset]
print("[INFO] Formatted dataset size:", len(dataset))

# --- Tokenizer and Model ---
model_name = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("[INFO] Model loaded:", model_name)

# --- Apply LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("[INFO] LoRA applied")

# --- Tokenize Dataset ---
def tokenize(example):
    inputs = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
    targets = tokenizer(
        example["label_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = targets["input_ids"]
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    inputs["labels"] = labels
    return inputs


dataset = Dataset.from_pandas(pd.DataFrame(dataset))
tokenized_dataset = dataset.map(tokenize, batched=False)
print("[INFO] Tokenized dataset size:", len(tokenized_dataset))

# --- Training ---
training_args = TrainingArguments(
    output_dir="lora-t5-small",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_dir="logs",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# --- Save the model and tokenizer ---
model.save_pretrained("lora-t5small-hate-neutral")
tokenizer.save_pretrained("lora-t5small-hate-neutral")