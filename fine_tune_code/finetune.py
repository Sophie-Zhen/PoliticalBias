import json
import os
import pandas as pd
import pickle
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
# Warnings
import warnings
warnings.filterwarnings("ignore")

# read file and make dataset
def covert_ds(file_path):
    df = pd.read_json(file_path)
    df_relevant = df[['title', 'text']]
    df_relevant.loc[:, 'text'] = df_relevant['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    # convert to hugging face dataset
    ds = Dataset.from_pandas(df_relevant)
    return ds

# preprocess: tokenize
def preprocess_news(ds, base_name, tokenized_file_path):
    tokenizer = AutoTokenizer.from_pretrained("Your Hugging Face Model", use_auth_token='Your HuggingFace Access Token')
    try:
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Define the preprocessing function
        def preprocess_function(examples):
            # Tokenize a batch of examples
            combined_texts = [f"Title: {title} Text: {text}" for title, text in zip(examples['title'], examples['text'])]
            tokenized = tokenizer(
                combined_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()  # Use input_ids as labels
            return tokenized
        
        # Use smaller batch size to avoid memory issues
        tokenized_ds = ds.map(preprocess_function, batched=True, batch_size=16)
        # save to file
        with open(tokenized_file_path, 'wb') as f:
            pickle.dump(tokenized_ds, f)
        print(f"Tokenized dataset saved to: {tokenized_file_path}")
        return tokenized_ds
    except Exception as e:
        print(f"Error tokenizing dataset {base_name}: {e}")
        return None

def train_model(tokenized_ds, output_dir="results",model_checkpoint="Your Hugging Face Model"):

    # load model
    model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto",  # Automatically places model layers across devices
    torch_dtype=torch.float32,  # Ensure appropriate tensor type
    offload_folder="offload",  # Temporary folder for offloaded layers
    # offload_state_dict=True,  # Ensures the state dict is offloaded to disk
    use_auth_token='Your Hugging Face Model'
    )
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")
    print(f"Model loaded on device: {next(model.parameters()).device}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token='Your Hugging Face Access Tokens')

    model = prepare_model_for_kbit_training(model)
    model = model.to_empty(device='mps') #replace 'mps' with 'cuda' or 'cpu' as needed

    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")
    # LoRA configuration
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["c_attn"],#replace "c_attn" with the value of your model, "c_attn" is for gpt-2
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # split into train and eval set
    train_eval_split = tokenized_ds.train_test_split(test_size=0.1, shuffle=True, seed = 42)
    train_ds = train_eval_split["train"]
    eval_ds = train_eval_split["test"]
    
    # add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ensure the model is resized if new tokens are added
    model.resize_token_embeddings(len(tokenizer))

    # setting args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        logging_dir="./logs",
        logging_steps=10,
        use_mps_device=True,
        no_cuda=True,#if you use cuda, comment it
        optim="adamw_torch",
        skip_memory_metrics=True,  # Prevent memory tracking issues
    )

    # initialize Trainer    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # Training dataset
        eval_dataset=eval_ds,
    ) 
    # train
    trainer.train()
    # save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')

# directory to the folder
news_dir = "path to training dataset"
intermediate_dir = 'path to store tokenized dataset'# avoiding repeat token
# list of json files
file_list = [f for f in os.listdir(news_dir)]

# process each file iteratively
for file_name in file_list:
    file_path = os.path.join(news_dir, file_name)
    base_name = file_name.split('.')[0]
    print(f"Processing file: {base_name}")
    tokenized_file_path = os.path.join(intermediate_dir,f"tokenized_{base_name}.pkl")

    if os.path.exists(tokenized_file_path):
        print(f"Tokenized dataset already exists: {tokenized_file_path}")
        with open(tokenized_file_path, "rb") as f:
            tokenized_ds =  pickle.load(f)
    else:
        ds = covert_ds(file_path)
        print(f"convert to ds done: {base_name}")
        tokenized_ds = preprocess_news(ds, base_name, tokenized_file_path)
        print(f"tokenize ds done: {base_name}")
    train_model(tokenized_ds, output_dir=f"results_{base_name}")