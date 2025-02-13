import dataclasses
import os
import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import DPOConfig


os.environ["WANDB_DISABLED"] = "true"
 
@dataclasses.dataclass
class SFTConfig:
    """
    Config for supervised fine-tuning
    hf_key:                 your huggingface key
    sft_model_name:         model name in the hf form of "organization/model_name"
    sft_dataset_path:       local path to the dataset
    sft_output_dir:         path where to save the fine-tuned model adapter
    sft_model_cache_dir:    path to cache the model so hf doesnt download it every time
    """
    
    hf_key: str = os.getenv('HF_KEY')
    sft_model_name: str = "facebook/opt-1.3b" #"facebook/opt-1.3b" # nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    sft_dataset_path: str = "./data/imdb_sentiment_data.csv"
    sft_output_dir: str = "./sft_models"
    sft_model_cache_dir: str = "/shares/bcs516/ryan/huggingface/hub/"
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
        
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing =False,
        max_grad_norm= 0.3,
        num_train_epochs=1, 
        save_steps= 100,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir='./sft_models',
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="none",
    )
    
    generate_max_length: int = 64

@dataclasses.dataclass
class MyDPOConfig:
    """
    Config for direct preference optimization
    hf_key:                 your huggingface key
    sft_model_name:         model name in the hf form of "organization/model_name" should be the same as the one used for SFT
    dpo_dataset_path:       local path to the dataset
    sft_adapter_path:       path to the adapter for the SFT tuned adapter
    dpo_output_dir:         path where to save the adapter of the DPO model
    sft_model_cache_dir:    path to cache the model so hf doesnt download it every time
    """
    
    hf_key: str = os.getenv('HF_KEY')
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B" #"facebook/opt-1.3b" # nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    dpo_dataset_path: str = "./data/imdb_sentiment_data.csv"
    sft_adapter_path: str = "./sft_models"
    dpo_output_dir: str = "./dpo_models"
    sft_model_cache_dir: str = "/shares/bcs516/ryan/huggingface/hub/"
    
    train_test_split_ratio: float = 0.2
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    training_args = DPOConfig(output_dir=dpo_output_dir, 
                              per_device_train_batch_size=2,
                              per_device_eval_batch_size=2,
                              num_train_epochs=10,
                              logging_steps=10,
                              learning_rate=2e-4,
                              eval_strategy="epoch",
                              eval_steps=10,
                              bf16=True,
                              lr_scheduler_type='cosine',
                              warmup_steps=5,
                              )
    
    
    generate_max_length = 64
