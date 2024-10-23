import dataclasses
import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import DPOConfig

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
    
    hf_key: str = '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'
    sft_model_name: str = "meta-llama/Llama-2-7b-hf"
    sft_dataset_path: str = "./data/text_completion_dataset.csv"
    sft_output_dir: str = "/home/ubuntu/huggingface/sft_models"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
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
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
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
    
    hf_key: str = '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'
    sft_model_name: str = "meta-llama/Llama-2-7b-hf"
    dpo_dataset_path: str = "./data/text_completion_dataset.csv"
    sft_adapter_path: str = "/home/ubuntu/huggingface/sft_models"
    dpo_output_dir: str = "/home/ubuntu/huggingface/dpo_models"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    
    train_test_split_ratio: float = 0.2
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    training_args = DPOConfig(output_dir=dpo_output_dir, 
                              per_device_train_batch_size=2,
                              per_device_eval_batch_size=2,
                              num_train_epochs=50,
                              logging_steps=10,
                              learning_rate=2e-4,
                              eval_strategy="epoch",
                              eval_steps=10,
                              bf16=True,
                              lr_scheduler_type='cosine',
                              warmup_steps=5,
                              )
    
    
    generate_max_length = 64