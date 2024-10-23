import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from config import MyDPOConfig

def load_model(config: MyDPOConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 quantization_config=config.bnb_config,
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_inference(config: MyDPOConfig, input_text: str):
    model, tokenizer = load_model(config)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def main():
    parser = argparse.ArgumentParser(description="Run inference with a given input text.")
    parser.add_argument('-i', '--input_text', type=str, required=True, help='Input text for inference')
    args = parser.parse_args()
    
    config = MyDPOConfig()
    
    output_text = run_inference(config, args.input_text)
    print(output_text)

if __name__ == "__main__":
    main()