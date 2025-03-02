# optionally save the logits on a specific set of tokens e.g., 0 or 1
# evaluate both the base model and our fine-tuned model 
# save the results in a json file

from transformers import LlamaForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import os
import einops 
import gc

from src.model_utils import model_paths

def load_model(model_path, device='cpu'):
    """Load model and tokenizer, optionally moving to CPU"""
    if device == 'cpu':
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            torch_dtype=torch.float32  # Use float32 for CPU
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map=device
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Add padding to the left side
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset='train_data/test.json', batch_size=64, device='cpu'):
    """
    Evaluate model on dataset using batch inference with aggressive memory clearing
    """
    with open(dataset, 'r') as f:
        data = json.load(f)[0:200]
    
    predictions = []
    all_logits = []
    all_logit_values = []
    labels = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        
        # Prepare batch
        if 'chat' in dataset:
            prompts = [[{'role': 'user', 'content': item['conversations'][0]['value']}] for item in batch]
            batch_labels = [item['conversations'][1]['value'] for item in batch]
        else:
            prompts = [item['prompt'] for item in batch]
            batch_labels = [item['label'] for item in batch]
        
        # Tokenize all prompts in batch
        if 'chat' in dataset:
            inputs = tokenizer.apply_chat_template(prompts, padding=True, tokenize=True, return_tensors='pt', add_generation_prompt=True)
            print(tokenizer.decode(inputs[0]))
            if device != 'cpu':
                inputs = inputs.to(device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    inputs,  # For chat, inputs is already the input_ids tensor
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, 
                    output_logits=True,
                    return_dict_in_generate=True
                )
        else:
            inputs = tokenizer(prompts, padding=True, return_tensors='pt')
            if device != 'cpu':
                inputs = inputs.to(device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, 
                    output_logits=True,
                    return_dict_in_generate=True
                )
            
        # Move tensors to CPU immediately
        logits = outputs.logits
        logits = torch.stack(logits)
        logits = einops.rearrange(logits, 's b h -> b s h')
        tokens = logits.argmax(dim=-1)
        top_logits = logits.topk(k=5, dim=-1).indices.cpu().tolist()
        top_logit_values = logits.topk(k=5, dim=-1).values.cpu().tolist()
        
        # Convert to CPU and lists before deletion
        # batch_logits = logits.cpu().tolist()
        tokens = tokens.cpu()
        batch_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in tokens]
        
        # Extend results
        predictions.extend(batch_preds)
        labels.extend(batch_labels)
        all_logits.extend(top_logits)
        all_logit_values.extend(top_logit_values)
        # Clear memory
        del outputs
        del logits
        del tokens
        del inputs
            
        # Clear CUDA cache
        if device != 'cpu':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print examples
        for pred, label in zip(batch_preds[:1], batch_labels[:1]):
            print(f"Label: {label}")
            print(f"Prediction: {pred}")
            print("===============================")
    
    return predictions, labels, all_logits, all_logit_values

def save_results(predictions, labels, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"predictions": predictions, "labels": labels}, f)

def main(model_name, overwrite=False):
    save_dir = f'train_data/predictions/{model_name}'
    if not os.path.exists(save_dir) or overwrite:
        model_path = model_paths[model_name]
        model, tokenizer = load_model(model_path, device='cuda')
        if 'instruct' in model_name:
            predictions, labels, logits, logit_values = evaluate_model(model, tokenizer, 'train_data/test_chat.json', batch_size=16, device='cuda')
        else:
            predictions, labels, logits, logit_values = evaluate_model(model, tokenizer, 'train_data/test.json', batch_size=16, device='cuda')
        os.makedirs(save_dir, exist_ok=True)
        save_results(predictions, labels, f'{save_dir}/test.json')
        save_results(logits, labels, f'{save_dir}/test_logits.json')
        save_results(logit_values, labels, f'{save_dir}/test_logit_values.json')

if __name__ == "__main__":
    #model_names = 'llama-3.1-8b', 'llama-3.2-1b', 'qlora-llama-3.1-8b', 'qlora-llama-3.2-1b'
    model_names = model_paths.keys()
    for model_name in model_names:
        # if 'qlora' not in model_name:
        #     continue
        print(f"\nEvaluating model: {model_name}")
        main(model_name, overwrite=True)
        print(f"Finished evaluating: {model_name}\n")
