import json
import os

def json_to_jsonl(input_json_path, output_jsonl_path):
    """
    Convert a JSON file to a JSONL file, extracting 'llm' and 'human' fields.
    
    Args:
        input_json_path (str): Path to the input JSON file.
        output_jsonl_path (str): Path to the output JSONL file.
    """
    # 1. Read the JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # 2. Iterate through the data and write to JSONL
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in data_dict:
            # Construct the JSON object for LLM text, label=1
            llm_obj = {
                "text": item['llm'],
                "label": 1
            }
            f.write(json.dumps(llm_obj, ensure_ascii=False) + '\n')
            
            # Construct the JSON object for human text, label=0
            human_obj = {
                "text": item['human'],
                "label": 0
            }
            f.write(json.dumps(human_obj, ensure_ascii=False) + '\n')

# Example usage
if __name__ == "__main__":
    legal_datasets = ['LawStack', 'OALC']
    medical_datasets = ['mimic_discharge', 'pubmedqa']
    models = ['dsr1', 'dsv3', 'gpt4o', 'gpto3mini']
    
    # Define base output directory
    base_output_dir = "../divscore"
    
    for model in models:
        for legal_dataset in legal_datasets:
            INPUT_JSON = f"DivScore/datasets/core/legal/{model}_{legal_dataset}.json"
            OUTPUT_JSONL = f"{base_output_dir}/{model}_{legal_dataset}.jsonl"
            
            if os.path.exists(INPUT_JSON):
                json_to_jsonl(INPUT_JSON, OUTPUT_JSONL)
                print(f"Conversion complete! JSONL file saved to: {OUTPUT_JSONL}")
            else:
                print(f"Warning: Input file {INPUT_JSON} not found. Skipping.")
                
    for model in models:
        for medical_dataset in medical_datasets:
            INPUT_JSON = f"DivScore/datasets/core/medical/{model}_{medical_dataset}.json"
            OUTPUT_JSONL = f"{base_output_dir}/{model}_{medical_dataset}.jsonl"
            
            if os.path.exists(INPUT_JSON):
                json_to_jsonl(INPUT_JSON, OUTPUT_JSONL)
                print(f"Conversion complete! JSONL file saved to: {OUTPUT_JSONL}")
            else:
                print(f"Warning: Input file {INPUT_JSON} not found. Skipping.")