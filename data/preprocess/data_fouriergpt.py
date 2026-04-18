import json
import os

# models = ['chatgpt', 'cohere', 'cohere-chat', 'gpt2', 'gpt3', 'gpt4', 'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat', 'human']
# attacks = ['alternative_spelling', 'article_deletion', 'homoglyph', 'insert_paragraphs', 'none', 'number', 'paraphrase', 'synonym', 'upper_lower', 'whitespace', 'zero_width_space']

# model_evobench = [
#     "gpt-4o-2024-05-13",
#     "gpt-4o-2024-08-06",
#     "gpt-4o-2024-11-20",
#     "chatgpt-4o-latest",
#     "gpt-4o-mini-2024-07-18",
#     "gpt-4-1106-preview",
#     "gpt-4-0125-preview",
#     "gpt-4-turbo-2024-04-09",
#     "claude-3-sonnet-20240229",
#     "claude-3-5-sonnet-20240620",
#     "claude-3-5-sonnet-20241022",
#     "Qwen1.5-7B-Chat",
#     "Qwen2-7B-Instruct",
#     "Qwen2.5-7B-Instruct",
#     "Meta-Llama-3.1-8B-Instruct",
#     "Meta-Llama-3.1-70B-Instruct",
#     "Meta-Llama-3.2-1B-Instruct",
#     "Meta-Llama-3.2-3B-Instruct",
#     "gemini-1.5-flash",
#     "gemini-1.5-flash-exp-0827",
#     "gemini-1.5-flash-latest"
# ]

# divscore
models = ['dsr1', 'dsv3', 'gpt4o', 'gpto3mini']
datasets = ['LawStack', 'mimic_discharge', 'OALC', 'pubmedqa']

for label in [0, 1]:
    tag = 'human' if label == 0 else 'model'
    for model in models:
        for dataset in datasets:
            input_jsonl_path = f"./divscore/{model}_{dataset}.jsonl"
            output_txt_path = f"./FourierGPT/data/attack/{model}_{dataset}_{tag}.txt"
            os.makedirs("./baseline/FourierGPT/data/attack", exist_ok=True)

            count = 0

            with open(input_jsonl_path, 'r', encoding='utf-8') as f_in, \
                open(output_txt_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                        
                    json_data = json.loads(line)
                    text = json_data.get("text", "")
                    
                    if text and json_data['label'] == label:
                        text = text.replace('\n', '\\n').replace('\r', '')
                        
                        count += 1
                        f_out.write(text + "\n")

            print(f"✅ 处理完成！所有text已逐行写入 → {output_txt_path}")
            print(f"Total count for {model}: {count}")