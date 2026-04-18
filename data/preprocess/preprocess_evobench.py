import os
import re
import json
import argparse
import random

def find_evobench_files(root_dir):
    """
    Recursively find all *.raw_data.json files in the specified directory.
    Naming convention: {domain}_{model_name}.raw_data.json
    Returns a list of dictionaries: [{"domain":..., "model_name":..., "path":...}]
    """
    results = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".raw_data.json"):
                full_path = os.path.join(root, file)
                m = re.match(r"([a-zA-Z0-9]+)_(.+)\.raw_data\.json", file)
                if not m:
                    print(f"⚠️ Skipping unparseable filename: {file}")
                    continue
                domain, model_name = m.groups()
                results.append({
                    "domain": domain,
                    "model_name": model_name,
                    "path": full_path
                })
    print(f"\n✅ Found {len(results)} EvoBench files in total")
    return results


def convert_file_to_jsonl(src_path, save_path):
    """
    Convert EvoBench's {original, sampled} JSON format to JSONL format.
    label=0 -> human (original)
    label=1 -> LLM (sampled)
    """
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for text in data.get("original", []):
        if text.strip():
            lines.append({"text": text.strip(), "label": 0})
            
    for text in data.get("sampled", []):
        if text.strip():
            lines.append({"text": text.strip(), "label": 1})
            
    random.shuffle(lines)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ {os.path.basename(src_path)} → {save_path} ({len(lines)} samples)")


def convert_all_evobench(root_dir, output_dir):
    """
    Main function: Recursively scan and batch convert files into a flat structure.
    Files will be saved as: output_dir/{domain}_{model_name}.jsonl
    """
    files = find_evobench_files(root_dir)
    os.makedirs(output_dir, exist_ok=True)

    for item in files:
        domain = item["domain"]
        model_name = item["model_name"]
        src_path = item["path"]

        save_path = os.path.join(output_dir, f"{domain}_{model_name}.jsonl")
        convert_file_to_jsonl(src_path, save_path)

    print(f"\n🎯 All files have been converted successfully. Saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EvoBench .raw_data.json to flat JSONL format")
    parser.add_argument("--root_dir", type=str, default="./EvoBench", help="Path to EvoBench root directory")
    parser.add_argument("--output_dir", type=str, default="../evobench", help="Output directory for .jsonl files")
    args = parser.parse_args()

    convert_all_evobench(args.root_dir, args.output_dir)