# detect_cli.py
import argparse
import json

import pandas as pd
import os
from detectors.detector import get_detector
from raid.detect import run_detection_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the detector model you wish to run")
    parser.add_argument(
        "-d", "--data_path", type=str, default='data/raid/test.jsonl', help="Specify path to the data jsonl file to run detection on"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="predictions.jsonl", help="The file name to write the results to"
    )
    parser.add_argument(
        "-g", "--group", type=str, default='false', help='set true for infering DivScore dataset'
    )
    parser.add_argument(
        "-ckpt", "--ckpt_name", type=str, default=None, help='For Waveletcnn and QwenCls input the ckpt name'
    )
    args = parser.parse_args()

    if args.group == 'false':
        print(f"Loading dataset with name {args.data_path}...")
    
        df = pd.read_json(args.data_path, lines=True)

        print(f"Loading detector with name {args.model}...")
        detector = get_detector(args.model, ckpt_name=args.ckpt_name)

        print(f"Running detection...")
        scores = run_detection_jsonl(detector.inference, df)

        print(f"Done! Writing predictions to output path: {args.output_path}")

        with open(args.output_path, "w") as f:
            for obj, s in zip(df.to_dict(orient="records"), scores):
                obj["score"] = s
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    else:
        # for domain shift dataset
        folder_path = 'data/divscore'  # your divscore data path
        models = ['dsr1', 'dsv3', 'gpt4o', 'gpto3mini']
        datasets = ['LawStack', 'mimic_discharge', 'OALC', 'pubmedqa']
        print(f"Loading detector with name {args.model}...")
        detector = get_detector(args.model, ckpt_name=args.ckpt_name)
        detector_model = args.model
        if args.model == 'waveletcnn' or args.model == 'qwen2cls':
            detector_name = args.ckpt_name
        else:
            detector_name = args.model
        for model in models:
            for dataset in datasets:
                file_path = folder_path + f'/{model}_{dataset}.jsonl'
                output_path = f'/home/cxsj25f/output/divscore/{detector_name}/divscore_{detector_name}_{model}_{dataset}.jsonl'
                os.makedirs(f'/home/cxsj25f/output/divscore/{detector_name}/', exist_ok=True)
                print(f"Loading dataset with name {file_path}...")
                # 读取 JSONL 文件
                df = pd.read_json(file_path, lines=True)

                print(f"Running detection...")
                scores = run_detection_jsonl(detector.inference, df)

                print(f"Done! Writing predictions to output path: {output_path}")

                with open(output_path, "w") as f:
                    for obj, s in zip(df.to_dict(orient="records"), scores):
                        obj["score"] = s
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
