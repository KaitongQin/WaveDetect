import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def load_jsonl(path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tpr_at_fpr(y_true, y_score, target_fpr=0.001):
    """
    Compute TPR at a given FPR using linear interpolation on the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])

    return float(np.interp(target_fpr, fpr, tpr))

def evaluate_models_vs_human(data, human_name="human", target_fpr=0.001):
    """
    Evaluate how well the detector separates each specific generator model from human text.
    
    Args:
        data: list of dict, each dict contains at least 'model' and 'score'.
        human_name: the string identifier for human-written text.
        target_fpr: false positive rate threshold.
        
    Returns:
        pd.DataFrame sorted by AUROC containing metrics for each model vs human.
    """
    model2samples = defaultdict(list)
    for x in data:
        model2samples[x["model"]].append(x)

    results = []

    for model_name in model2samples:
        if model_name == human_name:
            continue

        # Combine human samples and the specific model's samples
        samples = model2samples[human_name] + model2samples[model_name]

        y_true = []
        y_score = []

        for x in samples:
            # Positive class: Machine-generated (current model)
            # Negative class: Human-written
            y_true.append(1 if x["model"] == model_name else 0)
            y_score.append(x["score"])

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Skip if we don't have both classes (e.g., missing human or missing model data)
        if len(np.unique(y_true)) < 2:
            continue

        # AUROC
        auroc = roc_auc_score(y_true, y_score)

        # TPR@target_fpr
        tpr_001 = tpr_at_fpr(y_true, y_score, target_fpr)

        results.append({
            "model": model_name,
            "AUROC": auroc,
            "TPR@0.1%FPR": tpr_001,
            "num_pos": int((y_true == 1).sum()),
            "num_neg": int((y_true == 0).sum()),
        })

    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results).sort_values("AUROC", ascending=False)

def evaluate_by_attack_label(data, target_fpr=0.001):
    """
    For each attack type:
    - label=1 as positive class (Machine)
    - label=0 as negative class (Human)
    - Compute AUROC and TPR@target_fpr directly
    """
    attacks = sorted(set(
        x.get("attack", "none") for x in data
    ))

    attack_results = {}

    for attack in attacks:
        attack_data = [
            x for x in data
            if x.get("attack", "none") == attack
        ]

        if len(attack_data) == 0:
            continue

        y_true = np.array([x["label"] for x in attack_data])
        y_score = np.array([x["score"] for x in attack_data])

        # Must have both positive and negative samples
        if len(np.unique(y_true)) < 2:
            attack_results[attack] = {
                "AUROC": np.nan,
                "TPR@0.1%FPR": np.nan,
                "num_pos": int((y_true == 1).sum()),
                "num_neg": int((y_true == 0).sum()),
            }
            continue

        auroc = roc_auc_score(y_true, y_score)
        tpr_001 = tpr_at_fpr(y_true, y_score, target_fpr)

        attack_results[attack] = {
            "AUROC": float(auroc),
            "TPR@0.1%FPR": float(tpr_001),
            "num_pos": int((y_true == 1).sum()),
            "num_neg": int((y_true == 0).sum()),
        }

    return attack_results

if __name__ == "__main__":
    baselines = [
        "binoculars",
        "chatgpt-roberta",
        "fastdetectgpt",
        "radar",
        "wavedetect-basic",
        "wavedetect-all",
    ]

    output_path = "output/result/raid-all/raid-all-baseline.xlsx"
    os.makedirs("output/result/raid-all", exist_ok=True)

    # Tables for Attack Evaluation
    auroc_table_attack = {}
    tpr_table_attack = {}
    all_attacks = set()

    # Tables for Generator Model vs Human Evaluation
    auroc_table_model = {}
    tpr_table_model = {}
    all_models = set()

    for baseline in baselines:
        print(f"Processing baseline: {baseline}")

        data = load_jsonl(f"output/raid/raid_{baseline}.jsonl")

        # ---------------------------------------------------------
        # 1. Evaluate by Attack Type
        # ---------------------------------------------------------
        attack_results = evaluate_by_attack_label(data, target_fpr=0.001)

        auroc_table_attack[baseline] = {}
        tpr_table_attack[baseline] = {}

        for attack, metrics in attack_results.items():
            auroc_table_attack[baseline][attack] = metrics["AUROC"]
            tpr_table_attack[baseline][attack] = metrics["TPR@0.1%FPR"]
            all_attacks.add(attack)

        # ---------------------------------------------------------
        # 2. Evaluate Generator Models vs Human
        # ---------------------------------------------------------
        # This evaluates how well this baseline separates e.g., "gpt4" from "human"
        model_results_df = evaluate_models_vs_human(data, human_name="human", target_fpr=0.001)
        
        auroc_table_model[baseline] = {}
        tpr_table_model[baseline] = {}

        if not model_results_df.empty:
            for _, row in model_results_df.iterrows():
                gen_model = row["model"]
                auroc_table_model[baseline][gen_model] = row["AUROC"]
                tpr_table_model[baseline][gen_model] = row["TPR@0.1%FPR"]
                all_models.add(gen_model)

    # Convert gathered data into DataFrames
    attacks = sorted(all_attacks)
    gen_models = sorted(all_models)

    auroc_df_attack = pd.DataFrame.from_dict(auroc_table_attack, orient="index")[attacks]
    tpr_df_attack = pd.DataFrame.from_dict(tpr_table_attack, orient="index")[attacks]
    
    auroc_df_model = pd.DataFrame.from_dict(auroc_table_model, orient="index")[gen_models]
    tpr_df_model = pd.DataFrame.from_dict(tpr_table_model, orient="index")[gen_models]

    # Save everything to the Excel file in distinct sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        auroc_df_attack.to_excel(writer, sheet_name="Attack_AUROC")
        tpr_df_attack.to_excel(writer, sheet_name="Attack_TPR@0.1%FPR")
        
        auroc_df_model.to_excel(writer, sheet_name="Model_vs_Human_AUROC")
        tpr_df_model.to_excel(writer, sheet_name="Model_vs_Human_TPR@0.1%FPR")

    print(f"\nSaved Attack-wise and Model-wise results to {output_path}")