import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tpr_at_fpr(y_true, y_score, target_fpr=0.001):
    """
    Compute TPR at a given FPR using linear interpolation on the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # safety: roc_curve guarantees fpr is sorted ascending
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])

    return float(np.interp(target_fpr, fpr, tpr))

def evaluate_divscore_with_detectors(
    base_folder,
    detectors,
    models,
    datasets,
    target_fpr=0.001,
):
    results = []

    for detector in detectors:
        detector_folder = os.path.join(base_folder, detector)

        for model in models:
            for dataset in datasets:
                file_path = os.path.join(
                    detector_folder,
                    f"divscore_{detector}_{model}_{dataset}.jsonl"
                )

                if not os.path.exists(file_path):
                    print(f"[Skip] {file_path}")
                    continue

                data = load_jsonl(file_path)

                y_true = np.array([x["label"] for x in data])
                y_score = np.array([x["score"] for x in data])

                auroc = roc_auc_score(y_true, y_score)
                tpr_001 = tpr_at_fpr(y_true, y_score, target_fpr)

                results.append({
                    "detector": detector,
                    "model": model,
                    "dataset": dataset,
                    "AUROC": auroc,
                    "TPR@0.1%FPR": tpr_001,
                    "num_pos": int((y_true == 1).sum()),
                    "num_neg": int((y_true == 0).sum()),
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    base_folder = "output/divscore"
    output_excel = "output/result/divscore_baselines.xlsx"

    detectors = [
        "binoculars",
        "chatgpt-roberta",
        "fastdetectgpt",
        "radar",
        "wavedetect-basic",
        'wavedetect-all',
    ]

    models = ['dsr1', 'dsv3', 'gpt4o', 'gpto3mini']
    datasets = ['LawStack', 'mimic_discharge', 'OALC', 'pubmedqa']

    df = evaluate_divscore_with_detectors(
        base_folder=base_folder,
        detectors=detectors,
        models=models,
        datasets=datasets,
        target_fpr=0.001
    )

    df = df.sort_values(
        by=["detector", "model", "dataset"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="divscore", index=False)

    print(f"Saved results to {output_excel}")
