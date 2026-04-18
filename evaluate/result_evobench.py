import json
import math
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def export_group_avg_auroc_stats(
    jsonl_path: str,
    output_excel: str,
    model_groups: dict,
):
    data = load_jsonl(jsonl_path)
    domains = sorted(set(item["domain"] for item in data))

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for group_name, models in model_groups.items():
            rows = []

            # model -> domain -> auroc
            model_domain_auroc = {}

            # ---------- step 1: each model × domain AUROC ----------
            for model_name in models:
                pos_model = model_name
                neg_model = f"{model_name}_human"

                domain2auroc = {}

                for domain in domains:
                    y_true, y_score = [], []

                    for item in data:
                        if item["domain"] != domain:
                            continue
                        if item["model"] not in (pos_model, neg_model):
                            continue

                        score = float(item["score"])
                        if not math.isfinite(score):
                            continue

                        y_true.append(1 if item["model"] == pos_model else 0)
                        y_score.append(score)

                    if len(set(y_true)) < 2:
                        domain2auroc[domain] = np.nan
                    else:
                        domain2auroc[domain] = roc_auc_score(y_true, y_score)

                model_domain_auroc[model_name] = domain2auroc

            # ---------- step 2: each model domain-avg ----------
            model_avg = {}
            for model_name, d2a in model_domain_auroc.items():
                vals = [v for v in d2a.values() if not np.isnan(v)]
                model_avg[model_name] = float(np.mean(vals)) if vals else np.nan

            # ---------- step 3: delta vs the earliest model ----------
            ref_model = models[0]
            ref_value = model_avg.get(ref_model, np.nan)

            # ---------- step 4: organize table ----------
            for model_name in models:
                row = {"model_name": model_name}

                # each domain AUROC
                for domain in domains:
                    v = model_domain_auroc[model_name][domain]
                    row[domain] = round(v, 4) if not np.isnan(v) else np.nan

                avg = model_avg.get(model_name, np.nan)
                delta = (
                    avg - ref_value
                    if not np.isnan(avg) and not np.isnan(ref_value)
                    else np.nan
                )

                row["domain_avg_auroc"] = round(avg, 4) if not np.isnan(avg) else np.nan
                row["delta_vs_first"] = round(delta, 4) if not np.isnan(delta) else np.nan

                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.set_index("model_name")

            # ---------- step 5: GROUP_STD ----------
            std_row = {}

            for col in df.columns:
                vals = df[col].dropna().values
                std_row[col] = round(float(np.std(vals)), 4) if len(vals) > 0 else np.nan

            df.loc["[GROUP_STD]"] = std_row

            df.to_excel(
                writer,
                sheet_name=group_name[:31],
            )

    print(f"Saved group-level AUROC statistics to: {output_excel}")


if __name__ == "__main__":
    baselines = [
        "binoculars",
        "chatgpt-roberta",
        "fastdetectgpt",
        "radar",
        "wavedetect-all",
        "wavedetect-basic"
    ]

    MODEL_GROUPS = {
        "gpt4o": [
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "chatgpt-4o-latest",
        ],
        "gpt4": [
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo-2024-04-09",
        ],
        "claude": [
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
        ],
        "gemini-1.5": [
            "gemini-1.5-flash",
            "gemini-1.5-flash-exp-0827",
            "gemini-1.5-flash-latest",
        ],
    }

    for baseline in baselines:
        jsonl_path = f"output/evobench/evobench_{baseline}.jsonl"
        os.makedirs('output/result/evobench/', exist_ok=True)
        output_excel = f"output/result/evobench/evobench_{baseline}.xlsx"

        export_group_avg_auroc_stats(
            jsonl_path,
            output_excel,
            MODEL_GROUPS,
        )
