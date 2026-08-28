import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def load_results(directory, model_name, dataset_name):
    """
    Load evaluation results for each algorithm variant (baseline/edit methods)
    """
    path = os.path.join(directory, model_name)
    results = {}
    for filename in os.listdir(path):
        if filename.startswith(dataset_name):
            alg_name = filename.replace(dataset_name + "_", "").replace(".jsonl", "")
            with open(os.path.join(path, filename), "r") as f:
                data = [json.loads(line) for line in f]
                results[alg_name] = data
    return results

def compute_avg_metrics(results):
    """
    Compute average efficacy/specificity scores and magnitudes for each algorithm and format.
    Returns a DataFrame with: algorithm, format, metric_type, avg_score, avg_magnitude
    """
    alg_name_map = {
        "baseline": "Baseline",
        "ROME_completion": "ROME$_{completion}$",
        "MEMIT_completion": "MEMIT$_{completion}$",
        "AlphaEdit_completion": "AlphaEdit$_{completion}$",
        "MEMIT_triplet": "MEMIT$_{triplet}$",
        "MEMIT_ODQA": "MEMIT$_{ODQA}$",
        "MEMIT_MC": "MEMIT$_{MC}$",
        "MEMIT_TF": "MEMIT$_{TF}$",
        "MEMIT_YN": "MEMIT$_{YN}$",
        "ICE_demonstration": "IKE$_{completion}$",
        "MEMIT_merge_TF_completion": "MEMIT$_{MF}$ (K=2)",
        "MEMIT_merge_TF_completion_triplet": "MEMIT$_{MF}$ (K=3)",
        "MEMIT_merge_MC_TF_completion_triplet": "MEMIT$_{MF}$ (K=4)",
        "MEMIT_merge_MC_ODQA_TF_YN_completion_triplet": "MEMIT$_{MF}$ (all)",
    }
    summary = []
    # cross-format (base: completion)
    for alg_name in ["baseline", "ROME_completion", "MEMIT_completion", "AlphaEdit_completion", "ICE_demonstration"]:

    # # cross-format (extended)
    # for alg_name in ["baseline", "MEMIT_completion", "MEMIT_triplet", "MEMIT_ODQA", "MEMIT_MC", "MEMIT_TF", "MEMIT_YN"]:

    # # multi-format editing (six human-curated formats)
    # for alg_name in ["MEMIT_completion", "MEMIT_merge_TF_completion", "MEMIT_merge_TF_completion_triplet", "MEMIT_merge_MC_TF_completion_triplet", "MEMIT_merge_MC_ODQA_TF_YN_completion_triplet"]:
        entries = results.get(alg_name, [])
        for fmt in ["", "triplet_", "ODQA_", "MC_", "TF_", "YN_"]:  # Task format prefixes
            fmt_name = fmt.replace("_", "") or "Completion"

            for metric_type, score_key, mag_key in [
                ("Efficacy", f"{fmt}efficacy_score", f"{fmt}efficacy_magnitude"),
                ("Specificity", f"{fmt}specificity_score", f"{fmt}specificity_magnitude")
            ]:
                scores = [ex.get(score_key) for ex in entries if score_key in ex]
                mags = [ex.get(mag_key) for ex in entries if mag_key in ex]
                if scores:
                    summary.append({
                        "algorithm": alg_name_map.get(alg_name, alg_name),
                        "format": fmt_name.replace("Completion", "completion"),
                        "metric_type": metric_type,
                        "avg_score": sum(scores) / len(scores),
                        "avg_magnitude": sum(mags) / len(mags),
                    })
    return pd.DataFrame(summary)

def plot_heatmaps(df, eval_dir, model_name, dataset_name):
    """
    Generate heatmaps for Efficacy and Specificity (Score only).
    Saves to: evaluation/{model_name}/summary/{dataset_name}_{metric_type}_heatmap.png
    """
    import numpy as np

    sns.set_theme(style="white")

    save_dir = os.path.join(eval_dir, model_name, "summary")
    os.makedirs(save_dir, exist_ok=True)

    metric_types = ["Efficacy", "Specificity"]
    formats = ["completion", "triplet", "ODQA", "MC", "TF", "YN"]
    formats_specificity = ["completion", "ODQA", "MC", "TF", "YN"]

    methods = df["algorithm"].unique()
    figsize = (1.5 * len(formats) + 3, 0.6 * len(methods) + 3)

    for metric_type in metric_types:
        metric_df = df[(df["metric_type"] == metric_type)]

        pivot_df = metric_df.pivot(
            index="algorithm",
            columns="format",
            values="avg_score"
        ).reindex(index=methods, columns=formats if metric_type != "Specificity" else formats_specificity)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "Average Score"},
            annot_kws={"size": 20}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.yaxis.label.set_size(28)
        # ax.set_title(f"{metric_type} Score Heatmap", fontsize=20)
        ax.set_xlabel("Format", fontsize=28)
        ax.set_ylabel("Method", fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=28)

        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{dataset_name}_{metric_type.lower()}_score_heatmap.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


        pivot_df = metric_df.pivot(
            index="algorithm",
            columns="format",
            values="avg_magnitude"
        ).reindex(index=methods, columns=formats if metric_type != "Specificity" else formats_specificity)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "Average Magnitude"},
            annot_kws={"size": 20}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.yaxis.label.set_size(28)
        # ax.set_title(f"{metric_type} Magnitude Heatmap", fontsize=20)
        ax.set_xlabel("Format", fontsize=28)
        ax.set_ylabel("Method", fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=28)

        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{dataset_name}_{metric_type.lower()}_magnitude_heatmap.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama_Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="multiformat_counterfact_1000")
    parser.add_argument("--eval_dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    results = load_results(args.eval_dir, args.model_name, args.dataset_name)
    df = compute_avg_metrics(results)
    # plot_metrics(df, args.eval_dir, args.model_name, args.dataset_name)
    plot_heatmaps(df, args.eval_dir, args.model_name, args.dataset_name)

    print("Done. Plots saved.")


if __name__ == "__main__":
    main()
