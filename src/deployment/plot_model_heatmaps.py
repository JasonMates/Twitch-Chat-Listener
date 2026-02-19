import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from bert_sentiment_model import build_splits, prepare_gold_message_df
from eval_compare_models import BertClassifier, LRClassifier
from realtime_analyzer import HybridClassifier

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {i: k for i, k in enumerate(LABELS)}


def label_ids(pred_labels: list[str]) -> np.ndarray:
    return np.asarray([LABEL_TO_ID.get(p, LABEL_TO_ID["Neutral"]) for p in pred_labels], dtype=np.int64)


def plot_confusion_heatmaps(y_true: np.ndarray, preds: dict, output_path: Path, normalize: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=140, constrained_layout=True)
    model_names = ["BoW", "LR", "Cardiff"]
    vmax = 1.0 if normalize else None

    for ax, model_name in zip(axes, model_names):
        y_pred = preds[model_name]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(np.float64)
        cm_show = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) if normalize else cm
        im = ax.imshow(cm_show, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_title(model_name)
        ax.set_xticks([0, 1, 2], LABELS, rotation=20, ha="right")
        ax.set_yticks([0, 1, 2], LABELS)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                val = cm_show[i, j]
                txt = f"{val:.2f}" if normalize else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.015)
    cbar.set_label("Row-normalized" if normalize else "Count")
    fig.suptitle("Model Confusion Heatmaps", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_datapoint_heatmap(y_true: np.ndarray, preds: dict, messages: list[str], output_path: Path, n_points: int) -> None:
    n = min(n_points, len(messages))
    cols = ["True", "BoW", "LR", "Cardiff"]
    arr = np.zeros((n, 4), dtype=np.int64)
    arr[:, 0] = y_true[:n]
    arr[:, 1] = preds["BoW"][:n]
    arr[:, 2] = preds["LR"][:n]
    arr[:, 3] = preds["Cardiff"][:n]

    fig_h = max(5, min(14, 0.24 * n + 1.6))
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=140, constrained_layout=True)
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(n), [str(i + 1) for i in range(n)])
    ax.set_xlabel("Source")
    ax.set_ylabel("Datapoint Index")
    ax.set_title(f"Per-datapoint Labeling (first {n} samples)")

    for i in range(n):
        for j in range(4):
            label = ID_TO_LABEL[int(arr[i, j])][0:3]
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(LABELS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Twitch_Sentiment_Labels.csv")
    ap.add_argument("--split", choices=["validation", "prior_test", "balanced_test"], default="prior_test")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_model", default="data/lr_sentiment_model.joblib")
    ap.add_argument("--cardiff_model_dir", default="data/cardiff_sentiment_model_v2")
    ap.add_argument("--emote_lexicon", default="twitch_emote_vader_lexicon.txt")
    ap.add_argument("--max_points", type=int, default=80, help="Rows in datapoint heatmap")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    df_msg = prepare_gold_message_df(args.data)
    splits = build_splits(df_msg, test_size=args.test_size, seed=args.seed)
    split_df = {
        "validation": splits.val_df,
        "prior_test": splits.prior_test_df,
        "balanced_test": splits.balanced_test_df,
    }[args.split].reset_index(drop=True)

    messages = split_df["message"].astype(str).tolist()
    y_true = split_df["sent_id"].to_numpy(dtype=np.int64)

    bow = HybridClassifier(emote_lexicon_path=args.emote_lexicon, use_vader=True)
    lr = LRClassifier(Path(args.lr_model))
    cardiff = BertClassifier(Path(args.cardiff_model_dir), emote_lexicon_path=args.emote_lexicon, max_length=128)

    bow_pred = label_ids([bow.predict(m)[0] for m in messages])
    lr_pred = label_ids([lr.predict_label(m) for m in messages])
    cardiff_pred = label_ids([cardiff.predict_label(m) for m in messages])

    preds = {"BoW": bow_pred, "LR": lr_pred, "Cardiff": cardiff_pred}
    out_dir = Path(args.out_dir)

    plot_confusion_heatmaps(
        y_true,
        preds,
        out_dir / f"heatmap_confusion_{args.split}.png",
        normalize=False,
    )
    plot_confusion_heatmaps(
        y_true,
        preds,
        out_dir / f"heatmap_confusion_{args.split}_normalized.png",
        normalize=True,
    )
    plot_datapoint_heatmap(
        y_true,
        preds,
        messages,
        out_dir / f"heatmap_datapoints_{args.split}.png",
        n_points=args.max_points,
    )

    print(f"Saved: {out_dir / f'heatmap_confusion_{args.split}.png'}")
    print(f"Saved: {out_dir / f'heatmap_confusion_{args.split}_normalized.png'}")
    print(f"Saved: {out_dir / f'heatmap_datapoints_{args.split}.png'}")


if __name__ == "__main__":
    main()
