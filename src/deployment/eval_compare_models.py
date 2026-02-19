import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert_sentiment_model import build_splits, prepare_gold_message_df
from realtime_analyzer import HybridClassifier

LABEL_ORDER = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}
_EPS = 1e-9

_SPECIAL_RE = re.compile(r"""[!@#$%^&*()_+\-=\[\]{};:'",.<>?/\\|`~]""")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_MENTION_RE = re.compile(r"@\w+")
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_AT_MENTION_RE = re.compile(r"@\w+")
_LEADING_SPEAKER_RE = re.compile(r"^\s*[A-Za-z0-9_]{2,25}:\s+")


def emote_candidates(token: str) -> list:
    base = token.strip()
    if not base:
        return []
    stripped = _EMOTE_EDGE_RE.sub("", base)
    raw = [base, base.lower(), stripped, stripped.lower()]
    out = []
    seen = set()
    for item in raw:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def normalize_twitter_text(text: str) -> str:
    msg = str(text or "")
    msg = _URL_RE.sub("http", msg)
    msg = _AT_MENTION_RE.sub("@user", msg)
    msg = _LEADING_SPEAKER_RE.sub("@user ", msg)
    return msg


def extract_emote_features(text: str, emote_sentiments: dict) -> dict:
    scores = []
    pos_count = 0
    neg_count = 0

    for token in str(text).split():
        matched = None
        for cand in emote_candidates(token):
            if cand in emote_sentiments:
                matched = emote_sentiments[cand]
                break
        if matched is None:
            continue
        scores.append(matched)
        if matched > 0:
            pos_count += 1
        elif matched < 0:
            neg_count += 1

    if scores:
        s = np.asarray(scores, dtype=np.float32)
        return {
            "emote_count": float(len(s)),
            "emote_avg_sentiment": float(np.mean(s)),
            "emote_max_sentiment": float(np.max(s)),
            "emote_min_sentiment": float(np.min(s)),
            "emote_sum_sentiment": float(np.sum(s)),
            "emote_positive_count": float(pos_count),
            "emote_negative_count": float(neg_count),
            "emote_abs_sum": float(np.sum(np.abs(s))),
            "has_positive_emote": float(pos_count > 0),
            "has_negative_emote": float(neg_count > 0),
            "emote_variance": float(np.var(s)),
        }

    return {
        "emote_count": 0.0,
        "emote_avg_sentiment": 0.0,
        "emote_max_sentiment": 0.0,
        "emote_min_sentiment": 0.0,
        "emote_sum_sentiment": 0.0,
        "emote_positive_count": 0.0,
        "emote_negative_count": 0.0,
        "emote_abs_sum": 0.0,
        "has_positive_emote": 0.0,
        "has_negative_emote": 0.0,
        "emote_variance": 0.0,
    }


def extract_text_features(text: str) -> dict:
    message = str(text)
    words = message.split()
    word_count = float(len(words))
    unique_ratio = (len(set(words)) / word_count) if word_count else 0.0

    return {
        "word_count": word_count,
        "char_count": float(len(message)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "has_caps": float(any(c.isupper() for c in message)),
        "all_caps": float(message.isupper() and len(message) > 2),
        "exclamation_count": float(message.count("!")),
        "question_count": float(message.count("?")),
        "repeated_chars": float(bool(_REPEAT_RE.search(message))),
        "repeated_chars_count": float(len(_REPEAT_RE.findall(message))),
        "mention_count": float(len(_MENTION_RE.findall(message))),
        "special_chars": float(len(_SPECIAL_RE.findall(message))),
        "digit_count": float(sum(c.isdigit() for c in message)),
        "unique_word_ratio": float(unique_ratio),
    }


class LRClassifier:
    def __init__(self, model_path: Path):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.tfidf = payload["tfidf"]
        self.char_tfidf = payload["char_tfidf"]
        self.scaler = payload["scaler"]
        self.num_feature_names = payload["num_feature_names"]
        self.emote_sentiments = payload.get("emote_sentiments", {})
        self.label_names = payload.get("label_names", {0: "Negative", 1: "Neutral", 2: "Positive"})

    def predict_label(self, text: str) -> str:
        msg = str(text or "")
        em = extract_emote_features(msg, self.emote_sentiments)
        tx = extract_text_features(msg)
        merged = {}
        merged.update(em)
        merged.update(tx)
        values = [float(merged.get(name, 0.0)) for name in self.num_feature_names]
        Xn = self.scaler.transform(np.asarray(values, dtype=np.float32).reshape(1, -1))

        Xw = self.tfidf.transform([msg])
        Xc = self.char_tfidf.transform([msg])
        X = sparse.hstack([Xw, Xc, sparse.csr_matrix(Xn)], format="csr")

        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        cls = int(self.model.classes_[idx])
        return self.label_names.get(cls, "Neutral")


class BertClassifier:
    def __init__(
            self,
            model_dir: Path,
            max_length: int = 128,
            emote_lexicon_path: str = "",
            min_confidence: float = 0.0,
            min_margin: float = 0.0,
            use_prior_correction: bool = False,
            target_priors_csv: str = "",
            train_priors_csv: str = "",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        id2label = getattr(self.model.config, "id2label", None) or ID_TO_LABEL
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.neutral_id = int(self.label2id.get("Neutral", 1))
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.use_prior_correction = bool(use_prior_correction)

        meta_path = model_dir / "model_meta.json"
        self.use_emote_tags = False
        self.normalize_twitter = False
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.use_emote_tags = bool(meta.get("use_emote_tags", False))
            self.normalize_twitter = bool(meta.get("normalize_twitter", False))
            if not emote_lexicon_path:
                emote_lexicon_path = str(meta.get("emote_lexicon_path", ""))

        self.emote_sentiments = self._load_emote_lexicon(emote_lexicon_path) if self.use_emote_tags else {}
        self.target_priors = self._parse_prior_csv(target_priors_csv)
        self.train_priors = self._parse_prior_csv(train_priors_csv)
        if not (self.target_priors and self.train_priors):
            self.use_prior_correction = False

    @staticmethod
    def _load_emote_lexicon(path: str) -> dict:
        lex = {}
        if not path or not os.path.exists(path):
            return lex
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                token = parts[0].strip()
                if not token:
                    continue
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                lex[token] = score
                low = token.lower()
                if low not in lex:
                    lex[low] = score
        return lex

    def _parse_prior_csv(self, csv_value: str):
        if not csv_value:
            return None
        parts = [p.strip() for p in csv_value.split(",")]
        if len(parts) != 3:
            return None
        try:
            vals = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        except ValueError:
            return None
        if np.any(vals <= 0):
            return None
        vals = vals / vals.sum()
        return {
            int(self.label2id.get("Negative", 0)): float(vals[0]),
            int(self.label2id.get("Neutral", 1)): float(vals[1]),
            int(self.label2id.get("Positive", 2)): float(vals[2]),
        }

    @staticmethod
    def _augment_with_emote_tags(text: str, lex: dict) -> str:
        msg = str(text or "")
        if not lex:
            return msg

        pos = 0
        neg = 0
        neu = 0
        for tok in msg.split():
            matched = None
            for cand in emote_candidates(tok):
                if cand in lex:
                    matched = float(lex[cand])
                    break
            if matched is None:
                continue
            if matched > 0.05:
                pos += 1
            elif matched < -0.05:
                neg += 1
            else:
                neu += 1

        tags = []
        if (pos + neg + neu) > 0:
            tags.append("has_emote")
        tags.extend(["emote_pos"] * min(pos, 3))
        tags.extend(["emote_neg"] * min(neg, 3))
        tags.extend(["emote_neu"] * min(neu, 3))
        if not tags:
            return msg
        return f"{msg} {' '.join(tags)}"

    @torch.no_grad()
    def predict_label(self, text: str) -> str:
        msg = str(text or "")
        if self.normalize_twitter:
            msg = normalize_twitter_text(msg)
        if self.use_emote_tags:
            msg = self._augment_with_emote_tags(msg, self.emote_sentiments)
        enc = self.tokenizer(
            msg,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float64)
        class_ids = [int(i) for i in range(len(probs))]

        if self.use_prior_correction and self.target_priors and self.train_priors:
            logp = np.log(np.clip(probs, _EPS, 1.0))
            adjusted = []
            for i, cls in enumerate(class_ids):
                target = max(self.target_priors.get(cls, _EPS), _EPS)
                train = max(self.train_priors.get(cls, _EPS), _EPS)
                adjusted.append(logp[i] + np.log(target) - np.log(train))
            adjusted = np.asarray(adjusted, dtype=np.float64)
            adjusted = adjusted - np.max(adjusted)
            expv = np.exp(adjusted)
            probs = expv / expv.sum()

        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_prob = float(probs[top_idx])
        margin = float(top_prob - float(probs[second_idx]))
        pred_id = top_idx
        if top_prob < self.min_confidence or margin < self.min_margin:
            pred_id = self.neutral_id
        return self.id2label.get(int(pred_id), "Neutral")


def metrics_for(y_true_ids: np.ndarray, y_pred_labels: list[str]) -> dict:
    y_pred_ids = np.asarray([LABEL_TO_ID.get(l, LABEL_TO_ID["Neutral"]) for l in y_pred_labels], dtype=np.int64)
    return {
        "macro_f1": f1_score(y_true_ids, y_pred_ids, average="macro"),
        "micro_f1": f1_score(y_true_ids, y_pred_ids, average="micro"),
        "acc": accuracy_score(y_true_ids, y_pred_ids),
        "pred_ids": y_pred_ids,
    }


def print_confusion(cm: np.ndarray) -> None:
    print(f"{'':10}Negative  Neutral  Positive")
    for i, lab in enumerate(LABEL_ORDER):
        print(f"{lab:10}{cm[i, 0]:8d}{cm[i, 1]:8d}{cm[i, 2]:8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Twitch_Sentiment_Labels.csv")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["validation", "prior_test", "balanced_test"], default="prior_test")
    ap.add_argument("--lr_model", default="data/lr_sentiment_model.joblib")
    ap.add_argument("--bert_model_dir", default="data/bert_sentiment_model")
    ap.add_argument("--emote_lexicon", default="twitch_emote_vader_lexicon.txt")
    ap.add_argument("--bert_max_length", type=int, default=128)
    ap.add_argument("--show_examples", type=int, default=0)
    ap.add_argument("--bert_min_confidence", type=float, default=0.0)
    ap.add_argument("--bert_min_margin", type=float, default=0.0)
    ap.add_argument("--bert_use_prior_correction", action="store_true")
    ap.add_argument("--bert_target_priors", default="")
    ap.add_argument("--bert_train_priors", default="")
    args = ap.parse_args()

    df_msg = prepare_gold_message_df(args.data)
    splits = build_splits(df_msg, test_size=args.test_size, seed=args.seed)

    split_df = {
        "validation": splits.val_df,
        "prior_test": splits.prior_test_df,
        "balanced_test": splits.balanced_test_df,
    }[args.split].reset_index(drop=True)

    messages = split_df["message"].astype(str).tolist()
    y_true = split_df["sent_id"].values.astype(np.int64)

    bow = HybridClassifier(emote_lexicon_path=args.emote_lexicon, use_vader=True)
    lr = LRClassifier(Path(args.lr_model))
    bert = BertClassifier(
        Path(args.bert_model_dir),
        max_length=args.bert_max_length,
        emote_lexicon_path=args.emote_lexicon,
        min_confidence=args.bert_min_confidence,
        min_margin=args.bert_min_margin,
        use_prior_correction=args.bert_use_prior_correction,
        target_priors_csv=args.bert_target_priors,
        train_priors_csv=args.bert_train_priors,
    )

    bow_pred = [bow.predict(m)[0] for m in messages]
    lr_pred = [lr.predict_label(m) for m in messages]
    bert_pred = [bert.predict_label(m) for m in messages]

    bow_m = metrics_for(y_true, bow_pred)
    lr_m = metrics_for(y_true, lr_pred)
    bert_m = metrics_for(y_true, bert_pred)

    print("=" * 72)
    print(f"SIDE-BY-SIDE EVAL ({args.split})")
    print("=" * 72)
    print(f"{'model':>10} {'macro_f1':>10} {'micro_f1':>10} {'acc':>10}")
    print(f"{'BoW':>10} {bow_m['macro_f1']:10.4f} {bow_m['micro_f1']:10.4f} {bow_m['acc']:10.4f}")
    print(f"{'LR':>10} {lr_m['macro_f1']:10.4f} {lr_m['micro_f1']:10.4f} {lr_m['acc']:10.4f}")
    print(f"{'BERT':>10} {bert_m['macro_f1']:10.4f} {bert_m['micro_f1']:10.4f} {bert_m['acc']:10.4f}")

    for name, pred_ids in [("BoW", bow_m["pred_ids"]), ("LR", lr_m["pred_ids"]), ("BERT", bert_m["pred_ids"])]:
        print("\n" + "-" * 72)
        print(f"{name} - CLASSIFICATION REPORT")
        print("-" * 72)
        print(classification_report(y_true, pred_ids, target_names=LABEL_ORDER))
        print(f"{name} - CONFUSION MATRIX")
        cm = confusion_matrix(y_true, pred_ids, labels=[0, 1, 2])
        print_confusion(cm)

    if args.show_examples > 0:
        print("\n" + "-" * 72)
        print(f"DISAGREEMENT EXAMPLES (up to {args.show_examples})")
        print("-" * 72)
        shown = 0
        for i, msg in enumerate(messages):
            trio = (bow_pred[i], lr_pred[i], bert_pred[i])
            if len(set(trio)) < 2:
                continue
            truth = ID_TO_LABEL[int(y_true[i])]
            print(f"[{shown + 1}] true={truth} | bow={trio[0]} | lr={trio[1]} | bert={trio[2]}")
            print(f"    {msg[:160]}")
            shown += 1
            if shown >= args.show_examples:
                break


if __name__ == "__main__":
    main()
