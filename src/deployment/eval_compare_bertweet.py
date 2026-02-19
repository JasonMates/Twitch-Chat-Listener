import argparse
from pathlib import Path

from eval_compare_models import main as compare_main


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side eval for BoW, LR, and BERTweet",
        conflict_handler="resolve",
    )
    parser.add_argument("--data", default="Twitch_Sentiment_Labels.csv")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["validation", "prior_test", "balanced_test"], default="prior_test")
    parser.add_argument("--lr_model", default="data/lr_sentiment_model.joblib")
    parser.add_argument("--bertweet_model_dir", default="data/bertweet_sentiment_model")
    parser.add_argument("--emote_lexicon", default="twitch_emote_vader_lexicon.txt")
    parser.add_argument("--bert_max_length", type=int, default=128)
    parser.add_argument("--show_examples", type=int, default=0)
    parser.add_argument("--bert_min_confidence", type=float, default=0.0)
    parser.add_argument("--bert_min_margin", type=float, default=0.0)
    parser.add_argument("--bert_use_prior_correction", action="store_true")
    parser.add_argument("--bert_target_priors", default="")
    parser.add_argument("--bert_train_priors", default="")
    args = parser.parse_args()

    model_path = Path(args.bertweet_model_dir)
    print(f"Using BERTweet model dir: {model_path.resolve()}")

    argv = [
        "--data",
        args.data,
        "--test_size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--split",
        args.split,
        "--lr_model",
        args.lr_model,
        "--bert_model_dir",
        str(model_path),
        "--emote_lexicon",
        args.emote_lexicon,
        "--bert_max_length",
        str(args.bert_max_length),
        "--show_examples",
        str(args.show_examples),
        "--bert_min_confidence",
        str(args.bert_min_confidence),
        "--bert_min_margin",
        str(args.bert_min_margin),
    ]
    if args.bert_use_prior_correction:
        argv.append("--bert_use_prior_correction")
    if args.bert_target_priors:
        argv.extend(["--bert_target_priors", args.bert_target_priors])
    if args.bert_train_priors:
        argv.extend(["--bert_train_priors", args.bert_train_priors])

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        compare_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
