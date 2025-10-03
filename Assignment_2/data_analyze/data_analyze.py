# data_analyze/data_analyze.py
# Runs EDA and writes ALL outputs to: outputs/analyze/

from __future__ import annotations
from pathlib import Path
import os, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless image writing
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

# ----------------------------
# Robust project path handling
# ----------------------------
def _resolve_root() -> Path:
    """
    If run as a script: ROOT = parent of data_analyze/.
    If run from Jupyter: walk up from CWD until we find both 'outputs' and 'datasets'.
    """
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        cur = Path.cwd().resolve()
        for _ in range(6):
            if (cur / "datasets").exists() and (cur / "outputs").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        raise RuntimeError(
            "Couldn't locate project root. Open the notebook from 'Assignment_2' "
            "so that 'datasets/' and 'outputs/' are visible."
        )

ROOT = _resolve_root()
PROCESSED_DIR = ROOT / "outputs" / "processed"
ANALYZE_DIR   = ROOT / "outputs" / "analyze"    # <— all EDA outputs go here
ANALYZE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = PROCESSED_DIR / "emails_merged.processed.csv"

print("[INFO] ROOT        :", ROOT)
print("[INFO] INPUT CSV   :", DEFAULT_INPUT)
print("[INFO] ANALYZE DIR :", ANALYZE_DIR)

# ----------------------------
# Small helpers
# ----------------------------
def basic_length_stats(texts: pd.Series) -> pd.DataFrame:
    s = texts.fillna("").astype(str)
    num_chars = s.str.len()
    num_words = s.str.split().apply(len)
    avg_word_len = np.where(num_words > 0, num_chars / num_words, 0.0)
    return pd.DataFrame({
        "num_chars": num_chars,
        "num_words": num_words,
        "avg_word_len": avg_word_len,
    }, index=texts.index)

def _ensure_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df
    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(
            f"Expected processed file at {DEFAULT_INPUT}.\n"
            "Run the processing step first to create it."
        )
    return pd.read_csv(DEFAULT_INPUT)

# ----------------------------
# Main EDA routine
# ----------------------------
def analyze(df: pd.DataFrame | None = None, out_dir: Path | None = None) -> Path:
    """
    Perform EDA and write all artifacts to out_dir (defaults to outputs/analyze).
    If df is None, loads DEFAULT_INPUT.
    Returns the output directory path.
    """
    out = Path(out_dir) if out_dir else ANALYZE_DIR
    out.mkdir(parents=True, exist_ok=True)

    df = _ensure_df(df)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input DataFrame must have columns: 'text' and 'label'.")

    print("Running EDA on DataFrame with shape:", df.shape)

    # ------------------------------
    # 1) Top unigrams by class
    # ------------------------------
    vectorizer_uni = CountVectorizer(stop_words="english", min_df=5)
    X_all = vectorizer_uni.fit_transform(df["text"])
    vocab = np.array(vectorizer_uni.get_feature_names_out())

    X_ham = X_all[df["label"].values == 0]
    X_spam = X_all[df["label"].values == 1]

    ham_counts = np.asarray(X_ham.sum(axis=0)).ravel()
    spam_counts = np.asarray(X_spam.sum(axis=0)).ravel()

    top_ham = pd.DataFrame({"token": vocab, "count": ham_counts}).sort_values("count", ascending=False).head(25)
    top_spam = pd.DataFrame({"token": vocab, "count": spam_counts}).sort_values("count", ascending=False).head(25)

    plt.figure(figsize=(10,6))
    plt.bar(top_ham["token"], top_ham["count"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 25 Unigrams - Ham")
    plt.xlabel("token"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out / "top25_unigrams_ham.png", bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10,6))
    plt.bar(top_spam["token"], top_spam["count"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 25 Unigrams - Spam")
    plt.xlabel("token"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out / "top25_unigrams_spam.png", bbox_inches="tight"); plt.close()

    print("Saved unigram plots.")

    # ------------------------------
    # 2) Top bigrams by class
    # ------------------------------
    vectorizer_bi = CountVectorizer(stop_words="english", min_df=5, ngram_range=(2,2))
    X_all_bi = vectorizer_bi.fit_transform(df["text"])
    vocab_bi = np.array(vectorizer_bi.get_feature_names_out())

    X_ham_bi = X_all_bi[df["label"].values == 0]
    X_spam_bi = X_all_bi[df["label"].values == 1]

    ham_counts_bi = np.asarray(X_ham_bi.sum(axis=0)).ravel()
    spam_counts_bi = np.asarray(X_spam_bi.sum(axis=0)).ravel()

    top_ham_bi = pd.DataFrame({"bigram": vocab_bi, "count": ham_counts_bi}).sort_values("count", ascending=False).head(25)
    top_spam_bi = pd.DataFrame({"bigram": vocab_bi, "count": spam_counts_bi}).sort_values("count", ascending=False).head(25)

    plt.figure(figsize=(10,6))
    plt.bar(top_ham_bi["bigram"], top_ham_bi["count"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 25 Bigrams - Ham")
    plt.xlabel("bigram"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out / "top25_bigrams_ham.png", bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10,6))
    plt.bar(top_spam_bi["bigram"], top_spam_bi["count"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 25 Bigrams - Spam")
    plt.xlabel("bigram"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out / "top25_bigrams_spam.png", bbox_inches="tight"); plt.close()

    print("Saved bigram plots.")

    # ------------------------------
    # 3) TF-IDF + Chi-square discriminative terms
    # ------------------------------
    tfidf = TfidfVectorizer(stop_words="english", min_df=5)
    X_tfidf = tfidf.fit_transform(df["text"])
    y = df["label"].values
    feature_names = np.array(tfidf.get_feature_names_out())

    try:
        chi2_scores, pvals = chi2(X_tfidf, y)
        X_spam_tfidf = X_tfidf[y == 1]
        X_ham_tfidf  = X_tfidf[y == 0]
        mean_spam = np.asarray(X_spam_tfidf.mean(axis=0)).ravel()
        mean_ham  = np.asarray(X_ham_tfidf.mean(axis=0)).ravel()

        df_chi = pd.DataFrame({
            "feature": feature_names,
            "chi2": chi2_scores,
            "pval": pvals,
            "mean_spam": mean_spam,
            "mean_ham": mean_ham,
            "spam_minus_ham": mean_spam - mean_ham
        }).sort_values("chi2", ascending=False)

        df_chi.to_csv(out / "chi2_discriminative_terms.csv", index=False)
        print("Saved chi2 discriminative terms.")
    except Exception as e:
        print("Warning: chi2 step failed:", e)

    # ------------------------------
    # 4) Keyword presence rates
    # ------------------------------
    keywords = [
        "free", "win", "winner", "prize", "money", "cash", "click", "offer", "buy", "discount",
        "urgent", "account", "verify", "http", "www", "unsubscribe", "viagra", "loan", "credit", "bitcoin"
    ]

    def keyword_presence_rate(texts: pd.Series, kw: str) -> float:
        return texts.str.contains(rf"\b{re.escape(kw)}\b", regex=True, case=False).mean()

    rows = []
    for kw in keywords:
        rows.append({
            "keyword": kw,
            "rate_spam": keyword_presence_rate(df.loc[df["label"]==1, "text"], kw),
            "rate_ham":  keyword_presence_rate(df.loc[df["label"]==0, "text"], kw)
        })

    kw_df = pd.DataFrame(rows).sort_values("rate_spam", ascending=False)
    kw_df.to_csv(out / "keyword_presence_rates.csv", index=False)

    # plot (top 15 by spam rate)
    top15 = kw_df.head(15)
    x = np.arange(len(top15))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, top15["rate_spam"], width, label="spam")
    plt.bar(x + width/2, top15["rate_ham"], width, label="ham")
    plt.xticks(x, top15["keyword"], rotation=75, ha="right")
    plt.title("Keyword presence rates (top 15 by spam rate)")
    plt.xlabel("keyword"); plt.ylabel("fraction of messages")
    plt.legend()
    plt.tight_layout(); plt.savefig(out / "keyword_presence_rates_fixed.png", bbox_inches="tight"); plt.close()

    print("Saved keyword presence rates and plot.")

    # ------------------------------
    # 5) Token lift (log-ratio spam vs ham)
    # ------------------------------
    # reuse unigram totals
    ham_total = ham_counts.sum()
    spam_total = spam_counts.sum()
    ham_pct  = (ham_counts  + 1) / (ham_total  + len(ham_counts))
    spam_pct = (spam_counts + 1) / (spam_total + len(spam_counts))

    log_ratio = np.log(spam_pct / ham_pct)
    lift_df = pd.DataFrame({"token": vocab, "log_ratio": log_ratio, "spam_pct": spam_pct, "ham_pct": ham_pct})

    top_spam_skew = lift_df.sort_values("log_ratio", ascending=False).head(30)
    top_ham_skew  = lift_df.sort_values("log_ratio", ascending=True).head(30)

    (out / "token_lift_top_spam.csv").write_text("")  # ensure file presence on some FS
    top_spam_skew.to_csv(out / "token_lift_top_spam.csv", index=False)
    top_ham_skew.to_csv(out / "token_lift_top_ham.csv", index=False)

    print("Saved token lift CSVs.")

    # ------------------------------
    # 6) Quick text summary
    # ------------------------------
    n_total = len(df)
    n_spam  = int(df["label"].sum())
    n_ham   = n_total - n_spam

    len_means = (
        pd.concat([df[["label"]], basic_length_stats(df["text"])], axis=1)
          .groupby("label")[["num_chars","num_words","avg_word_len"]]
          .mean()
    )

    summary_lines = []
    summary_lines.append(f"Total messages: {n_total:,}")
    summary_lines.append(f"Ham (0): {n_ham:,}  |  Spam (1): {n_spam:,}  |  Spam rate: {n_spam/n_total:.2%}")
    summary_lines.append("")
    summary_lines.append("Average lengths by label:")
    for lbl, row in len_means.iterrows():
        summary_lines.append(f"  Label {lbl}: num_chars={row['num_chars']:.1f}, num_words={row['num_words']:.1f}, avg_word_len={row['avg_word_len']:.2f}")
    summary_lines.append("")
    summary_lines.append("Top spam-skewed tokens (log-ratio):")
    summary_lines.append(", ".join(top_spam_skew.head(15)["token"].tolist()))
    summary_lines.append("")
    summary_lines.append("Top ham-skewed tokens (log-ratio):")
    summary_lines.append(", ".join(top_ham_skew.head(15)["token"].tolist()))

    with open(out / "quick_insights.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("Saved quick summary.")

    # List produced files
    print("\nAll files written to", out.resolve())
    for p in sorted(out.iterdir()):
        print(" -", p.name)

    return out

# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    analyze()   # loads DEFAULT_INPUT and writes to ANALYZE_DIR

if __name__ == "__main__":
    main()
