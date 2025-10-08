# save as: /Users/datschef/Documents/GitHub/cos30049_spam_detection/Assignment_2/data_processing/analyze_emails.py
# Runs in Jupyter or as a standalone script.
# Produces 4 detailed visualizations saved under: Assignment_2/outputs/analyzed/

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend if not in Jupyter
if not any(k.startswith("JPY_PARENT_PID") for k in os.environ):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Paths --------------------
try:
    here = Path(__file__).resolve().parent   # script mode
except NameError:
    here = Path.cwd().resolve()              # Jupyter mode

# Try a few likely bases (this dir, parents, and with/without 'Assignment_2')
candidates = [here, here.parent, here.parent.parent]
candidates += [b / "Assignment_2" for b in list(candidates)]

DATA = None
for base in candidates:
    probe = base / "outputs" / "processed" / "emails_merged.processed.csv"
    if probe.exists():
        ROOT = base
        DATA = probe
        break

if DATA is None:
    # Fallback to your explicit project path
    ROOT = Path("/Users/datschef/Documents/GitHub/cos30049_spam_detection/Assignment_2")
    DATA = ROOT / "outputs" / "processed" / "emails_merged.processed.csv"

OUTDIR = ROOT / "outputs" / "analyzed"
OUTDIR.mkdir(parents=True, exist_ok=True)
print(f"Reading: {DATA}")

# -------------------- Load ---------------------
df = pd.read_csv(DATA)
assert {"text", "label"}.issubset(df.columns), "emails_merged.processed.csv must have columns: text,label"

# -------------------- Feature engineering -------------
df["text"] = df["text"].astype(str)
df["char_len"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["has_digit"] = df["text"].str.contains(r"\d")
df["digit_count"] = df["text"].str.count(r"\d")
df["digit_density"] = df["digit_count"] / df["char_len"].replace(0, np.nan)

# =============================================================================
# ==================== Insight 1: Spam keyword categories (PIE) ====================
# Categorize spam messages by simple keyword rules
categories = {
    "Financial":   [r"\b(pay|cash|credit|loan|account|bank|invoice|usd|dollar|price|cost)\b"],
    "Promotional": [r"\b(free|win|offer|promo|deal|discount|sale|subscribe|prize|bonus)\b"],
    "Action":      [r"\b(click|confirm|verify|download|open|reply|call|visit)\b"],
    "Urgency":     [r"\b(urgent|now|immediately|act|limited|expires|last chance)\b"],
    "Technical":   [r"\b(http|www|link|email|password|unsubscribe)\b"],
}
spam_df = df[df["label"] == 1].copy()

def assign_category(text: str) -> str:
    for cat, patterns in categories.items():
        for pat in patterns:
            if re.search(pat, text):
                return cat
    return "Other"

spam_df["category"] = spam_df["text"].str.lower().map(assign_category)
cat_counts = spam_df["category"].value_counts().reindex(
    ["Financial","Promotional","Action","Urgency","Technical","Other"]
).fillna(0)

fig1 = plt.figure(figsize=(8,8))
plt.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Spam Keyword Categories Distribution")
plt.tight_layout()
fig1_path = OUTDIR / "keyword_categories_pie.png"
plt.savefig(fig1_path); plt.close(fig1)

print("[Insight 1] Spam category shares:\n", (cat_counts / max(cat_counts.sum(),1)).round(3))

# =============================================================================
# ========== Insight 2: Keyword presence rates (top 15 by spam rate) (BAR, optimized) ==========
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"\b[a-z0-9]{3,}\b",  # ignore very-short tokens
    max_features=2000,                  # speed guard
)
X = vectorizer.fit_transform(df["text"])
vocab = np.array(vectorizer.get_feature_names_out())

# Use integer indices (not boolean Series) for sparse row selection
spam_idx = np.flatnonzero(df["label"].to_numpy() == 1)
ham_idx  = np.flatnonzero(df["label"].to_numpy() == 0)

# Presence (True if token appears at least once in a message)
spam_means = (X[spam_idx] > 0).mean(axis=0).A1
ham_means  = (X[ham_idx]  > 0).mean(axis=0).A1

# Top 15 by spam presence rate
idx = np.argsort(spam_means)[::-1][:15]
top_words = vocab[idx]
s_vals = spam_means[idx]
h_vals = ham_means[idx]

# Plot
x = np.arange(len(top_words))
width = 0.42
fig2 = plt.figure(figsize=(12,7))
plt.bar(x - width/2, s_vals, width, label="Spam", color="#ff7f7f")
plt.bar(x + width/2, h_vals, width, label="Ham",  color="#b59b3d")
plt.xticks(x, top_words, rotation=45, ha="right")
plt.ylabel("Fraction of messages containing word")
plt.xlabel("Keyword")
plt.title("Keyword Presence Rates (Top 15 by Spam Rate)")
plt.legend()
plt.tight_layout()
fig2_path = OUTDIR / "keyword_presence_rates.png"
plt.savefig(fig2_path); plt.close(fig2)

# Textual insight
top5 = [f"{w} (spam {s:.2%} vs ham {h:.2%})" for w, s, h in zip(top_words[:5], s_vals[:5], h_vals[:5])]
print("[Insight 2] Top spam-related words:\n  " + "\n  ".join(top5))


# =============================================================================
# ===== Insight 3: Top-15 tokens by class (dual horizontal BARs, side-by-side) =====
tokens = texts.str.findall(token_re)
spam_tok = tokens[df["label"] == 1].explode().value_counts().head(15)
ham_tok  = tokens[df["label"] == 0].explode().value_counts().head(15)

fig3, ax = plt.subplots(1, 2, figsize=(16,6), sharey=False)
ax[0].barh(ham_tok.index[::-1], ham_tok.values[::-1], color="#86c5da")
ax[0].set_title("Top 15 Tokens in Ham Emails"); ax[0].set_xlabel("Frequency")

ax[1].barh(spam_tok.index[::-1], spam_tok.values[::-1], color="#e78b8b")
ax[1].set_title("Top 15 Tokens in Spam Emails"); ax[1].set_xlabel("Frequency")

plt.tight_layout()
fig3_path = OUTDIR / "token_frequency_comparison.png"
plt.savefig(fig3_path); plt.close(fig3)

print("[Insight 3] Overlap between top-15 ham & spam tokens:",
      len(set(ham_tok.index).intersection(set(spam_tok.index))))


# =============================================================================
# Insight 4 (SCATTER): Word count vs digit density — with trend lines & correlations
# =============================================================================
mask0 = (df["label"] == 0)
mask1 = (df["label"] == 1)

x0, y0 = df.loc[mask0, "word_count"].values, df.loc[mask0, "digit_density"].values
x1, y1 = df.loc[mask1, "word_count"].values, df.loc[mask1, "digit_density"].values

fig4 = plt.figure()
plt.scatter(x0, y0, alpha=0.3, label="harmless (0)", s=15)
plt.scatter(x1, y1, alpha=0.3, label="spam (1)", s=15, marker="^")

# add simple linear fits (no extra deps)
if np.isfinite(x0).all() and np.isfinite(y0).all() and len(x0) > 1:
    m0, b0 = np.polyfit(x0[np.isfinite(x0) & np.isfinite(y0)],
                        y0[np.isfinite(x0) & np.isfinite(y0)], 1)
    xgrid0 = np.linspace(np.nanmin(x0), np.nanmax(x0), 100)
    plt.plot(xgrid0, m0*xgrid0 + b0, linestyle="--", label="trend (0)")

if np.isfinite(x1).all() and np.isfinite(y1).all() and len(x1) > 1:
    m1, b1 = np.polyfit(x1[np.isfinite(x1) & np.isfinite(y1)],
                        y1[np.isfinite(x1) & np.isfinite(y1)], 1)
    xgrid1 = np.linspace(np.nanmin(x1), np.nanmax(x1), 100)
    plt.plot(xgrid1, m1*xgrid1 + b1, linestyle=":", label="trend (1)")

plt.title("Word Count vs Digit Density (with linear trends)")
plt.xlabel("Word Count")
plt.ylabel("Digit Density (digits / characters)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
fig4_path = OUTDIR / "scatter_wordcount_vs_digitdensity.png"
plt.savefig(fig4_path)
plt.close(fig4)

# correlations
def safe_corr(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    b = b.replace([np.inf, -np.inf], np.nan).dropna()
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    return np.corrcoef(a[:n], b[:n])[0,1]

r0 = safe_corr(x0, y0)
r1 = safe_corr(x1, y1)
print(f"[Insight 4] Correlation(word_count, digit_density) — harmless={r0:.3f}, spam={r1:.3f}.")

# -------------------- Save a small README -----------------
with open(OUTDIR / "README.txt", "w", encoding="utf-8") as f:
    f.write(
        "Figures generated by analyze_emails.py\n"
        f"- {fig1_path.name}: Label distribution with counts & % (bar)\n"
        f"- {fig2_path.name}: Top-10 words — class-wise proportions, size~freq (scatter)\n"
        f"- {fig3_path.name}: CDF of char length (overall & by class) + quantiles (line)\n"
        f"- {fig4_path.name}: Word count vs digit density with trend lines (scatter)\n"
    )

print(f"Saved all visualizations to: {OUTDIR}")
