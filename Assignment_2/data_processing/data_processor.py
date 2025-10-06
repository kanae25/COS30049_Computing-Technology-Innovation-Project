# data_processor.py
# Outputs are ALL: columns -> text,label  (0 = harmless, 1 = spam)

from __future__ import annotations
import os, re, unicodedata, string
import pandas as pd

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(SCRIPT_DIR)              # ASSIGNMENT_2
DATA_IN  = os.path.join(ROOT, "datasets")
DATA_OUT = os.path.join(ROOT, "outputs", "processed")
os.makedirs(DATA_OUT, exist_ok=True)

FILES = {
    "email_spam": os.path.join(DATA_IN, "email_spam.csv"),
    "emails":     os.path.join(DATA_IN, "emails.csv"),
    "text_spam":  os.path.join(DATA_IN, "text_spam.csv"),
}

# ---------- Cleaning helpers ----------
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_RE  = re.compile(r"<[^>]+>")
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "“”‘’—–…•·”’“"})

def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\u00A0", " ")
    t = HTML_RE.sub(" ", t)                    # drop HTML tags
    t = URL_RE.sub(" ", t)                     # drop full URLs
    t = EMAIL_RE.sub(" ", t)                   # drop email addresses
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower().translate(PUNCT_TABLE)       # remove punctuation, keep digits
    # NOTE: no re.sub(r"\d+", " ", t) here — we keep numbers
    return _ws(t)

def drop_empty_dupes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Ensure it's a string Series (guard against duplicate-named columns)
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]  # take the first 'text' if duplicates exist
    s = col_data.astype(str)

    df = df.assign(**{col: s})
    df = df[df[col].str.strip().astype(bool)]
    return df.drop_duplicates(subset=[col])

def to_2col(df: pd.DataFrame) -> pd.DataFrame:
    # Build the 2-column frame explicitly to avoid name collisions
    text_series = df["clean_text"] if "clean_text" in df.columns else df["text"]
    out = pd.DataFrame({
        "text": text_series.astype(str),
        "label": df["label"].astype(int).clip(0, 1)
    })
    out = drop_empty_dupes(out, "text").reset_index(drop=True)
    return out

def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    # map any common strings -> 0/1  (0 = harmless/ham, 1 = spam)
    map_str = {
        "spam": 1, "ham": 0, "not spam": 0, "non-spam": 0, "non spam": 0, "legit": 0,
        "harmless": 0
    }
    if df[label_col].dtype.kind in "OUSU":
        df["label"] = df[label_col].astype(str).str.lower().map(map_str)
    else:
        df["label"] = df[label_col].astype(int).clip(0, 1)
    # any unknown strings -> NaN; drop them
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

# CSV reader with UTF-8 handling
def safe_read_csv(path: str) -> pd.DataFrame:
    """
    First read CSV using UTF-8, then common fallbacks to handle mixed encodings
    without adding third-party dependencies.
    """
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return pd.read_csv(path, encoding="cp1252")
    except UnicodeDecodeError:
        pass
    return pd.read_csv(path, encoding="latin1")

# ---------- Dataset processors ----------
def process_email_spam(path: str) -> pd.DataFrame:
    df = safe_read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    title = cols.get("title")
    text  = cols.get("text")
    lab   = cols.get("type") or cols.get("label") or cols.get("spam")
    if not text:
        raise ValueError("email_spam.csv needs a 'text' column")
    if not lab:
        raise ValueError("email_spam.csv needs a 'type'/'label'/'spam' column")

    # combine title + text if title exists
    if title:
        df["text"] = (df[title].fillna("").astype(str).str.strip() + " " +
                      df[text].fillna("").astype(str).str.strip()).str.strip()
    else:
        df["text"] = df[text].fillna("").astype(str)

    df = standardize_labels(df, lab)
    df = drop_empty_dupes(df, "text")
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return to_2col(df)

def process_emails(path: str) -> pd.DataFrame:
    df = safe_read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text = cols.get("text")
    lab  = cols.get("spam") or cols.get("label") or cols.get("type")
    if not text or not lab:
        raise ValueError("emails.csv needs 'text' and 'spam/label/type' columns")

    df["text"] = df[text].fillna("").astype(str)
    df["text"] = df["text"].str.replace(r"^\s*subject\s*:\s*", "", regex=True)
    df = standardize_labels(df, lab)
    df = drop_empty_dupes(df, "text")
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return to_2col(df)

def process_text_spam(path: str) -> pd.DataFrame:
    df = safe_read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Accept capitalized headers from this file
    text = cols.get("text") or cols.get("message") or cols.get("content")
    lab  = (cols.get("label") or cols.get("spam") or cols.get("type") or
            cols.get("target") or cols.get("category"))  # <-- added 'category'

    if not text or not lab:
        raise ValueError("text_spam.csv needs 'text/message/content' AND 'label/spam/type/target/category'")

    df["text"] = df[text].fillna("").astype(str)
    df = standardize_labels(df, lab)   # maps 'ham'->0, 'spam'->1 already
    df = drop_empty_dupes(df, "text")
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return to_2col(df)

# ---------- Main ----------
def main():
    print("== Preprocessing to 2-column schema: text,label (0 harmless, 1 spam) ==")
    a = process_email_spam(FILES["email_spam"])
    b = process_emails(FILES["emails"])
    c = process_text_spam(FILES["text_spam"])

    a_out = os.path.join(DATA_OUT, "email_spam.processed.csv")
    b_out = os.path.join(DATA_OUT, "emails.processed.csv")
    c_out = os.path.join(DATA_OUT, "text_spam.processed.csv")
    a.to_csv(a_out, index=False); print(f"Saved: {a_out}  {a.shape}")
    b.to_csv(b_out, index=False); print(f"Saved: {b_out}  {b.shape}")
    c.to_csv(c_out, index=False); print(f"Saved: {c_out}  {c.shape}")

    merged = pd.concat([a, b, c], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    m_out = os.path.join(DATA_OUT, "emails_merged.processed.csv")
    merged.to_csv(m_out, index=False); print(f"Saved: {m_out}  {merged.shape}")

    # Optional stratified split
    try:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(merged, test_size=0.2, random_state=42, stratify=merged["label"])
        train.to_csv(os.path.join(DATA_OUT, "emails_merged.train.csv"), index=False)
        test.to_csv(os.path.join(DATA_OUT, "emails_merged.test.csv"), index=False)
        print("Saved: emails_merged.train.csv / emails_merged.test.csv")
    except Exception as e:
        print(f"(split skipped) {e}")

if __name__ == "__main__":
    main()
