# pip install pandas torch transformers tqdm deep-translator textblob vaderSentiment openpyxl langdetect
# python -m textblob.download_corpora

import pandas as pd
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from deep_translator import GoogleTranslator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =========================
# Load data
# =========================
df1 = pd.read_excel("combined4_clean.xlsx")

df = df1.iloc[1:10].copy()

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["month_year"] = df["datetime"].dt.to_period("M")

# =========================
# Basic cleaning
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def remove_hashtags(text):
    if not text:
        return ""
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def remove_underscore_words(text):
    if not text:
        return ""
    
    # remove words that contain at least one underscore
    text = re.sub(r"\b\w*_\w*\b", "", text)
    
    # clean extra spaces after removal
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# =========================
# Remove repeated punctuation
# e.g. "great!!!" -> "great!"
#      "what???" -> "what?"
#      "wow...." -> "wow."
# =========================
def remove_repeated_punctuation(text):
    if not text:
        return ""
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
df["post_text_original"] = df["post_text"].apply(clean_text)
df["post_text_original"] = df["post_text_original"].apply(remove_hashtags)
df["post_text_original"] = df["post_text_original"].apply(remove_underscore_words)
df["post_text_original"] = df["post_text_original"].apply(remove_repeated_punctuation)


# =========================
# Translation to English
# =========================


import time
import random
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from tqdm.auto import tqdm

translator = GoogleTranslator(source="auto", target="en")

def translate_to_english(text):
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text).strip()

    for attempt in range(3):
        try:
            result = translator.translate(text)

            # Delay between successful requests
            time.sleep(random.uniform(2, 4))
            return result

        except TooManyRequests:
            wait_time = 60 * (attempt + 1)
            print(f"Rate limited. Waiting {wait_time} seconds.")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Translation error: {e}")
            return None

    print("Translation failed after 3 attempts.")
    return None


tqdm.pandas(desc="Translating posts")

df["post_text_original_english"] = (
    df["post_text_original"]
    .progress_apply(translate_to_english)
)

# =========================
# Load Indic sentiment model
# =========================
MODEL_NAME = "ai4bharat/IndicBERTv2-alpha-SentimentClassification"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Number of labels:", model.config.num_labels)
print("id2label:", getattr(model.config, "id2label", None))

# =========================
# Indic sentiment batch function
# =========================
def infer_indic_sentiment_batch(texts, batch_size=16, max_length=256):
    all_neg, all_pos, all_compound, all_label = [], [], [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Running IndicSent"):
        batch_texts = texts[i:i + batch_size]

        empty_mask = [not t for t in batch_texts]
        safe_texts = [t if t else "[PAD]" for t in batch_texts]

        inputs = tokenizer(
            safe_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for is_empty, prob in zip(empty_mask, probs):
            if is_empty:
                all_neg.append(None)
                all_pos.append(None)
                all_compound.append(None)
                all_label.append(None)
            else:
                neg = float(prob[0])
                pos = float(prob[1])
                compound = pos - neg
                label = "positive" if pos >= neg else "negative"

                all_neg.append(neg)
                all_pos.append(pos)
                all_compound.append(compound)
                all_label.append(label)

    return all_neg, all_pos, all_compound, all_label

# =========================
# Apply IndicSent to two original columns
# =========================
for col in ["post_text_original"]:
    neg, pos, compound, label = infer_indic_sentiment_batch(
        df[col].tolist(),
        batch_size=16,
        max_length=256
    )

    df[f"{col}_indic_neg"] = neg
    df[f"{col}_indic_pos"] = pos
    df[f"{col}_indic_compound"] = compound
    df[f"{col}_indic_label"] = label

# =========================
# TextBlob + VADER for translated columns
# =========================
vader = SentimentIntensityAnalyzer()

def textblob_sentiment(text):
    if not text:
        return None
    return TextBlob(str(text)).sentiment.polarity

def vader_sentiment(text):
    if not text:
        return {
            "vader_neg": None,
            "vader_neu": None,
            "vader_pos": None,
            "vader_compound": None
        }
    scores = vader.polarity_scores(str(text))
    return {
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "vader_pos": scores["pos"],
        "vader_compound": scores["compound"]
    }

for col in ["post_text_original_english"]:
    df[f"{col}_textblob_polarity"] = df[col].apply(textblob_sentiment)

    vader_scores = df[col].apply(vader_sentiment).apply(pd.Series)
    vader_scores = vader_scores.add_prefix(f"{col}_")

    df = pd.concat([df, vader_scores], axis=1)

# =========================
# Save output
# =========================
output_filename = "combined4_sentiment_translation_1_10.xlsx"
df.to_excel(output_filename, index=False)

print(f"Saved file: {output_filename}")