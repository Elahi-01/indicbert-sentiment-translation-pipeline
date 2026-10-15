"""Original cleaning rules and sentiment calculations from indicbert1(5).py.
Hashtag removal includes Unicode combining marks so Hindi tags are removed
whole. Tokenizer loading has a compatibility fallback. Model, class indices,
thresholds, batch size, max length and sentiment calculations are unchanged.
"""
import re
import os
import unicodedata
import pandas as pd

_MODEL_RESOURCES = None
_VADER_ANALYZER = None

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def remove_hashtags(text):
    if not text:
        return ""
    # Python's \w excludes vowel signs/virama, so #\w+ cuts Hindi hashtags
    # midway. Keep the same remove-hashtag rule, consuming the whole tag.
    pieces = []
    keep_from = 0
    start = text.find("#")
    while start != -1:
        end = start + 1
        while end < len(text):
            character = text[end]
            if not (character.isalnum() or character == "_"
                    or unicodedata.category(character).startswith("M")
                    or character in "\u200c\u200d"):
                break
            end += 1
        if end > start + 1:
            pieces.append(text[keep_from:start])
            keep_from = end
        start = text.find("#", end)
    pieces.append(text[keep_from:])
    text = "".join(pieces)
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

def remove_repeated_punctuation(text):
    if not text:
        return ""
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_original_tokenizer(model_name, log):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except ValueError as exc:
        if "IndicBERTSentencePieceTokenizer" not in str(exc):
            raise
        from huggingface_hub import hf_hub_download
        from tokenizers import AddedToken
        log("Loading the SAME model's tokenizer.json (custom-class compatibility fix).")
        path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_file=path,
            bos_token="<s>", eos_token="</s>", unk_token="<unk>",
            sep_token="</s>", cls_token="<s>", pad_token="<pad>",
            mask_token=AddedToken("<mask>", single_word=False, lstrip=True,
                                  rstrip=False, normalized=True, special=True),
            model_max_length=512,
            model_input_names=["input_ids", "attention_mask"],
        )


def load_original_resources(log=print):
    """Load the same model once per process; keep its original precision/device."""
    global _MODEL_RESOURCES
    if _MODEL_RESOURCES is None:
        import torch
        from transformers import AutoModelForSequenceClassification
        name = "ai4bharat/IndicBERTv2-alpha-SentimentClassification"
        tokenizer = load_original_tokenizer(name, log)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _MODEL_RESOURCES = tokenizer, model, device
        log("Number of labels:", model.config.num_labels)
        log("id2label:", getattr(model.config, "id2label", None))
    return _MODEL_RESOURCES


def load_original_vader():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def add_original_sentiment_columns(df, log=print, stop_event=None, progress=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm.auto import tqdm as real_tqdm
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Stopped before sentiment analysis.")

    def tqdm(iterable, **kwargs):
        total = len(iterable)
        for count, item in enumerate(real_tqdm(iterable, disable=progress is None, **kwargs), 1):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Stopped between sentiment batches.")
            if progress:
                progress("sentiment", count, total, None)
            yield item

    # The following block is the original code, with only tokenizer loading
    # delegated above. The local print routes model details to the UI log.
    print = log

    tokenizer, model, device = load_original_resources(log)
    
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
    vader = load_original_vader()
    
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
    
    return df
