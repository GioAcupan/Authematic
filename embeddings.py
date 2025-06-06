import os
from pathlib import Path
from typing import Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import hashlib

# 1. Load once at import time
MODEL_NAME = "allenai/scibert_scivocab_uncased"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model     = AutoModel.from_pretrained(MODEL_NAME)
_model.eval()  # disable dropout

# In-memory cache: { text_hash: vector }
_embed_cache: Dict[str, np.ndarray] = {}

def embed_text(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Returns a 768-dim NumPy embedding for the given text using SciBERT.
    Mean-pools the last hidden layer over all tokens.
    """
    # 2. Cache key — you can also key by DOI if embedding papers
    # Use a stable hash to ensure cache keys remain consistent across runs
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if use_cache and key in _embed_cache:
        return _embed_cache[key]

    # 3. Tokenize
    inputs = _tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # 4. Forward pass (no grad)
    with torch.no_grad():
        outputs = _model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)

    # 5. Mean-pool (ignoring pad tokens)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)                   # (1, 768)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)   # (1, 1)
    mean_pooled = summed / counts                        # (1, 768)
    vector = mean_pooled.squeeze(0).cpu().numpy()        # (768,)

    # 6. Cache & return
    if use_cache:
        _embed_cache[key] = vector
    return vector
