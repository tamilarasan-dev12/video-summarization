from transformers import pipeline, AutoTokenizer
import torch

# Lazy singletons
summarizer = None
_tokenizer = None
_MODEL_NAME = "facebook/bart-large-cnn"
# BART can encode up to 1024 tokens; keep a safety margin
_MAX_SRC_TOKENS = 900
_CHUNK_TOKENS = 800


def get_summarizer():
    """Lazy load the BART summarization pipeline."""
    global summarizer
    if summarizer is None:
        device_id = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model=_MODEL_NAME, device=device_id)
    return summarizer


def get_tokenizer():
    """Lazy load tokenizer for token-length-aware chunking."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return _tokenizer


def _token_len(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tok = get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), max_tokens):
        sub = ids[i : i + max_tokens]
        chunks.append(tok.decode(sub, skip_special_tokens=True))
    return chunks


def _summarize_block(block: str, max_len: int, min_len: int) -> str:
    model = get_summarizer()
    # Clamp min/max based on input size to avoid index errors
    tlen = _token_len(block)
    if tlen < 30:
        max_len = min(max_len, 60)
        min_len = 10
    elif tlen < 100:
        max_len = min(max_len, 120)
        min_len = min(min_len, 30)
    try:
        out = model(block, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        # Fallback: reduce lengths and retry once
        out = model(block, max_length=max(60, max_len // 2), min_length=max(10, min_len // 2), do_sample=False)
        return out[0]["summary_text"]


def summarize_text(text: str, max_len: int = 200) -> str:
    """Summarize using token-aware chunking to stay within model limits."""
    if not text or not text.strip():
        return "No content to summarize."

    # If the whole text fits within token budget, summarize directly
    if _token_len(text) <= _MAX_SRC_TOKENS:
        return _summarize_block(text, max_len=max_len, min_len=min(80, max(40, max_len // 3)))

    # Otherwise, split by tokens and map-reduce
    parts = _split_by_tokens(text, _CHUNK_TOKENS)
    partials: list[str] = []
    for p in parts:
        partials.append(_summarize_block(p, max_len=160, min_len=40))

    combined = " ".join(partials)
    # Trim combined to fit
    if _token_len(combined) > _MAX_SRC_TOKENS:
        combined = _split_by_tokens(combined, _MAX_SRC_TOKENS)[0]

    return _summarize_block(combined, max_len=max_len, min_len=min(100, max(50, max_len // 2)))
