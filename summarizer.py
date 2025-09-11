from transformers import pipeline

# Use a stronger summarization model for better accuracy
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, max_len: int = 180) -> str:
    """Summarize text using BART-large-cnn model."""
    if not text.strip():
        return "No content to summarize."
    # Truncate input if too long (BART's limit ~1024 tokens)
    words = text.split()
    if len(words) > 900:
        words = words[:900]
    truncated_text = " ".join(words)
    summary = summarizer(truncated_text, max_length=max_len, min_length=60, do_sample=False)
    return summary['summary_text']
