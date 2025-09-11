import numpy as np
from sentence_transformers import SentenceTransformer, util

# Sentence transformer encoder for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def score_summary(summary: str, topic_keywords: str) -> float:
    # Semantic cosine similarity
    semantic_score = float(util.cos_sim(model.encode(summary), model.encode(topic_keywords)))
    # Conciseness (reward summaries ~80-150 words)
    words = summary.split()
    conciseness_score = min(1.0, 140 / max(60, len(words)))
    # Keyword/topic coverage ratio
    keywords = set(topic_keywords.lower().split())
    summary_words = set(summary.lower().split())
    coverage = len(keywords & summary_words) / max(1, len(keywords))
    # Weighted sum for robust ranking
    return 0.6 * semantic_score + 0.25 * coverage + 0.15 * conciseness_score

def choose_best(summaries: list[str], topic_keywords: str):
    scores = [score_summary(s, topic_keywords) for s in summaries]
    best_index = int(np.argmax(scores))
    return best_index, scores
