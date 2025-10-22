import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import torch

# Sentence transformer encoder for semantic similarity
model = None

def get_model():
    """Lazy load the SentenceTransformer model."""
    global model
    if model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model

def _compute_scores(summaries: List[str], topic_keywords: str) -> Tuple[List[float], List[Dict[str, float]]]:
    """Compute scores and diagnostics for a list of summaries vs topic."""
    transformer_model = get_model()
    # Batch encode for speed
    embeds = transformer_model.encode(summaries, normalize_embeddings=True)
    topic_emb = transformer_model.encode([topic_keywords], normalize_embeddings=True)[0]
    # Cosine similarity
    sem_scores = util.cos_sim(embeds, topic_emb).cpu().numpy().flatten().tolist()
    results: List[float] = []
    details: List[Dict[str, float]] = []
    topic_keywords_set = set(topic_keywords.lower().split())
    for i, summary in enumerate(summaries):
        words = summary.split()
        conciseness = min(1.0, 140 / max(60, len(words)))
        summ_words = set(summary.lower().split())
        coverage = len(topic_keywords_set & summ_words) / max(1, len(topic_keywords_set))
        semantic = float(sem_scores[i])
        # Weighted blend
        score = 0.65 * semantic + 0.2 * coverage + 0.15 * conciseness
        results.append(score)
        details.append({
            "semantic": round(semantic, 4),
            "coverage": round(coverage, 4),
            "conciseness": round(conciseness, 4),
            "final": round(score, 4),
            "length_words": len(words),
        })
    return results, details

def choose_best(summaries: List[str], topic_keywords: str):
    scores, details = _compute_scores(summaries, topic_keywords)
    best_index = int(np.argmax(scores))
    return best_index, scores, details
