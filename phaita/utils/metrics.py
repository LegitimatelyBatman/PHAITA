"""
Metrics for evaluating model performance and diversity.
"""

import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter
import re


def compute_diagnosis_metrics(true_labels: List[str], predicted_labels: List[str], 
                            confidences: List[float]) -> Dict[str, float]:
    """
    Compute diagnosis accuracy and related metrics.
    
    Args:
        true_labels: True condition codes
        predicted_labels: Predicted condition codes
        confidences: Prediction confidence scores
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
    
    # Precision, recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    
    metrics['precision_macro'] = precision
    metrics['recall_macro'] = recall
    metrics['f1_macro'] = f1
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )
    
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w
    
    # Confidence statistics
    metrics['mean_confidence'] = np.mean(confidences)
    metrics['std_confidence'] = np.std(confidences)
    
    # Confusion matrix statistics
    cm = confusion_matrix(true_labels, predicted_labels)
    if len(cm) > 1:
        # Calculate per-class accuracies while guarding against zero-row divisions
        row_sums = cm.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_accuracies = np.divide(
                cm.diagonal(),
                row_sums,
                out=np.zeros_like(row_sums, dtype=float),
                where=row_sums != 0
            )

        valid_mask = row_sums != 0
        if np.any(valid_mask):
            valid_accuracies = class_accuracies[valid_mask]
            metrics['min_class_accuracy'] = np.min(valid_accuracies)
            metrics['max_class_accuracy'] = np.max(valid_accuracies)
            metrics['std_class_accuracy'] = np.std(valid_accuracies)

    return metrics



def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated text.
    
    Args:
        texts: List of generated texts
    
    Returns:
        Dictionary of diversity metrics
    """
    if not texts:
        return {}
    
    metrics = {}
    
    # Lexical diversity (Type-Token Ratio)
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    if all_words:
        unique_words = len(set(all_words))
        total_words = len(all_words)
        metrics['lexical_diversity'] = unique_words / total_words
        metrics['vocabulary_size'] = unique_words
        metrics['total_words'] = total_words
    else:
        metrics['lexical_diversity'] = 0.0
        metrics['vocabulary_size'] = 0
        metrics['total_words'] = 0
    
    # Self-repetition rate
    text_counts = Counter(texts)
    duplicates = sum(1 for count in text_counts.values() if count > 1)
    metrics['repetition_rate'] = duplicates / len(texts) if texts else 0.0
    
    # Average sentence length
    avg_length = np.mean([len(text.split()) for text in texts])
    metrics['avg_sentence_length'] = avg_length
    
    # Length diversity (coefficient of variation)
    lengths = [len(text.split()) for text in texts]
    if len(lengths) > 1:
        metrics['length_diversity'] = np.std(lengths) / np.mean(lengths)
    else:
        metrics['length_diversity'] = 0.0
    
    # N-gram diversity
    metrics.update(_compute_ngram_diversity(texts))
    
    return metrics


def _compute_ngram_diversity(texts: List[str], n_values: List[int] = [2, 3, 4]) -> Dict[str, float]:
    """
    Compute n-gram diversity metrics.
    
    Args:
        texts: List of texts
        n_values: List of n-gram sizes to compute
    
    Returns:
        Dictionary of n-gram diversity metrics
    """
    metrics = {}
    
    for n in n_values:
        all_ngrams = []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            all_ngrams.extend(ngrams)
        
        if all_ngrams:
            unique_ngrams = len(set(all_ngrams))
            total_ngrams = len(all_ngrams)
            diversity = unique_ngrams / total_ngrams
        else:
            diversity = 0.0
        
        metrics[f'{n}gram_diversity'] = diversity
    
    return metrics


def compute_semantic_similarity_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute semantic similarity metrics from embeddings.
    
    Args:
        embeddings: Array of shape [num_texts, embedding_dim]
    
    Returns:
        Dictionary of similarity metrics
    """
    if len(embeddings) < 2:
        return {}
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)
    
    # Compute pairwise similarities
    similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Remove diagonal (self-similarities)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    off_diagonal = similarities[mask]
    
    metrics = {
        'mean_similarity': np.mean(off_diagonal),
        'std_similarity': np.std(off_diagonal),
        'min_similarity': np.min(off_diagonal),
        'max_similarity': np.max(off_diagonal),
        'semantic_diversity': 1.0 - np.mean(off_diagonal)  # Inverse of similarity
    }
    
    return metrics


def compute_medical_relevance_metrics(texts: List[str], medical_terms: Set[str]) -> Dict[str, float]:
    """
    Compute medical relevance metrics.
    
    Args:
        texts: List of generated texts
        medical_terms: Set of relevant medical terms
    
    Returns:
        Dictionary of medical relevance metrics
    """
    if not texts or not medical_terms:
        return {}
    
    metrics = {}
    
    # Medical term coverage
    found_terms = set()
    term_frequencies = []
    
    for text in texts:
        words = set(re.findall(r'\b\w+\b', text.lower()))
        text_medical_terms = words.intersection(medical_terms)
        found_terms.update(text_medical_terms)
        term_frequencies.append(len(text_medical_terms))
    
    metrics['medical_term_coverage'] = len(found_terms) / len(medical_terms)
    metrics['avg_medical_terms_per_text'] = np.mean(term_frequencies)
    metrics['std_medical_terms_per_text'] = np.std(term_frequencies)
    
    return metrics