"""RAG Evaluation Functions"""

import numpy as np
from typing import List, Dict

def is_relevant_result(retrieved_metadata: Dict, test_case: Dict) -> bool:
    """Check if retrieved document is relevant"""
    product_match = retrieved_metadata['product'] in test_case['relevant_products']
    category_match = retrieved_metadata['category'] in test_case['relevant_categories']
    return product_match and category_match

def calculate_precision_at_k(retrieved_metadatas: List[Dict], test_case: Dict, k: int) -> float:
    """Precision@K: (# relevant in top K) / K"""
    top_k = retrieved_metadatas[:k]
    relevant_count = sum(1 for meta in top_k if is_relevant_result(meta, test_case))
    return relevant_count / k if k > 0 else 0

def calculate_recall_at_k(retrieved_metadatas: List[Dict], test_case: Dict, k: int, total_relevant: int) -> float:
    """Recall@K: (# relevant retrieved) / (total relevant)"""
    top_k = retrieved_metadatas[:k]
    found_count = sum(1 for meta in top_k if is_relevant_result(meta, test_case))
    return found_count / total_relevant if total_relevant > 0 else 0

def calculate_hit_rate(retrieved_metadatas: List[Dict], test_case: Dict, k: int) -> float:
    """Hit Rate: Found at least one relevant doc?"""
    top_k = retrieved_metadatas[:k]
    for meta in top_k:
        if is_relevant_result(meta, test_case):
            return 1.0
    return 0.0

def calculate_mrr(retrieved_metadatas: List[Dict], test_case: Dict) -> float:
    """MRR: 1 / (rank of first relevant doc)"""
    for rank, meta in enumerate(retrieved_metadatas, start=1):
        if is_relevant_result(meta, test_case):
            return 1.0 / rank
    return 0.0

def evaluate_rag_system(rag_system, test_cases: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict:
    """Run full evaluation"""
    results = []

    for test_case in test_cases:
        query = test_case['query']
        search_results = rag_system.search(query, n_results=max(k_values))
        retrieved_metadatas = search_results['metadatas']

        result = {
            'test_id': test_case['id'],
            'query': query,
        }

        for k in k_values:
            result[f'precision@{k}'] = calculate_precision_at_k(retrieved_metadatas, test_case, k)
            result[f'recall@{k}'] = calculate_recall_at_k(retrieved_metadatas, test_case, k, 5)
            result[f'hit_rate@{k}'] = calculate_hit_rate(retrieved_metadatas, test_case, k)

        result['mrr'] = calculate_mrr(retrieved_metadatas, test_case)
        results.append(result)

    # Calculate averages
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f'precision@{k}'] = np.mean([r[f'precision@{k}'] for r in results])
        avg_metrics[f'recall@{k}'] = np.mean([r[f'recall@{k}'] for r in results])
        avg_metrics[f'hit_rate@{k}'] = np.mean([r[f'hit_rate@{k}'] for r in results])
    avg_metrics['mrr'] = np.mean([r['mrr'] for r in results])

    return {
        'detailed_results': results,
        'average_metrics': avg_metrics
    }
