# Standard library imports
import json
import logging
import re
from pathlib import Path
import sys
from typing import List, Dict, Set, Tuple, Any, Optional # Added Optional
from collections import Counter

# Third-party imports
import numpy as np
import torch # Retained as embeddings.py (imported by this file) uses it.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import joblib # Removed as joblib was not used in the provided code.

# Local application/module imports
from embeddings import embed_text # For SciBERT based semantic ranking

# --- Configuration for Logging ---
# It's good practice to configure logging if it's used.
# If not configured, logging.info() might not output anything by default.
# Example basic configuration (can be placed at the start of your main script, e.g., run_pipeline.py):
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- File Loading ---

def load_candidates_from_json(path_str: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Loads paper candidates, the query title, and domain terms from a JSON file.

    Args:
        path_str: The string path to the JSON file.

    Returns:
        A tuple containing:
            - The query title (str).
            - A list of valid paper dictionaries (List[Dict[str, Any]]).
            - A list of domain terms (List[str]).
        Returns empty structures if the file is not found or data is missing.
    """
    file_path = Path(path_str)
    if not file_path.exists():
        logging.error(f"Candidate file not found: {path_str}")
        return "", [], []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            container = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path_str}: {e}")
        return "", [], []
    except Exception as e:
        logging.error(f"Error reading file {path_str}: {e}")
        return "", [], []

    query_title: str = container.get("query_title", "")
    papers_data: List[Dict[str, Any]] = container.get("papers", [])
    domain_terms_data: List[str] = container.get("domain_terms", [])

    # Validate essential keys in each paper dictionary
    required_keys = ("doi", "title", "abstract", "year", "authors")
    valid_papers = [p for p in papers_data if isinstance(p, dict) and all(k in p for k in required_keys)]
    
    if len(valid_papers) < len(papers_data):
        logging.warning(
            f"Loaded {len(valid_papers)} valid papers out of {len(papers_data)} total entries from JSON. "
            f"{len(papers_data) - len(valid_papers)} entries were missing required keys."
        )
    else:
        logging.info(f"Loaded {len(valid_papers)} valid papers from JSON.")

    return query_title, valid_papers, domain_terms_data

# --- Filtering Functions ---

def filter_by_doi(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes paper entries that do not have a valid 'doi' key or have an empty/whitespace DOI string.

    Args:
        papers: A list of paper dictionaries.

    Returns:
        A new list containing only papers with valid DOIs.
    """
    initial_count = len(papers)
    # A paper is kept if it has a 'doi' key, the value is a string, and the string is not empty after stripping whitespace.
    filtered_papers = [
        p for p in papers 
        if p.get("doi") and isinstance(p.get("doi"), str) and str(p.get("doi", "")).strip()
    ]
    final_count = len(filtered_papers)
    logging.info(f"DOI filter: {initial_count} → {final_count} papers (removed {initial_count - final_count})")
    return filtered_papers

def filter_by_abstract(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes paper entries that do not have a valid 'abstract' key or have an empty/whitespace abstract string.

    Args:
        papers: A list of paper dictionaries.

    Returns:
        A new list containing only papers with valid abstracts.
    """
    initial_count = len(papers)
    # A paper is kept if it has an 'abstract' key, the value is a string, and the string is not empty after stripping.
    filtered_papers = [
        p for p in papers
        if p.get("abstract") and isinstance(p.get("abstract"), str) and str(p.get("abstract", "")).strip()
    ]
    final_count = len(filtered_papers)
    logging.info(f"Abstract filter: {initial_count} → {final_count} papers (removed {initial_count - final_count})")
    return filtered_papers

def dedupe_by_doi(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates a list of papers based on their DOI, keeping the first occurrence encountered.
    DOI matching is case-insensitive.

    Args:
        papers: A list of paper dictionaries. Assumes 'doi' key is present and valid.

    Returns:
        A new list containing papers with unique DOIs.
    """
    initial_count = len(papers)
    seen_dois: Set[str] = set()
    unique_papers: List[Dict[str, Any]] = []
    for paper_item in papers:
        # Ensure DOI exists and is a string before processing
        doi_value = paper_item.get("doi")
        if isinstance(doi_value, str):
            normalized_doi = doi_value.strip().lower()
            if normalized_doi and normalized_doi not in seen_dois: # Also check if normalized_doi is not empty
                unique_papers.append(paper_item)
                seen_dois.add(normalized_doi)
        else:
            # Optionally, keep papers with missing/invalid DOIs if desired, or log them.
            # For strict deduplication, papers without DOIs might be passed through or handled separately.
            # Current behavior: if DOI is not a string or missing, the paper won't be added to unique_papers unless it's the first such case and 'None' is added to seen_dois.
            # To be safe, let's assume we only process papers with valid string DOIs here for deduplication.
            # If a paper made it this far without a string DOI, it might indicate an issue upstream or a choice to keep it.
            # For this function's purpose (deduplication by DOI), it will effectively ignore papers without string DOIs for the 'seen' check.
            # To ensure such papers are also passed through if they should be, one might add:
            # if not isinstance(doi_value, str) or not doi_value.strip():
            #     unique_papers.append(paper_item) # Pass through papers without a valid DOI for dedupe.
            # This part depends on desired behavior for items lacking a proper DOI for deduplication.
            # The original code implicitly skips them from the `seen` check if `doi` is not a string.
            # To match original logic of only adding if DOI is new:
            pass # If DOI is not a string, it won't be in seen_dois, and won't be added unless this is the only one

    final_count = len(unique_papers)
    logging.info(f"Deduplication by DOI: {initial_count} → {final_count} papers (removed {initial_count - final_count} duplicates)")
    return unique_papers

# --- Test Functions for Filters ---
# These functions are typically used for module-level testing and not part of the main pipeline execution.

def _test_doi_filter():
    """Tests the filter_by_doi function."""
    sample_papers = [
        {"doi": "10.123/abc", "title": "A", "abstract": "x", "year": 2021, "authors": []},
        {"title": "No DOI", "abstract": "y", "year": 2022, "authors": []}, # No 'doi' key
        {"doi": "", "title": "Empty DOI", "abstract": "z", "year": 2020, "authors": []}, # Empty 'doi' string
        {"doi": " ", "title": "Whitespace DOI", "abstract": "w", "year": 2020, "authors": []}, # Whitespace 'doi'
        {"doi": None, "title": "None DOI", "abstract": "v", "year": 2020, "authors": []}, # None 'doi'
    ]
    result = filter_by_doi(sample_papers)
    assert len(result) == 1 and result[0]["doi"] == "10.123/abc"
    print("✔️ DOI filter test passed")
    
def _test_abstract_filter():
    """Tests the filter_by_abstract function."""
    sample_papers = [
        {"doi": "1", "abstract": "Valid abstract", "title": "A", "year": 2021, "authors": []},
        {"doi": "2", "abstract": "", "title": "B", "year": 2022, "authors": []}, # Empty abstract
        {"doi": "3", "title": "C", "year": 2020, "authors": []},  # Missing 'abstract' key
        {"doi": "4", "abstract": "   ", "title": "D", "year": 2020, "authors": []}, # Whitespace abstract
    ]
    result = filter_by_abstract(sample_papers)
    assert len(result) == 1 and result[0]["doi"] == "1"
    print("✔️ Abstract filter test passed")
    
def _test_dedupe():
    """Tests the dedupe_by_doi function."""
    sample_papers = [
        {"doi": "10.123/ABC", "title": "Paper One", "abstract": "x", "year": 2021, "authors": []},
        {"doi": "10.123/abc", "title": "Paper One Duplicate", "abstract": "y", "year": 2022, "authors": []}, # Duplicate DOI, different case
        {"doi": "10.456/DEF", "title": "Paper Two", "abstract": "z", "year": 2020, "authors": []},
        {"doi": "10.123/ABC ", "title": "Paper One Trim", "abstract": "w", "year": 2023, "authors": []}, # Duplicate DOI with trailing space
    ]
    result = dedupe_by_doi(sample_papers)
    # Expected: First "10.123/ABC" and "10.456/DEF"
    assert len(result) == 2 
    result_dois = {p["doi"].strip().lower() for p in result}
    assert "10.123/abc" in result_dois
    assert "10.456/def" in result_dois
    print("✔️ Deduplication by DOI test passed")
    
# --- Ranking Functions ---

def rank_papers(
    query: str,
    papers: List[Dict[str, Any]],
    top_n: int = 10,
    boost_domain: bool = True, # Parameter to control domain boosting
    boost_term: str = "molecular", # Example term for boosting
    boost_factor: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Ranks papers based on TF-IDF similarity (unigrams and bigrams) to a query.
    Optionally boosts scores for papers containing a specific domain term.
    Scores are normalized to a [0, 1] range.

    Args:
        query: The search query string.
        papers: A list of paper dictionaries to rank.
        top_n: The number of top papers to return.
        boost_domain: Whether to apply domain term boosting.
        boost_term: The term to check for boosting.
        boost_factor: The factor by which to boost scores (e.g., 0.15 for 15% boost).

    Returns:
        A list of top_n ranked paper dictionaries, each with an added 'score' key.
    """
    if not papers:
        logging.warning("rank_papers called with an empty list of papers.")
        return []

    # 1. Concatenate title and abstract for TF-IDF analysis
    texts: List[str] = [f"{p.get('title','')} {p.get('abstract','')}" for p in papers]

    # 2. Initialize and fit TF-IDF Vectorizer
    # Considers both single words (unigrams) and two-word phrases (bigrams).
    # Ignores common English stop words and terms appearing too frequently or too rarely.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8, # Ignore terms that appear in more than 80% of documents
        min_df=1,   # Ignore terms that appear in only one document (can be adjusted)
        ngram_range=(1, 2) # Use unigrams and bigrams
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        # This can happen if all texts are empty after preprocessing by TfidfVectorizer
        logging.error(f"TF-IDF Vectorization failed: {e}. This might be due to empty texts or all stop words.")
        return papers # Return unranked or handle error appropriately

    # 3. Transform the query into a TF-IDF vector
    query_vec = vectorizer.transform([query])

    # 4. Calculate cosine similarity between the query vector and all paper vectors
    raw_scores: np.ndarray = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 5. Optionally apply a boost to scores if a specific domain term is present
    boosted_scores_list: List[float] = []
    if boost_domain and boost_term: # Ensure boost_term is not empty
        boost_term_lower = boost_term.lower()
        for score, text_content in zip(raw_scores, texts):
            if boost_term_lower in text_content.lower():
                score *= (1 + boost_factor)
            boosted_scores_list.append(float(score))
    else:
        boosted_scores_list = [float(s) for s in raw_scores]

    # 6. Normalize scores to a [0, 1] range for better comparability
    # Handles cases where all scores are the same or list is empty.
    if not boosted_scores_list: # Should not happen if papers list was not empty
        normalized_scores = []
    else:
        max_s = max(boosted_scores_list)
        min_s = min(boosted_scores_list)
        score_range = max_s - min_s
        if score_range == 0: # All scores are the same
            # Assign a mid-range score (e.g., 0.5) or 0, depending on preference for identical items.
            # Or, if all scores are positive, could assign 1. Here, assign 0 if all are same and 0, else 0.5.
            normalized_scores = [0.5 if max_s > 0 else 0.0 for _ in boosted_scores_list]
        else:
            normalized_scores = [(s - min_s) / score_range for s in boosted_scores_list]

    # 7. Annotate papers with their normalized scores and sort
    for paper_item, norm_score in zip(papers, normalized_scores):
        paper_item['score'] = norm_score # Ensure it's float

    ranked_papers = sorted(papers, key=lambda x: x.get('score', 0.0), reverse=True)
    return ranked_papers[:top_n]

def _test_rank_papers():
    """Minimal test for the TF-IDF based rank_papers function."""
    sample_papers = [
        {"title": "Deep learning in robotics", "abstract": "We apply deep nets for robotics", "doi":"d1", "year":2020, "authors":[]},
        {"title": "Classical control methods", "abstract": "PID controllers for systems", "doi":"d2", "year":2018, "authors":[]},
        {"title": "Reinforcement learning for games", "abstract": "RL algorithms in robotics", "doi":"d3", "year":2019, "authors":[]},
    ]
    ranked = rank_papers("deep reinforcement learning in robotics", sample_papers, top_n=2, boost_term="robotics")
    assert len(ranked) == 2
    # Specific assertion about which paper is first can be tricky due to TF-IDF nuances
    # but for this query, d1 or d3 are likely candidates.
    if ranked: # Check if ranked is not empty
        print(f"✔️ Rank papers (TF-IDF) test passed (Top 1 DOI: {ranked[0]['doi']})")
    else:
        print("⚠️ Rank papers (TF-IDF) test produced empty result.")

def semantic_rank_papers(
    query: str,
    papers: List[Dict[str, Any]],
    top_n: Optional[int] = None, # Allow ranking all papers if top_n is None
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Ranks papers based on cosine similarity of their SciBERT embeddings to the query embedding.

    Args:
        query: The search query string.
        papers: A list of paper dictionaries to rank.
        top_n: The number of top papers to return. If None, returns all ranked papers.
        use_cache: Whether to use caching for SciBERT embeddings.

    Returns:
        A list of ranked paper dictionaries, each with an added 'score' (cosine similarity).
    """
    if not papers:
        logging.warning("semantic_rank_papers called with an empty list of papers.")
        return []

    # 1. Compute or load SciBERT embeddings for each paper's title and abstract
    doc_vectors: List[np.ndarray] = []
    valid_papers_for_ranking: List[Dict[str, Any]] = [] # Papers for which embedding was successful

    for paper_item in papers:
        text_content = f"{paper_item.get('title','')} {paper_item.get('abstract','')}"
        if not text_content.strip(): # Skip if no text content for embedding
            logging.warning(f"Paper with DOI {paper_item.get('doi', 'N/A')} has no text for embedding. Skipping.")
            continue
        try:
            vector = embed_text(text_content, use_cache=use_cache)
            if vector is not None: # Ensure embed_text did not return None
                 doc_vectors.append(vector)
                 valid_papers_for_ranking.append(paper_item) # Keep track of papers that were embedded
            else:
                logging.warning(f"Embedding for paper DOI {paper_item.get('doi', 'N/A')} resulted in None. Skipping.")

        except Exception as e:
            logging.error(f"Error embedding text for paper DOI {paper_item.get('doi', 'N/A')}: {e}")
            # Optionally, assign a very low score or skip this paper

    if not doc_vectors: # If no papers could be embedded
        logging.warning("No document vectors could be generated for semantic ranking.")
        # Return original papers, perhaps with a default low score or unranked
        for p in papers: p['score'] = 0.0
        return papers[:top_n] if top_n is not None else papers


    doc_matrix: np.ndarray = np.vstack(doc_vectors) # Stack into a matrix (num_papers, embedding_dim)

    # 2. Embed the query
    try:
        query_vector: Optional[np.ndarray] = embed_text(query, use_cache=use_cache)
        if query_vector is None: # Handle case where query embedding fails
            logging.error(f"Failed to embed query: '{query}'. Returning unranked papers.")
            for p in valid_papers_for_ranking: p['score'] = 0.0
            return valid_papers_for_ranking[:top_n] if top_n is not None else valid_papers_for_ranking
    except Exception as e:
        logging.error(f"Error embedding query '{query}': {e}")
        for p in valid_papers_for_ranking: p['score'] = 0.0
        return valid_papers_for_ranking[:top_n] if top_n is not None else valid_papers_for_ranking

    # 3. Compute cosine similarities
    # Ensure correct calculation and handle potential normalization issues
    # sim = (X @ Y.T) / (||X|| * ||Y||)
    # Here, query_vector is Y.T (effectively) and doc_matrix is X
    
    # Normalize document vectors and query vector (optional, cosine_similarity handles it, but explicit can be clearer)
    # doc_matrix_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9)
    # query_vector_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
    # similarities = doc_matrix_norm @ query_vector_norm

    # Using sklearn's cosine_similarity is often more robust and handles normalization.
    # It expects 2D arrays, so reshape query_vector.
    similarities: np.ndarray = cosine_similarity(doc_matrix, query_vector.reshape(1, -1)).flatten()

    # 4. Attach scores to the papers that were successfully embedded and sort
    for paper_item, sim_score in zip(valid_papers_for_ranking, similarities):
        paper_item["score"] = float(sim_score)

    # Add papers that couldn't be ranked (e.g., due to embedding failure) back with a very low score
    # This ensures the output list contains all original papers if desired, or only ranked ones.
    # For now, we only rank and return from valid_papers_for_ranking.
    
    ranked_papers = sorted(valid_papers_for_ranking, key=lambda p_item: p_item.get("score", 0.0), reverse=True)
    
    if top_n is not None:
        return ranked_papers[:top_n]
    return ranked_papers


def _test_semantic_rank():
    """Minimal test for the semantic_rank_papers function."""
    sample_papers = [
        {"title": "Molecular property prediction for novel drugs", "abstract": "We test computational methods on ADMET benchmarks.", "doi":"d1","year":2020,"authors":[]},
        {"title": "Graph signal processing in complex networks", "abstract": "Novel sampling techniques on large graphs.", "doi":"d2","year":2020,"authors":[]},
    ]
    # This test requires embed_text to be functional.
    # Assuming embed_text works, the first paper should rank higher for the given query.
    try:
        ranked = semantic_rank_papers("molecular property prediction", sample_papers, top_n=2)
        if ranked and ranked[0]["doi"] == "d1":
            print("✔️ Semantic rank test passed")
        elif ranked:
            print(f"⚠️ Semantic rank test produced results, but unexpected order. Top DOI: {ranked[0]['doi']}")
        else:
            print("⚠️ Semantic rank test produced no results (possibly due to embedding issues).")
    except Exception as e:
        print(f"❌ Semantic rank test failed with error: {e}")


# --- Term-based Filtering Functions ---

def filter_by_terms(
    papers: List[Dict[str, Any]],
    terms_to_match: Set[str],
    filter_name: str = "Generic Term" # For logging
) -> List[Dict[str, Any]]:
    """
    A generic function to keep only papers whose title or abstract contains at least one term
    from the provided set of terms. Case-insensitive, whole-word matching.

    Args:
        papers: A list of paper dictionaries.
        terms_to_match: A set of strings (terms) to search for.
        filter_name: A name for this filtering step, used in logging.

    Returns:
        A new list of paper dictionaries that match at least one term.
    """
    if not terms_to_match: # If no terms are provided, no filtering occurs
        logging.info(f"{filter_name} filter: No terms provided, returning all {len(papers)} papers.")
        return papers
        
    initial_count = len(papers)
    # Compile regex patterns for whole-word, case-insensitive matching
    regex_patterns = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in terms_to_match if term # Ensure term is not empty
    ]

    if not regex_patterns: # If all terms were empty, effectively no patterns to match
        logging.info(f"{filter_name} filter: No valid regex patterns generated, returning all {len(papers)} papers.")
        return papers

    filtered_papers = []
    for paper_item in papers:
        text_content = f"{paper_item.get('title','')} {paper_item.get('abstract','')}".lower() # Lowercase once
        if any(pattern.search(text_content) for pattern in regex_patterns):
            filtered_papers.append(paper_item)
    
    final_count = len(filtered_papers)
    logging.info(f"{filter_name} filter: {initial_count} → {final_count} papers (removed {initial_count - final_count})")
    return filtered_papers

def filter_by_domain(
    papers: List[Dict[str, Any]],
    domain_terms: Set[str] # Expecting a set of domain terms
) -> List[Dict[str, Any]]:
    """
    Keeps only papers whose title or abstract contains at least one of the specified domain terms.
    Uses the generic filter_by_terms function.
    """
    return filter_by_terms(papers, domain_terms, filter_name="Domain Term")

def filter_by_core(
    papers: List[Dict[str, Any]],
    core_terms: Set[str] # Expecting a set of core/application terms
) -> List[Dict[str, Any]]:
    """
    Keeps only papers whose title or abstract contains at least one of the specified core terms.
    Uses the generic filter_by_terms function.
    """
    return filter_by_terms(papers, core_terms, filter_name="Core Application Term")

# --- Boost Term Inference ---

def infer_boost_terms(
    ranked_papers: List[Dict[str, Any]],
    domain_terms: List[str], # List of domain terms to consider for boosting
    top_k_papers_for_analysis: int = 20, # Number of top papers to analyze for term frequency
    frequency_multiplier_threshold: float = 1.5 # How much more frequent a term should be in top vs. bottom
) -> List[str]:
    """
    Infers which domain terms to use for boosting scores.
    It compares the frequency of domain terms in the top-ranked papers
    against their frequency in the lower-ranked papers. Terms significantly
    more frequent in top papers are selected as boost terms.

    Args:
        ranked_papers: A list of paper dictionaries, pre-sorted by relevance/score.
        domain_terms: A list of candidate domain terms to evaluate for boosting.
        top_k_papers_for_analysis: The number of top papers to consider for "high relevance" set.
        frequency_multiplier_threshold: The minimum "lift" (ratio of frequencies)
                                         for a term to be selected.

    Returns:
        A list of domain terms identified as good candidates for boosting.
    """
    if not ranked_papers or not domain_terms:
        return []

    # Prepare text content from top K and remaining (bottom) papers
    top_k_texts: List[str] = [
        f"{p.get('title','')} {p.get('abstract','')}".lower() 
        for p in ranked_papers[:top_k_papers_for_analysis]
    ]
    bottom_texts: List[str] = [
        f"{p.get('title','')} {p.get('abstract','')}".lower() 
        for p in ranked_papers[top_k_papers_for_analysis:]
    ]

    if not top_k_texts: # Not enough papers for analysis
        return []

    # Count occurrences of each domain term in top and bottom sets
    # Using Counter for efficient counting.
    top_term_counts = Counter()
    for text_content in top_k_texts:
        for term in domain_terms:
            if term.lower() in text_content: # Simple substring check, could use regex for whole word
                top_term_counts[term.lower()] += 1
    
    bottom_term_counts = Counter()
    if bottom_texts: # Only count if there are bottom texts
        for text_content in bottom_texts:
            for term in domain_terms:
                if term.lower() in text_content:
                    bottom_term_counts[term.lower()] += 1
    
    num_top_papers = len(top_k_texts)
    num_bottom_papers = len(bottom_texts) if bottom_texts else 1 # Avoid division by zero if bottom_texts is empty

    selected_boost_terms: List[str] = []
    for term in domain_terms:
        term_lower = term.lower()
        # Calculate frequency (proportion of documents containing the term)
        freq_in_top = top_term_counts[term_lower] / num_top_papers
        freq_in_bottom = bottom_term_counts[term_lower] / num_bottom_papers
        
        # Calculate lift: ratio of frequency in top vs. bottom. Add epsilon to avoid division by zero.
        lift = freq_in_top / (freq_in_bottom + 1e-9) 
        
        if lift >= frequency_multiplier_threshold:
            selected_boost_terms.append(term) # Keep original casing of the term
            
    logging.info(f"Inferred boost terms (lift >= {frequency_multiplier_threshold}): {selected_boost_terms}")
    return selected_boost_terms

# --- Main Block for Module Testing ---

if __name__ == "__main__":
    # This block is for testing the functions within this module directly.
    # Configure basic logging for test outputs.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s (%(funcName)s): %(message)s')

    # --- 1. Load raw candidate papers from a JSON file ---
    # Assumes "raw_candidates.json" exists in the same directory when running this script.
    # This file should be generated by paper_collector.py / run_pipeline.py first.
    test_query_title, test_candidate_papers, test_domain_terms = load_candidates_from_json("raw_candidates.json")
    
    if not test_candidate_papers:
        # Using print here as logging might not be fully set up if load_candidates_from_json fails early.
        print("❌ No valid candidate papers loaded from 'raw_candidates.json'. Ensure the file exists and is valid.")
        print("Skipping further tests in filter_and_rank.py.")
        sys.exit(1) # Exit if no papers to test with.

    print(f"\n--- Running Internal Tests for filter_and_rank.py ---")
    print(f"Loaded {len(test_candidate_papers)} papers for query: '{test_query_title}'")

    # --- Test individual filter functions ---
    _test_doi_filter()
    _test_abstract_filter()
    _test_dedupe()
    
    # --- Apply filters sequentially as an example workflow ---
    print("\n--- Applying Filters to Loaded Data ---")
    papers_after_doi_filter = filter_by_doi(test_candidate_papers)
    if not papers_after_doi_filter:
        print("❌ All candidates dropped by DOI filter—cannot proceed with further tests requiring papers.")
        sys.exit(1)
    
    papers_after_abstract_filter = filter_by_abstract(papers_after_doi_filter)
    if not papers_after_abstract_filter:
        print("❌ All candidates dropped by abstract filter—cannot proceed with further tests requiring papers.")
        sys.exit(1)

    final_papers_for_ranking = dedupe_by_doi(papers_after_abstract_filter) # Deduplicate before ranking
    if not final_papers_for_ranking:
        print("❌ All candidates dropped by deduplication—cannot proceed with ranking tests.")
        sys.exit(1)
    
    print(f"Papers remaining for ranking tests: {len(final_papers_for_ranking)}")

    # --- Test ranking functions ---
    print("\n--- Testing Ranking Functions ---")
    _test_rank_papers() # Tests TF-IDF ranker with its own sample data
    
    # Test semantic ranking with the loaded papers if a query title is available
    if test_query_title and final_papers_for_ranking:
        print(f"\nTesting semantic ranking with query: '{test_query_title}' on {len(final_papers_for_ranking)} papers...")
        top_semantically_ranked = semantic_rank_papers(test_query_title, final_papers_for_ranking, top_n=5)
        print(f"Top 5 semantically ranked papers for '{test_query_title}':")
        for i, p in enumerate(top_semantically_ranked):
            print(f"  {i+1}. DOI: {p.get('doi','N/A')}, Score: {p.get('score',0.0):.4f}, Title: {p.get('title','N/A')[:60]}...")
    else:
        print("Skipping semantic ranking test with loaded data (no query title or no papers after filtering).")
    _test_semantic_rank() # Tests semantic ranker with its own internal sample

    # --- Test term-based filters ---
    print("\n--- Testing Term-Based Filters ---")
    if final_papers_for_ranking and test_domain_terms:
        example_domain_terms_set = set(test_domain_terms[:3]) # Use a subset for testing
        print(f"Testing domain filter with terms: {example_domain_terms_set}")
        papers_after_domain_filter = filter_by_domain(final_papers_for_ranking, example_domain_terms_set)
        # Further tests could assert len(papers_after_domain_filter) or specific content.
        
        example_core_terms_set = {"molecular", "prediction"} # Example core terms
        print(f"Testing core filter with terms: {example_core_terms_set}")
        papers_after_core_filter = filter_by_core(final_papers_for_ranking, example_core_terms_set)

    # --- Test infer_boost_terms ---
    print("\n--- Testing Boost Term Inference ---")
    if final_papers_for_ranking and test_domain_terms:
        # Ensure papers are ranked before inferring boost terms (using existing scores or re-ranking)
        # For this test, assume final_papers_for_ranking might already have 'score' from a previous step, or sort by year as a proxy
        sorted_for_boost_test = sorted(final_papers_for_ranking, key=lambda x: x.get("year", 0), reverse=True)
        inferred_terms = infer_boost_terms(sorted_for_boost_test, test_domain_terms)
        print(f"Inferred boost terms from loaded data: {inferred_terms}")

    print("\n--- All Internal Tests for filter_and_rank.py Complete ---")
