# Standard library imports
import json
import sys
from pathlib import Path
import time
import re
import math
from typing import Dict, List # Added explicit Dict and List imports

# Third-party imports
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans

# Load environment variables at the very beginning
# Assumes .env file is in the same directory as this script or project root.
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Local application/module imports
from embeddings import embed_text # For SciBERT embeddings
# Functions from paper_collector.py
from paper_collector import (
    generate_topics,
    collect_papers,
    generate_subthemes,
    generate_keywords_by_subtheme,
    generate_domain_terms,
    generate_app_terms,
    generate_tech_terms
)
# Functions from filter_and_rank.py
from filter_and_rank import (
    load_candidates_from_json,
    filter_by_doi,
    filter_by_abstract,
    dedupe_by_doi,
    semantic_rank_papers,
)
# Functions from keyword_critic.py
from keyword_critic import critique_list

# --- Global Constants and Configuration ---

# Stop terms for filtering generated domain/application/technique terms
METHODOLOGY_STOP_TERMS = { # General methodology terms, often too broad for specific matching
    "survey", "review", "framework", "architecture", "architectures",
    "method", "methods", "approach", "approaches",
    "system", "systems", "analysis", "analyses",
    "algorithm", "algorithms", "technique", "techniques"
}
STOP_TIER1 = { # Tier-1: terms to be strictly excluded initially during term cleaning
    "survey", "review", "framework", "architecture", "architectures",
    "analysis", "analyses", "system", "systems",
}
STOP_TIER2 = { # Tier-2: terms to be excluded if Tier-1 filtering results in an empty list
    "method", "methods", "approach", "approaches",
    "algorithm", "algorithms", "technique", "techniques",
}

# --- Helper Functions ---

def clean_terms(terms: List[str]) -> List[str]:
    """
    Cleans a list of terms by removing items containing predefined stop words.
    It first attempts to remove Tier-1 stop words. If this filtering makes the
    list empty, it then attempts to remove only Tier-2 stop words from the
    original list. If both cleaning stages result in an empty list, the original
    list of terms is returned.

    Args:
        terms: A list of strings, where each string is a keyword or phrase.

    Returns:
        A list of strings, cleaned according to the stop word tiers.
    """
    # First pass: remove terms containing any Tier-1 stop word
    tier1_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER1)]
    if tier1_clean:
        return tier1_clean
    
    # Second pass (if tier1_clean was empty): remove terms containing any Tier-2 stop word
    tier2_clean = [t for t in terms if set(t.split()).isdisjoint(STOP_TIER2)]
    if tier2_clean:
        return tier2_clean
        
    # Fallback: if all cleaning results in an empty list, return the original terms
    return terms

def prompt_cutoff_year() -> int:
    """
    Prompts the user to enter a publication year cutoff and validates the input.

    Returns:
        An integer representing the four-digit cutoff year.
    """
    while True:
        yr = input("Enter publication year cutoff (e.g. 2015): ").strip()
        if yr.isdigit() and len(yr) == 4: # Validate for a 4-digit year
            return int(yr)
        print("⚠️  Invalid year. Please enter a four-digit year (e.g., 2015).")

# --- Main Pipeline Orchestration ---

def run_pipeline(title: str, cutoff_year: int, citation_style: str):
    """Run the paper collection and ranking pipeline and return results."""

    # Start timing the main pipeline operations
    pipeline_start_time = time.time()
    
    # --- 2. Keyword Generation and Refinement ---
    print("\nGenerating topics and subthemes based on the research title...")
    related_topics: List[str] = generate_topics(title) # Generates a list of 4 academic topics
    subthemes_by_topic: Dict[str, List[str]] = generate_subthemes(related_topics, max_subthemes=3) # Max 3 subthemes per topic
    
    print("Generating initial keywords for each subtheme...")
    # Generates a nested dictionary: {topic: {subtheme: [keywords]}}
    raw_keywords_by_subtheme_structure: Dict[str, Dict[str, List[str]]] = \
        generate_keywords_by_subtheme(subthemes_by_topic, max_terms=5) # Max 5 keywords per subtheme

    print("Refining generated keywords using the critic AI...")
    # Critique keywords for each subtheme, preserving the nested structure
    critiqued_keywords_nested: Dict[str, Dict[str, List[str]]] = {}
    for topic, subthemes_map in raw_keywords_by_subtheme_structure.items():
        refined_subthemes_for_topic: Dict[str, List[str]] = {}
        for subtheme, original_keywords_for_subtheme in subthemes_map.items():
            critic_label = f"Keywords for topic '{topic}' under subtheme '{subtheme}'"
            # The suggestions_map from critique_list is not used for these keywords, so assign to _
            refined_list, _ = critique_list(
                critic_label,
                original_keywords_for_subtheme
            )
            refined_subthemes_for_topic[subtheme] = refined_list
        critiqued_keywords_nested[topic] = refined_subthemes_for_topic
    print("✅ Sub-theme keywords refined via critic.")

    # Warning if the number of generated topics deviates from the expected count
    if len(related_topics) != 4:
        print(f"⚠️ Warning: Expected 4 topics from generate_topics, but got {len(related_topics)}.")
    
    # --- 3. Paper Collection ---
    # Configuration for how many papers each API source should attempt to fetch for a keyword
    papers_to_fetch_per_keyword_source: int = 3
    
    # # Optional: Debug print for inspecting the structure passed to collect_papers
    # print("\n--- DEBUGGING: Data being passed to collect_papers ---")
    # print(f"Variable name: critiqued_keywords_nested")
    # print(f"Type: {type(critiqued_keywords_nested)}")
    # if isinstance(critiqued_keywords_nested, dict) and critiqued_keywords_nested:
    #     first_topic_key_debug = list(critiqued_keywords_nested.keys())[0]
    #     print(f"First topic key: '{first_topic_key_debug}'")
    #     value_for_first_topic_debug = critiqued_keywords_nested[first_topic_key_debug]
    #     print(f"Value type for first topic: {type(value_for_first_topic_debug)}")
    #     if isinstance(value_for_first_topic_debug, dict) and value_for_first_topic_debug:
    #         first_subtheme_key_debug = list(value_for_first_topic_debug.keys())[0]
    #         print(f"  First subtheme key: '{first_subtheme_key_debug}'")
    # print("--- END DEBUG ---\n")

    print(f"\nCollecting papers (target {papers_to_fetch_per_keyword_source} per keyword per source)...")
    collected_papers: List[Dict] = collect_papers(
        keywords_by_topic=critiqued_keywords_nested,
        cutoff_year=cutoff_year,
        paper_per_keyword=papers_to_fetch_per_keyword_source
        # min_per_bucket and max_fetch_attempts will use defaults from collect_papers
    )
    
    # --- 4. Initial Save, Load & Filter of Collected Papers ---
    # This sequence allows for checkpointing and ensures a consistent dataset for subsequent filtering.
    
    print("\nGenerating and refining domain terms...")
    # Generate more terms initially than might be needed, to allow for effective critiquing.
    raw_domain_terms: List[str] = generate_domain_terms(title, max_terms=15)
    critiqued_domain_terms, domain_term_suggestions = critique_list("Domain terms for " + title, raw_domain_terms)

    raw_candidates_path = Path("raw_candidates.json")
    if raw_candidates_path.exists():
        raw_candidates_path.unlink() # Remove the previous run's file
        print(f"🗑️  Deleted old {raw_candidates_path.name}")

    # Save collected papers, query title, and the initially critiqued domain terms
    payload_to_save: Dict[str, any] = {
        "query_title": title,
        "domain_terms": critiqued_domain_terms, # Using the critiqued version for saving
        "papers": collected_papers
    }
    with raw_candidates_path.open("w", encoding="utf-8") as f:
        json.dump(payload_to_save, f, ensure_ascii=False, indent=2)
    print(f"✅ Collected and saved {len(collected_papers)} initial papers (≥ {cutoff_year}) to {raw_candidates_path.name}.")

    # Load data from the saved JSON file for further processing
    # This also reloads domain_terms, which were critiqued before saving.
    loaded_query_title, loaded_papers, loaded_domain_terms_from_file = load_candidates_from_json(str(raw_candidates_path))
    
    print("Applying initial filters (DOI validity, Abstract presence, Duplicates)...")
    papers_after_initial_filters: List[Dict] = filter_by_doi(loaded_papers)
    papers_after_initial_filters = filter_by_abstract(papers_after_initial_filters)
    papers_after_initial_filters = dedupe_by_doi(papers_after_initial_filters)
    print(f"🚀 {len(papers_after_initial_filters)} papers remain after initial DOI, abstract & dedup filters.")

    # --- 5. Generate and Refine Application and Technique Terms ---
    print("\nGenerating and refining application and technique terms...")
    # Generate slightly more raw terms than needed to provide a good base for critique and cleaning
    raw_app_terms: List[str] = generate_app_terms(title, max_terms=7)
    raw_tech_terms: List[str] = generate_tech_terms(title, max_terms=10)
    
    critiqued_app_terms, app_term_suggestions = critique_list("Application terms for " + title, raw_app_terms)
    critiqued_tech_terms, tech_term_suggestions = critique_list("Technique terms for " + title, raw_tech_terms)

    # Display suggestions from the critic AI for all term categories
    print(f"📝 Application term suggestions: {app_term_suggestions}")
    print(f"📝 Technique term suggestions: {tech_term_suggestions}")
    print(f"📝 Domain term suggestions: {domain_term_suggestions}") # Generated and critiqued earlier

    # Finalize application terms: clean, ensure multi-word, include title
    final_app_terms: List[str] = clean_terms(critiqued_app_terms)
    final_app_terms = [t for t in final_app_terms if len(t.split()) > 1] # Keep only phrases
    title_lower_stripped = title.lower().strip()
    if title_lower_stripped not in final_app_terms: # Ensure the main title/topic is an application term
        final_app_terms.insert(0, title_lower_stripped)
    if not final_app_terms: # Fallback if cleaning removed all terms
        print("⚠️ Application terms list became empty after cleaning; using top raw generated app terms as fallback.")
        final_app_terms = raw_app_terms[:5]

    # Finalize technique terms: clean, with fallback
    final_tech_terms: List[str] = clean_terms(critiqued_tech_terms)
    if not final_tech_terms and critiqued_tech_terms: # Fallback to critiqued if cleaning emptied the list
        print("⚠️ Technique terms list became empty after cleaning; using top 3 critiqued terms as fallback.")
        final_tech_terms = critiqued_tech_terms[:3]
    elif not final_tech_terms: # Further fallback to raw if even critiqued was empty
        print("⚠️ Technique terms list completely empty; using raw generated tech terms as fallback.")
        final_tech_terms = raw_tech_terms[:5]

    # Finalize domain terms: use critiqued terms loaded from file, then clean
    final_domain_terms: List[str] = clean_terms(loaded_domain_terms_from_file)
    if not final_domain_terms and loaded_domain_terms_from_file: # Fallback if cleaning emptied the list
        print("⚠️ Domain terms list became empty after cleaning; using originally critiqued domain terms as fallback.")
        final_domain_terms = loaded_domain_terms_from_file[:10]
    elif not final_domain_terms: # Further fallback to raw if even loaded/critiqued was empty
        print("⚠️ Domain terms list completely empty; using raw generated domain terms as fallback.")
        final_domain_terms = raw_domain_terms[:10]

    # Prepare term lists for regex pattern matching (can include suggestions for broader matching)
    app_terms_for_patterns: List[str] = final_app_terms + list(app_term_suggestions.values())
    tech_terms_for_patterns: List[str] = final_tech_terms + list(tech_term_suggestions.values())
    domain_terms_for_patterns: List[str] = final_domain_terms + list(domain_term_suggestions.values())
    
    print(f"🔑 Final Core Application Terms (for criteria matching): {final_app_terms}")
    print(f"🔑 Final Core Technique Terms (for criteria matching): {final_tech_terms}")
    print(f"🔑 Final Core Domain Terms (for criteria matching): {final_domain_terms}")

    # --- 6. Technique Term Clustering (for refinement if many terms) ---
    print("\nProcessing technique terms for potential clustering...")
    # Use the core final_tech_terms for clustering
    tech_term_vectors: List[np.ndarray] = [embed_text(term, use_cache=True) for term in final_tech_terms if term]

    representative_tech_terms: List[str] = final_tech_terms # Default to all terms
    if tech_term_vectors and len(final_tech_terms) > 4: # Perform clustering if enough terms and vectors
        print("Clustering technique terms to find representative set...")
        try:
            tech_vectors_stacked: np.ndarray = np.vstack(tech_term_vectors)
            max_clusters: int = 5
            # Determine a reasonable number of clusters
            num_clusters: int = min(max_clusters, max(2, len(final_tech_terms) // 2))
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
            cluster_labels: np.ndarray = kmeans.fit_predict(tech_vectors_stacked)

            clustered_tech_terms_map: Dict[int, List[str]] = {}
            for term, label in zip(final_tech_terms, cluster_labels):
                clustered_tech_terms_map.setdefault(int(label), []).append(term)

            print("🔑 Technique Clusters Formed:")
            for label_id, terms_in_cluster in clustered_tech_terms_map.items():
                print(f"    Cluster {label_id}: {terms_in_cluster}")

            # Select representative terms (e.g., closest to centroid from each cluster)
            selected_representative_terms_list: List[str] = []
            for label_id, terms_in_cluster in clustered_tech_terms_map.items():
                num_to_keep_from_cluster: int = max(1, math.ceil(math.sqrt(len(terms_in_cluster))))
                if len(terms_in_cluster) <= num_to_keep_from_cluster:
                    selected_representative_terms_list.extend(terms_in_cluster)
                else:
                    term_indices_in_cluster: List[int] = [final_tech_terms.index(t) for t in terms_in_cluster]
                    centroid_for_cluster: np.ndarray = kmeans.cluster_centers_[label_id]
                    
                    distances_to_centroid: List[tuple[str, float]] = []
                    for term, original_idx in zip(terms_in_cluster, term_indices_in_cluster):
                        if original_idx < len(tech_term_vectors): # Ensure index is valid
                             distances_to_centroid.append(
                                 (term, float(np.linalg.norm(tech_term_vectors[original_idx] - centroid_for_cluster)))
                             )
                        else:
                            print(f"Warning: Index mismatch during tech term clustering for term '{term}'. Skipping distance calculation for this term.")
                    
                    distances_to_centroid.sort(key=lambda x: x[1]) # Sort by distance
                    selected_representative_terms_list.extend([t_dist[0] for t_dist in distances_to_centroid[:num_to_keep_from_cluster]])
            
            # Update representative_tech_terms, ensuring uniqueness
            representative_tech_terms = sorted(list(set(selected_representative_terms_list))) 
            print(f"🔑 Representative Technique Terms after clustering: {representative_tech_terms}")
        except Exception as e:
            print(f"Error during technique term clustering: {e}. Using unclustered technique terms.")
            representative_tech_terms = final_tech_terms # Fallback to unclustered
    else:
        print("🔑 Technique clustering skipped (reason: ≤ 4 terms or no valid vectors).")
    
    # Use the most relevant set of technique terms for pattern matching
    # If clustering produced representative terms, those are likely more focused.
    final_tech_terms_for_patterns = representative_tech_terms if representative_tech_terms else tech_terms_for_patterns

    # --- 7. Semantic Ranking, Boosting, and Categorization of Papers ---
    print("\nRanking all collected papers semantically using SciBERT...")
    # Use the initially filtered papers for ranking
    semantically_ranked_papers: List[Dict] = semantic_rank_papers(
        query=loaded_query_title,
        papers=papers_after_initial_filters,
        top_n=len(papers_after_initial_filters) # Rank all available papers
    )
    
    # Compile regex patterns for matching Application, Technique, and Domain terms in paper text
    # Ensure terms are non-empty before creating patterns
    app_regex_patterns: List[re.Pattern] = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in app_terms_for_patterns if term]
    tech_regex_patterns: List[re.Pattern] = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in final_tech_terms_for_patterns if term]
    domain_regex_patterns: List[re.Pattern] = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in domain_terms_for_patterns if term]

    # Apply score boosts based on term matching
    print("Applying Core (Application) & Technique term boosts to paper scores...")
    for paper in semantically_ranked_papers:
        text_to_search = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
        matches_an_app_term = any(pattern.search(text_to_search) for pattern in app_regex_patterns)
        matches_a_tech_term = any(pattern.search(text_to_search) for pattern in tech_regex_patterns)

        if matches_an_app_term and matches_a_tech_term:
            paper["score"] = paper.get("score", 0.0) * 1.25  # Boost if both app and tech terms are present
        elif matches_an_app_term or matches_a_tech_term:
            paper["score"] = paper.get("score", 0.0) * 1.10  # Smaller boost if only one type (app or tech) is present
    
    # Re-sort papers after boosting scores
    semantically_ranked_papers.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    print("✅ Papers re-sorted after score boosting.")

    # Helper functions (closures) for categorizing papers based on term matching
    def check_matches_app(p_dict: Dict) -> bool:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return any(pat.search(txt) for pat in app_regex_patterns)

    def check_matches_tech(p_dict: Dict) -> bool:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return any(pat.search(txt) for pat in tech_regex_patterns)

    def count_domain_hits(p_dict: Dict) -> int:
        txt = (p_dict.get("title", "") + " " + p_dict.get("abstract", "")).lower()
        return sum(bool(dp.search(txt)) for dp in domain_regex_patterns)

    print("\nCategorizing top papers into 'Focused' and 'Exploratory' sets...")
    DESIRED_FOCUSED_COUNT: int = 20
    DESIRED_EXPLORATORY_COUNT: int = 10

    # --- Categorize Focused Papers ---
    focused_papers: List[Dict] = []
    # Tier 1: Must hit at least one app, one tech, AND one domain term
    for p in semantically_ranked_papers:
        if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
        if check_matches_app(p) and check_matches_tech(p) and count_domain_hits(p) >= 1:
            focused_papers.append(p)
    
    # Tier 2 (fallback 1): App AND Tech only, if still short of desired count
    if len(focused_papers) < DESIRED_FOCUSED_COUNT:
        for p in semantically_ranked_papers:
            if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
            if p in focused_papers: continue # Skip already added
            if check_matches_app(p) and check_matches_tech(p):
                focused_papers.append(p)
                
    # Tier 3 (fallback 2): Domain AND (App OR Tech), if still short
    if len(focused_papers) < DESIRED_FOCUSED_COUNT:
        for p in semantically_ranked_papers:
            if len(focused_papers) >= DESIRED_FOCUSED_COUNT: break
            if p in focused_papers: continue # Skip already added
            if count_domain_hits(p) >= 1 and (check_matches_app(p) or check_matches_tech(p)):
                focused_papers.append(p)
    
    final_focused_papers: List[Dict] = focused_papers[:DESIRED_FOCUSED_COUNT]
    focused_paper_dois: set[str] = {p["doi"] for p in final_focused_papers if p.get("doi")}

    # --- Categorize Exploratory Papers ---
    # Select from papers not already in the focused list
    exploratory_candidates: List[Dict] = [p for p in semantically_ranked_papers if p.get("doi") not in focused_paper_dois]
    exploratory_papers: List[Dict] = []

    # Tier 1: Require either ≥2 domain hits OR (≥1 domain AND ≥1 tech)
    for p in exploratory_candidates:
        if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT: break
        if count_domain_hits(p) >= 2 or (count_domain_hits(p) >= 1 and check_matches_tech(p)):
            exploratory_papers.append(p)
            
    # Tier 2 (fallback): Require ≥1 domain hit, if still short
    if len(exploratory_papers) < DESIRED_EXPLORATORY_COUNT:
        for p in exploratory_candidates:
            if len(exploratory_papers) >= DESIRED_EXPLORATORY_COUNT: break
            if p in exploratory_papers: continue # Skip already added
            if count_domain_hits(p) >= 1:
                exploratory_papers.append(p)
                
    final_exploratory_papers: List[Dict] = exploratory_papers[:DESIRED_EXPLORATORY_COUNT]

    # --- 8. Print Final Results ---
    print(f"\n🏆 Focused Top {len(final_focused_papers)} (Application, Technique & Domain relevance):")
    for i, p in enumerate(final_focused_papers, 1):
        print(f"  {i}. {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) — Score: {p.get('score', 0.0):.4f} — DOI: {p.get('doi', 'N/A')}")

    print(f"\n🔍 Exploratory Top {len(final_exploratory_papers)} (Broader Domain relevance):")
    for i, p in enumerate(final_exploratory_papers, 1):
        print(f"  {i}. {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) — Score: {p.get('score', 0.0):.4f} — DOI: {p.get('doi', 'N/A')}")

    # Log total elapsed time for the entire pipeline
    pipeline_elapsed_time = time.time() - pipeline_start_time
    print(f"\n⏱️  Total pipeline execution time: {pipeline_elapsed_time:.2f} seconds ({pipeline_elapsed_time/60:.2f} minutes).\n")

    return {
        "focused": final_focused_papers,
        "exploratory": final_exploratory_papers,
    }


def main():
    """CLI entry point for the pipeline."""
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
    else:
        title = input("Enter research title: ")

    cutoff_year = prompt_cutoff_year()
    citation_style = input("Enter citation style (e.g. APA): ").strip()
    print(f"Processing title: '{title}', for papers published from {cutoff_year} onwards.")

    run_pipeline(title, cutoff_year, citation_style)


if __name__ == "__main__":
    main()
