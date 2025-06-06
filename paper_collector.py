# Standard library imports
import os
import time
import re
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Set, Optional # Added Optional for type hints

# Third-party imports
import requests # For making HTTP requests
import feedparser # For parsing arXiv RSS feeds
from google.api_core import exceptions as google_exceptions # For Google API specific exceptions

# Local application/module imports
from api_client_manager import get_next_api_client # For Gemini API client management
from filter_and_rank import dedupe_by_doi, filter_by_abstract, filter_by_doi # For paper filtering

# --- Global Constants and Configuration ---

# Predefined set of methodology-related stop terms used for filtering generated domain terms.
METHODOLOGY_STOP_TERMS: Set[str] = {
    "survey", "review", "framework", "architecture", "architectures",
    "method", "methods", "approach", "approaches",
    "system", "systems", "analysis", "analyses",
    "algorithm", "algorithms", "technique", "techniques"
}

# Default number of papers to fetch per keyword from each literature source,
# if not overridden by parameters in calling functions.
DEFAULT_PAPER_LIMIT_PER_SOURCE: int = 4

# Default number of fetch attempts (e.g., pages or offsets) for each keyword.
DEFAULT_MAX_FETCH_ATTEMPTS: int = 1 # In the provided file, this was 1. Adjusted in collect_papers to 3 default.

# --- Enrichment Helper Functions ---

def fetch_pubmed_abstract(pmid: str) -> str:
    """
    Retrieves the abstract text for a given PubMed ID (PMID) using the NCBI EFetch utility.

    Args:
        pmid: The PubMed ID as a string.

    Returns:
        A string containing the abstract text, or an empty string if not found or an error occurs.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    try:
        resp = requests.get(url, params=params, timeout=15) # Added timeout
        resp.raise_for_status() # Check for HTTP errors
        root = ET.fromstring(resp.text)
        # Concatenate all AbstractText elements found.
        texts = [ab.text.strip() for ab in root.findall(".//AbstractText") if ab.text]
        return " ".join(texts)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: PubMed EFetch request failed for PMID {pmid}: {e}")
        return ""
    except ET.ParseError as e:
        print(f"ERROR: PubMed EFetch XML parsing failed for PMID {pmid}: {e}")
        return ""

def enrich_crossref(doi: str) -> Dict[str, Any]:
    """
    Fetches full metadata (specifically abstract and authors) for a given DOI
    from Crossref's work API endpoint.

    Args:
        doi: The DOI string (can be with or without "https://doi.org/" prefix).

    Returns:
        A dictionary containing 'abstract' and 'authors', or an empty dictionary on error.
    """
    if not doi:
        return {}
    cleaned_doi = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.crossref.org/works/{cleaned_doi}"
    headers = {"Accept": "application/json", "User-Agent": "AuthematicLiteratureCuration/1.0 (mailto:your_email@example.com)"} # Added User-Agent
    
    try:
        resp = requests.get(url, headers=headers, timeout=15) # Added timeout
        resp.raise_for_status()
        message = resp.json().get("message", {})
        
        authors_list = []
        for author_info in message.get("author", []):
            if isinstance(author_info, dict):
                author_name = " ".join(filter(None, [author_info.get("given"), author_info.get("family")]))
                if author_name:
                    authors_list.append(author_name)
        
        return {
            "abstract": message.get("abstract", ""), # Abstract might be further nested or require processing
            "authors": authors_list
        }
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Crossref enrichment failed for DOI {cleaned_doi}: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Crossref JSON decode failed for DOI {cleaned_doi}: {e}")
        return {}

def fetch_s2_details_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetches paper details (abstract, authors, title, year) from Semantic Scholar
    using a specific DOI via the /paper/{paper_id} endpoint.

    Args:
        doi: The DOI string (e.g., "10.1038/s41586-021-03491-6").

    Returns:
        A dictionary with paper details, or None if an error occurs or paper not found.
    """
    if not doi:
        return None

    doi_cleaned = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_cleaned}"
    params = {"fields": "abstract,authors,title,year,externalIds"}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        paper_data = response.json()
        
        if paper_data:
            authors_list = [
                author_info["name"] for author_info in paper_data.get("authors", [])
                if isinstance(author_info, dict) and author_info.get("name")
            ]
            external_ids = paper_data.get("externalIds", {})
            s2_doi_value = external_ids.get("DOI") if external_ids else None
            
            return {
                "title": paper_data.get("title"),
                "authors": authors_list,
                "abstract": paper_data.get("abstract"),
                "year": paper_data.get("year"),
                "doi_from_s2": s2_doi_value
            }
        return None
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            print(f"S2 Enrich INFO: Paper with DOI {doi_cleaned} not found on Semantic Scholar (404).")
        else:
            print(f"S2 Enrich HTTP error for DOI {doi_cleaned}: {http_err}")
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"S2 Enrich Request error for DOI {doi_cleaned}: {req_e}")
        return None
    except json.JSONDecodeError as json_e:
        print(f"S2 Enrich JSON decode error for DOI {doi_cleaned}: {json_e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"S2 Enrich Unexpected error for DOI {doi_cleaned}: {type(e).__name__} - {e}")
        return None

def enrich_with_semantic_scholar(papers: List[Dict[str, Any]]):
    """
    Enriches a list of paper dictionaries with missing abstracts or authors
    by fetching details from Semantic Scholar using their DOIs.
    A delay is included between API calls to respect API usage policies.
    """
    print(f"Starting Semantic Scholar enrichment for up to {len(papers)} papers.")
    enriched_count = 0
    for i, p_dict in enumerate(papers):
        abstract_present = isinstance(p_dict.get("abstract"), str) and bool(p_dict.get("abstract", "").strip())
        authors_present = bool(p_dict.get("authors"))

        if (not abstract_present or not authors_present) and p_dict.get("doi"):
            # print(f"  Attempting to enrich paper {i+1}/{len(papers)} (DOI: {p_dict['doi']})") # Verbose
            s2_details = fetch_s2_details_by_doi(p_dict["doi"])
            
            if s2_details:
                updated = False
                if not abstract_present and s2_details.get("abstract"):
                    p_dict["abstract"] = s2_details["abstract"]
                    updated = True
                if not authors_present and s2_details.get("authors"):
                    p_dict["authors"] = s2_details["authors"]
                    updated = True
                if updated:
                    enriched_count +=1
            
            time.sleep(1.5) # Crucial delay to avoid hammering the Semantic Scholar API
        
        if (i + 1) % 50 == 0 and i > 0: # Progress update
            print(f"  Processed {i + 1}/{len(papers)} papers for S2 enrichment...")
            
    print(f"Semantic Scholar enrichment phase complete. {enriched_count} papers had details fetched/updated.")

# --- Keyword and Term Generation Functions (Using Gemini API) ---

def generate_topics(title: str) -> List[str]:
    """
    Generates 4 distinct academic topic labels for a given research title using Gemini.
    Includes Core, Adjacent, and Emerging topic categories.
    """
    prompt = f"""You are an expert academic research assistant.
Given a research paper title, generate EXACTLY 4 distinct academic topic labels that best frame the “Related Work” section. You must include:
  • 2 Core topics—directly at the heart of the title’s domain  
  • 1 Adjacent topic—close sibling areas that often cross-pollinate  
  • 1 Emerging topic—a nascent or hot area on the horizon  
For each topic, output as:
<Label> (<Category>): <a specific 6–10 word phrase description> 
Do **not** include any explanation, preamble, or commentary.  
Respond **with just** the four topics, either:
  - **One per line**, OR** - **A single comma-separated list** Avoid bullets, numbering, or extra text.  
Research Title: {title}"""
    
    active_client = get_next_api_client()
    response = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = response.text.strip()
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and "," in lines[0]: # If one line and contains commas, assume comma-separated
        lines = [t.strip() for t in lines[0].split(",") if t.strip()]
    
    if len(lines) != 4: # Validate that exactly 4 topics were generated
        print(f"WARNING: generate_topics expected 4 topics, got {len(lines)}. Response: '{text}'")
        # Fallback or error handling strategy can be refined here if needed.
        # For now, return as is or raise error as before.
        # raise RuntimeError(f"Expected 4 topics, got {len(lines)}: {lines}")
    return lines

def generate_subthemes(related_topics: List[str], max_subthemes: int = 3) -> Dict[str, List[str]]:
    """
    For each high-level topic, generates a list of sub-themes (research niches) using Gemini.
    """
    subthemes_by_topic: Dict[str, List[str]] = {}
    for topic in related_topics:
        prompt = f"""You are an expert academic research assistant.
Given the academic topic "{topic}", list exactly {max_subthemes} key sub-themes
(research niches or angles) that often appear under this topic.
Respond with a single comma-separated list, no bullets or commentary."""

        active_client = get_next_api_client()
        resp = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = resp.text.strip()
        
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) == 1 and "," in lines[0]: # Check for comma separation if it's a single line
            subs = [s.strip() for s in lines[0].split(",") if s.strip()]
        else: # Assume multi-line if not a single comma-separated line
            subs = lines
        subthemes_by_topic[topic] = subs[:max_subthemes] # Take up to max_subthemes
    return subthemes_by_topic

def _best_match_for_json_keys(key_to_match: str, candidate_keys: List[str]) -> Optional[str]:
    """
    Helper to find the best matching candidate key for a given key,
    used for parsing Gemini's JSON responses.
    Priority: exact match > substring > Jaccard index >= 0.4.
    """
    key_lower = key_to_match.lower()
    key_tokens = set(key_lower.split())

    # 1) Exact case-insensitive match
    for candidate in candidate_keys:
        if key_lower == candidate.lower():
            return candidate
    # 2) Substring match (case-insensitive)
    for candidate in candidate_keys:
        candidate_lower = candidate.lower()
        if key_lower in candidate_lower or candidate_lower in key_lower:
            return candidate
    # 3) Token overlap (Jaccard similarity)
    scores = []
    for candidate in candidate_keys:
        candidate_tokens = set(candidate.lower().split())
        intersection_len = len(key_tokens.intersection(candidate_tokens))
        union_len = len(key_tokens.union(candidate_tokens))
        jaccard_score = intersection_len / max(1, union_len) # max(1,...) to avoid division by zero for empty inputs
        scores.append((jaccard_score, candidate))
    
    if scores:
        scores.sort(key=lambda item: item[0], reverse=True) # Sort by Jaccard score
        if scores[0][0] >= 0.4: # Threshold for Jaccard similarity
            return scores[0][1]
    return None

def generate_keywords_by_subtheme(
        subthemes_by_topic: Dict[str, List[str]],
        max_terms: int = 5,
        output_path_str: str = "keywords_by_subtheme.json" # Changed name to avoid conflict with pathlib.Path
) -> Dict[str, Dict[str, List[str]]]:
    """
    For every Topic & Sub-theme pair, generates search keywords using Gemini.
    It robustly matches Gemini's JSON output to the expected structure and
    saves the result to a JSON file.
    """
    output_file = Path(output_path_str)
    if output_file.exists(): # Use pathlib for path operations
        output_file.unlink()

    generated_keywords_data: Dict[str, Dict[str, List[str]]] = {}
    for raw_topic_input, subtheme_list_for_topic in subthemes_by_topic.items():
        prompt_lines = [
            "You are an expert academic research assistant.",
            f"For each sub-theme of the research topic below, give up to {max_terms} "
            "high-specificity search keywords **that combine the sub-theme’s focus "
            "with the parent topic’s context**. Each keyword should be 2-5 words, "
            "precise, and something likely to appear in title or abstract.",
            "The key words should differ semantically from its siblings (avoid near-synonyms), and "
            "be suitable for searching scholarly titles/abstracts.",
            "Respond with **only** a JSON object in the exact format:",
            '{"<Name of the Main Topic (exactly as provided to you)>": {"<Sub-theme 1 (exactly as provided)>": ["kw1","kw2",...], "<Sub-theme 2 (exactly as provided)>": [...]}}',
            "",
            "Topic and its Sub-themes:",
            f"- {raw_topic_input}:"
        ]
        for subtheme_item in subtheme_list_for_topic:
            prompt_lines.append(f"  • {subtheme_item}")
        prompt = "\n".join(prompt_lines)

        try:
            active_client = get_next_api_client()
            resp = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        except google_exceptions.GoogleAPIError as e:
            raise RuntimeError(f"Gemini API error during keyword generation for topic “{raw_topic_input}”: {e}") from e

        raw_response_text = resp.text.strip()
        # print(f"\n[DEBUG] Gemini raw response for keywords (topic: “{raw_topic_input}”):\n{raw_response_text}\n") # Keep for debugging if needed

        try:
            # Attempt to parse JSON, tolerating common issues like markdown code fences
            parsed_json_data = json.loads(raw_response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_response_text, re.DOTALL) # More robust regex for extracting JSON
            if match:
                json_str = match.group(0)
                try:
                    parsed_json_data = json.loads(json_str)
                except json.JSONDecodeError as e_inner:
                    raise RuntimeError(f"Could not parse extracted JSON for topic “{raw_topic_input}”. Extracted: '{json_str}'. Error: {e_inner}") from e_inner
            else:
                raise RuntimeError(f"No JSON object found in response for topic “{raw_topic_input}”. Response: '{raw_response_text}'")
        
        # Match the top-level topic key from Gemini's response
        json_top_level_keys = list(parsed_json_data.keys())
        actual_topic_key_in_json = None
        if len(json_top_level_keys) == 1:
            actual_topic_key_in_json = json_top_level_keys[0]
            # print(f"INFO: Assuming single JSON key '{actual_topic_key_in_json}' corresponds to input topic '{raw_topic_input}'.")
        elif len(json_top_level_keys) > 1:
            # Try to match based on the part of raw_topic_input before " ("
            key_to_match_best = raw_topic_input.split(" (")[0]
            actual_topic_key_in_json = _best_match_for_json_keys(key_to_match_best, json_top_level_keys)
            if actual_topic_key_in_json:
                 print(f"WARNING: Found multiple top-level keys in JSON for '{raw_topic_input}'. Matched to '{actual_topic_key_in_json}'.")
        
        if not actual_topic_key_in_json or not isinstance(parsed_json_data.get(actual_topic_key_in_json), dict):
            # Detailed error for easier debugging
            error_msg = (f"Could not reliably match or validate topic “{raw_topic_input}” in Gemini JSON response. "
                         f"Input topic part for matching: '{raw_topic_input.split(' (')[0]}'. "
                         f"JSON keys received: {json_top_level_keys}. "
                         f"Determined key (if any): '{actual_topic_key_in_json}'. "
                         f"Is value a dict: {isinstance(parsed_json_data.get(actual_topic_key_in_json), dict)}.")
            print(f"ERROR_DETAILS: {error_msg}")
            raise RuntimeError(error_msg)

        topic_content_dict = parsed_json_data[actual_topic_key_in_json]
        # Use the original raw_topic_input as the key for consistent structuring of generated_keywords_data
        generated_keywords_data[raw_topic_input] = {}

        # Match each expected sub-theme against the keys in Gemini's topic_content_dict
        for expected_subtheme in subtheme_list_for_topic:
            actual_subtheme_key_in_json = _best_match_for_json_keys(expected_subtheme, list(topic_content_dict.keys()))
            
            if not actual_subtheme_key_in_json:
                print(f"⚠️ No JSON entry found for sub-theme “{expected_subtheme}” under matched topic key “{actual_topic_key_in_json}” (original raw_topic: “{raw_topic_input}”); assigning empty list.")
                generated_keywords_data[raw_topic_input][expected_subtheme] = []
                continue

            keyword_list_from_json = topic_content_dict[actual_subtheme_key_in_json]
            if not isinstance(keyword_list_from_json, list):
                raise RuntimeError(f"Keywords for “{actual_subtheme_key_in_json}” (derived from sub-theme “{expected_subtheme}”) must be a list, got {type(keyword_list_from_json)}.")
            
            # Store keywords, ensuring they are strings and limiting to max_terms
            generated_keywords_data[raw_topic_input][expected_subtheme] = [str(k) for k in keyword_list_from_json[:max_terms]]

    # Persist the generated keywords to the specified output file
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(generated_keywords_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Keywords by subtheme saved to {output_file.name}")
    return generated_keywords_data

# --- Literature Source Search Functions ---

def search_arxiv(keyword: str, start_index: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Searches arXiv for papers matching the keyword.
    """
    search_limit = limit if limit is not None else DEFAULT_PAPER_LIMIT_PER_SOURCE
    papers_found: List[Dict[str, Any]] = []
    try:
        query_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{keyword.replace(' ', '%20')}" # URL encode keyword
            f"&start={start_index}&max_results={search_limit}"
        )
        # feedparser handles its own HTTP requests; direct timeout not straightforward
        parsed_feed = feedparser.parse(query_url) 
        
        for entry in parsed_feed.entries:
            arxiv_doi = getattr(entry, 'arxiv_doi', None) # Preferred DOI if available
            final_doi_str = ""
            if arxiv_doi:
                final_doi_str = f"https://doi.org/{arxiv_doi}"
            else: # Fallback to arXiv ID if no specific DOI field
                # entry.id is typically "http://arxiv.org/abs/YYMM.NNNNNvV"
                arxiv_id = entry.id.rsplit('/', 1)[-1]
                final_doi_str = f"arXiv:{arxiv_id}" # Using arXiv ID as a unique identifier

            papers_found.append({
                "title": entry.title,
                "authors": [author.name for author in entry.authors if hasattr(author, 'name')],
                "abstract": entry.summary,
                "doi": final_doi_str,
                "year": int(entry.published[:4]) if hasattr(entry, 'published') and entry.published else None,
                "source": "arXiv"
            })
        return papers_found
    except Exception as e:
        print(f"ERROR: arXiv search failed for keyword '{keyword}': {e}")
        return []

# search_semantic_scholar function for keyword search is intentionally removed/commented out
# due to persistent rate limit issues. Enrichment uses fetch_s2_details_by_doi instead.
#
# def search_semantic_scholar(keyword: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
#    """ (This function is currently not used for main keyword search) """
#    # ... implementation was here ...
#    pass


def search_crossref(keyword: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Searches CrossRef for papers matching the keyword.
    Includes a User-Agent for polite API interaction.
    """
    search_limit = limit if limit is not None else DEFAULT_PAPER_LIMIT_PER_SOURCE
    papers_found: List[Dict[str, Any]] = []
    try:
        base_url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": keyword, # Using query.bibliographic for better relevance
            "rows": search_limit,
            "offset": offset,
            "sort": "relevance",
            "order": "desc"
        }
        headers = { # Polite API usage: identify your application
            "User-Agent": "AuthematicLiteratureCuration/1.0 (mailto:your_project_email@example.com)"
        }
        
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        items = response.json().get("message", {}).get("items", [])
        for item in items:
            published_date_info = item.get("published-print") or item.get("published-online") or item.get("created")
            year_val = None
            if published_date_info and "date-parts" in published_date_info and published_date_info["date-parts"]:
                year_val = published_date_info["date-parts"][0][0] # First part of the date is usually the year

            authors_list = []
            for author_info in item.get("author", []):
                if isinstance(author_info, dict): # Ensure author_info is a dict
                    author_name_parts = filter(None, [author_info.get("given"), author_info.get("family")])
                    authors_list.append(" ".join(author_name_parts))

            doi_val = item.get("DOI", "")
            full_doi_url = f"https://doi.org/{doi_val}" if doi_val else ""
            
            # Abstract might be under 'abstract' or sometimes in 'subtitle' or other fields
            # Crossref often returns stripped-down abstracts or none at all in search results.
            abstract_text = item.get("abstract", "")
            if isinstance(abstract_text, str) and abstract_text.startswith("<jats:p>"): # Basic JATS abstract cleaning
                 abstract_text = re.sub('<[^<]+?>', '', abstract_text).strip()

            paper_title_list = item.get("title", [])
            paper_title = paper_title_list[0] if paper_title_list and isinstance(paper_title_list, list) else ""

            papers_found.append({
                "title": paper_title,
                "authors": authors_list,
                "abstract": abstract_text,
                "doi": full_doi_url,
                "year": int(year_val) if year_val else None,
                "source": "CrossRef"
            })
        return papers_found
    except requests.exceptions.RequestException as e:
        print(f"ERROR: CrossRef search failed for keyword '{keyword}': {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR: CrossRef JSON decode failed for keyword '{keyword}': {e}")
        return []
    except Exception as e: # Catch any other unexpected errors
        print(f"ERROR: Unexpected error in CrossRef search for keyword '{keyword}': {type(e).__name__} - {e}")
        return []

def search_pubmed(keyword: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Searches PubMed for papers matching the keyword.
    First searches for PMIDs, then fetches summaries for those PMIDs.
    """
    search_limit = limit if limit is not None else DEFAULT_PAPER_LIMIT_PER_SOURCE
    papers_found: List[Dict[str, Any]] = []
    
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    pmids_list: List[str] = []
    try:
        search_params = {
            "db": "pubmed", "term": keyword, "retmode": "json",
            "retstart": offset, "retmax": search_limit, "sort": "relevance"
        }
        search_resp = requests.get(esearch_url, params=search_params, timeout=20)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        pmids_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids_list:
            return []
        
        # Fetch summaries for the retrieved PMIDs
        summary_params = {"db": "pubmed", "id": ",".join(pmids_list), "retmode": "json"}
        summary_resp = requests.get(esummary_url, params=summary_params, timeout=20)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()
        
        for pmid_str in pmids_list:
            article_summary = summary_data.get("result", {}).get(pmid_str, {})
            if not article_summary:
                continue

            pub_date_str = article_summary.get("pubdate", "")
            year_val = None
            if pub_date_str:
                year_match = re.search(r'\b(19|20)\d{2}\b', pub_date_str)
                if year_match:
                    year_val = int(year_match.group(0))

            authors_list = [
                author_info["name"] for author_info in article_summary.get("authors", [])
                if isinstance(author_info, dict) and author_info.get("name")
            ]

            # Construct DOI or fallback URL. PubMed summaries might provide elocationid which can be a DOI.
            doi_str_from_pubmed = ""
            elocation_id = article_summary.get("elocationid", "") # Often contains DOI like "10.1234/journal.xxxx" or "doi: 10.1234..."
            if isinstance(elocation_id, str):
                # Try to extract DOI if present
                doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', elocation_id, re.IGNORECASE)
                if doi_match:
                    doi_str_from_pubmed = f"https://doi.org/{doi_match.group(1)}"
            
            if not doi_str_from_pubmed: # Fallback to PubMed URL if no clear DOI
                doi_str_from_pubmed = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"

            # Fetch full abstract separately using efetch for better quality
            abstract_text = fetch_pubmed_abstract(pmid_str)

            papers_found.append({
                "title": article_summary.get("title", "").strip(),
                "authors": authors_list,
                "abstract": abstract_text,
                "doi": doi_str_from_pubmed,
                "year": year_val,
                "source": "PubMed"
            })
        return papers_found
    except requests.exceptions.RequestException as e:
        print(f"ERROR: PubMed request failed for keyword '{keyword}': {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR: PubMed JSON decode failed for keyword '{keyword}': {e}")
        return []
    except Exception as e: # Catch any other unexpected errors
        print(f"ERROR: Unexpected error in PubMed search for keyword '{keyword}': {type(e).__name__} - {e}")
        return []

# --- Main Paper Collection Orchestrator ---

def fetch_from_source_wrapper(
    source_func_dict: Dict[str, Any], 
    keyword: str, 
    current_offset_or_start_index: int, 
    search_limit: int, 
    cutoff_year_val: int # cutoff_year is used by filters later, not directly by search funcs usually
) -> Dict[str, Any]:
    """
    Wrapper to call a specific literature source search function.
    Handles differences in parameters (offset vs. start_index) and provides
    a standardized return structure including error information.
    """
    source_func = source_func_dict["func"]
    source_name = source_func_dict["name"]
    
    try:
        if source_name == "arXiv": # arXiv API uses 'start_index'
            papers = source_func(keyword, start_index=current_offset_or_start_index, limit=search_limit)
        else: # Other APIs use 'offset'
            papers = source_func(keyword, offset=current_offset_or_start_index, limit=search_limit)
        return {"source_name": source_name, "papers": papers, "keyword": keyword, "error": None}
    except Exception as e:
        # Log the error with details
        print(f"      ERROR: During fetch from {source_name} for keyword '{keyword}' (offset/start: {current_offset_or_start_index}): {type(e).__name__} - {e}")
        return {"source_name": source_name, "papers": [], "keyword": keyword, "error": e}

def collect_papers(
    keywords_by_topic: Dict[str, Dict[str, List[str]]],
    cutoff_year: int,
    paper_per_keyword: int, # This is the 'limit' per source, per keyword query
    max_fetch_attempts: int = 3, # Number of pages/offsets to try per keyword
    min_papers_per_bucket: int = 35 # Target number of papers per sub-theme; adjust as needed
) -> List[Dict[str, Any]]:
    """
    Collects papers for each sub-theme by concurrently querying multiple literature sources.
    It iterates through keywords, attempting to fill each sub-theme "bucket" to
    `min_papers_per_bucket`. It handles multiple fetch attempts (offsets/pages)
    and then consolidates, enriches, and globally filters/deduplicates all results.
    """
    total_raw_fetches: int = 0 # Count of all paper entries initially fetched from APIs
    papers_by_bucket: Dict[str, List[Dict[str, Any]]] = {} # { "Topic ▶ Subtheme": [papers] }

    # Initialize an empty list for each sub-theme bucket
    for topic_name, subthemes_map in keywords_by_topic.items():
        for subtheme_name_val in subthemes_map:
            papers_by_bucket[f"{topic_name} ▶ {subtheme_name_val}"] = []

    # Configuration for literature search sources (Semantic Scholar is excluded)
    source_configurations: List[Dict[str, Any]] = [
        {"func": search_arxiv, "name": "arXiv"},
        # {"func": search_semantic_scholar, "name": "SemanticScholar"}, # Excluded
        {"func": search_pubmed, "name": "PubMed"},
        {"func": search_crossref, "name": "CrossRef"}
    ]
    num_concurrent_sources: int = len(source_configurations)

    # Iterate through each bucket (Topic ▶ Sub-theme)
    for bucket_label, papers_collected_for_bucket in papers_by_bucket.items():
        current_topic, current_subtheme = bucket_label.split(" ▶ ", 1)
        print(f"\n🔍 Collecting for Topic: \"{current_topic}\" ▶ Sub-theme: \"{current_subtheme}\"")
        
        keywords_for_current_subtheme: List[str] = keywords_by_topic.get(current_topic, {}).get(current_subtheme, [])
        if not keywords_for_current_subtheme:
            print(f"    ⚠️ No keywords found for sub-theme '{current_subtheme}', skipping.")
            continue

        # Iterate through fetch attempts (pages/offsets)
        for attempt_idx in range(max_fetch_attempts):
            if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                break # Bucket is already full for this sub-theme

            # Calculate current offset or start index for this attempt
            current_offset_val: int = attempt_idx * paper_per_keyword
            pass_description = "Initial Pass" if attempt_idx == 0 else f"Retry Pass {attempt_idx}"
            print(f"    🔷 {pass_description} (Offset/Start Index: {current_offset_val}) for '{current_subtheme}'")

            # Iterate through keywords for the current sub-theme
            for keyword_str in keywords_for_current_subtheme:
                if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                    break # Bucket filled by previous keywords in this attempt

                print(f"      → Querying with keyword: \"{keyword_str}\"")
                
                # Track DOIs added from this specific keyword's concurrent fetch round
                # to avoid adding the exact same paper if multiple sources return it for this specific query.
                dois_added_this_keyword_query_round: Set[str] = set()

                with ThreadPoolExecutor(max_workers=num_concurrent_sources) as executor:
                    # Submit tasks for all sources for the current keyword and offset
                    future_to_source_map = {
                        executor.submit(fetch_from_source_wrapper, src_config, keyword_str, current_offset_val, paper_per_keyword, cutoff_year): src_config["name"]
                        for src_config in source_configurations
                    }

                    # Process results as they complete
                    for future_task in as_completed(future_to_source_map):
                        source_name_from_future = future_to_source_map[future_task]
                        try:
                            result_from_source = future_task.result()
                            papers_from_source_hit = result_from_source.get("papers", [])
                            
                            if result_from_source.get("error"):
                                # Error already printed by the wrapper function
                                continue # Skip processing this source's results if an error occurred
                            
                            total_raw_fetches += len(papers_from_source_hit)
                            # print(f"        • {source_name_from_future}: Got {len(papers_from_source_hit)} raw hits for \"{keyword_str}\"") # Verbose

                            # --- Filter and Deduplicate results from THIS specific source hit ---
                            # 1. Basic validity checks: year, presence of DOI string, presence of abstract string
                            valid_papers_slice = [
                                p for p in papers_from_source_hit 
                                if p and isinstance(p, dict) and
                                   p.get("year", 0) >= cutoff_year and 
                                   isinstance(p.get("doi"), str) and p["doi"].strip() and
                                   isinstance(p.get("abstract"), str) and p["abstract"].strip()
                            ]
                            
                            # 2. Deduplicate within this source's currently fetched valid results
                            unique_valid_papers_slice = dedupe_by_doi(valid_papers_slice)

                            # --- Add to bucket if paper is new and bucket isn't full ---
                            if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                                # Bucket might have been filled by another concurrent task for this same keyword
                                continue 

                            # Get all DOIs already present in the main collection for this bucket
                            dois_in_current_bucket_collection = {p_item["doi"].lower().strip() for p_item in papers_collected_for_bucket if p_item.get("doi")}

                            for paper_candidate in unique_valid_papers_slice:
                                if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                                    break # Bucket is now full, stop adding

                                paper_candidate_doi_norm = paper_candidate["doi"].lower().strip()
                                # Add if DOI is not in the global bucket collection AND not just added in this specific keyword query round
                                if paper_candidate_doi_norm not in dois_in_current_bucket_collection and \
                                   paper_candidate_doi_norm not in dois_added_this_keyword_query_round:
                                    
                                    papers_collected_for_bucket.append(paper_candidate)
                                    dois_added_this_keyword_query_round.add(paper_candidate_doi_norm)
                            
                        except Exception as main_processing_exc: # Catch errors during result processing
                            print(f"      ⚠️ Main processing exception for {source_name_from_future} results (keyword '{keyword_str}'): {type(main_processing_exc).__name__} - {main_processing_exc}")
                
                # After all sources for a keyword have been processed (or attempted) for this offset:
                if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                    print(f"    ✔️ Bucket for '{current_subtheme}' filled to {len(papers_collected_for_bucket)}/{min_papers_per_bucket} papers.")
                    break # Break from the keywords loop for this attempt, as the bucket is full.
                
                # Brief pause after each keyword's batch of API calls to be polite to external APIs
                time.sleep(1) # Adjusted from 1.5 to 1, can be tuned.
            
            # After iterating through all keywords for the current attempt:
            if len(papers_collected_for_bucket) >= min_papers_per_bucket:
                break # Break from the attempts loop if bucket is full.
        
        # After all attempts for a bucket:
        if len(papers_collected_for_bucket) < min_papers_per_bucket:
            print(f"    ⚠️ Bucket for '{current_subtheme}' has {len(papers_collected_for_bucket)}/{min_papers_per_bucket} papers after all attempts.")
        else:
             print(f"    ✅ Bucket for '{current_subtheme}' collection complete with {len(papers_collected_for_bucket)} papers.")

    # --- Final Processing after all buckets are populated ---
    # Flatten the list of papers from all buckets
    all_papers_collected_flat: List[Dict[str, Any]] = []
    for bucket_paper_list in papers_by_bucket.values():
        all_papers_collected_flat.extend(bucket_paper_list)
    
    print(f"\n🔄 Total raw paper entries fetched (sum of all hits from all sources): {total_raw_fetches}")
    print(f"Collected {len(all_papers_collected_flat)} papers across all buckets before final global filtering and enrichment.")

    # Enrich papers with data from Semantic Scholar (for missing abstracts/authors via DOI lookup)
    # This step is sequential and calls an external API; could be slow if many papers need enrichment.
    if any((not (isinstance(p.get("abstract"), str) and bool(p.get("abstract", "").strip())) or not bool(p.get("authors"))) and p.get("doi") for p in all_papers_collected_flat):
        print("Enriching papers with Semantic Scholar data for missing abstracts/authors (if any)...")
        enrich_with_semantic_scholar(all_papers_collected_flat)
    else:
        print("Skipping Semantic Scholar enrichment as all papers appear to have abstracts and authors or lack DOIs for lookup.")
    
    # Apply final global filters (ensure DOI validity, abstract presence)
    print("Applying final global filters (DOI validity, Abstract presence)...")
    globally_filtered_papers: List[Dict[str, Any]] = filter_by_doi(all_papers_collected_flat) # Removes papers without valid DOI
    globally_filtered_papers = filter_by_abstract(globally_filtered_papers) # Removes papers without abstract
    
    # Final global deduplication based on DOI
    print(f"Deduplicating {len(globally_filtered_papers)} papers globally...")
    final_unique_papers_list: List[Dict[str, Any]] = dedupe_by_doi(globally_filtered_papers)

    print(f"✅ Final collection yields: {len(final_unique_papers_list)} unique, valid papers.")
    return final_unique_papers_list

# --- Functions for generating Domain, Application, and Technique terms ---
# These functions use Gemini and should call get_next_api_client()

def generate_domain_terms(title: str, max_terms: int = 10) -> List[str]:
    """
    Generates domain-specific keywords/phrases for a given research title using Gemini.
    The prompt asks for 15 terms, then the function filters and truncates.
    """
    prompt = f"""You are an expert academic research assistant.
Your task: Generate exactly 15 **domain-specific keywords or short phrases** that any relevant paper’s title or abstract should contain at least one of.
• These terms form your filter: they must capture the core subject matter or application area (including key subdomains and jargon).
• Do NOT include names of methods, algorithms, surveys, architectures, or generic frameworks.
• Return exactly 15 comma-separated, lowercase phrases, no numbering, no extra text.
Example for topic “Medical Image Segmentation”:
medical imaging, image segmentation, radiology, anatomical structures, lesion detection, pixel-level classification, semantic segmentation, instance segmentation, multimodal fusion, computer-aided diagnosis, quantitative imaging biomarkers, deep image analysis, segmentation uncertainty, atlas-based methods, volume rendering
Topic: {title}"""
    
    active_client = get_next_api_client()
    response = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = response.text.strip()
    
    raw_terms = [t.strip().lower() for t in text.split(",") if t.strip()] # Ensure terms are stripped and non-empty
    
    # Deduplicate while preserving order
    seen_terms: Set[str] = set()
    deduplicated_terms: List[str] = []
    for term in raw_terms:
        if term and term not in seen_terms: # Check 'term' is not empty after strip
            deduplicated_terms.append(term)
            seen_terms.add(term)
    
    # Filter out terms containing methodology stop words
    filtered_terms: List[str] = []
    for term in deduplicated_terms:
        term_words = set(term.split())
        if term_words.isdisjoint(METHODOLOGY_STOP_TERMS):
            filtered_terms.append(term)
    
    return filtered_terms[:max_terms] # Return up to max_terms

def generate_app_terms(title: str, max_terms: int = 7) -> List[str]:
    """
    Extracts application-centric phrases (domain + object) from the title using Gemini.
    """
    prompt = f"""You are an expert academic research assistant.
Your task: Given a research paper title, extract exactly {max_terms} **application phrases**—that is, the concrete objects, contexts or tasks being studied (the “what” and “where”).
• Only output noun or noun-phrase labels that describe the application domain.
• Do NOT include any method names, algorithm families, evaluation terms, or metrics.
• They must describe the problem space, objects, or domain context (e.g. body part, disease, modality, task)
• They should **NOT** contain methodology or algorithm words.
• Include at least ONE application term present in the title itself.
• Allow single word synonyms if you deem them as domain critical.
• Return exactly {max_terms} phrases, in a single comma-separated list, all lowercase, no numbering or extra words.
Example for title “Graph Neural Network Architectures for Molecular Property Prediction”:
molecular property prediction, molecular graphs, drug discovery, materials informatics, cheminformatics
Research Title: {title}"""
    
    active_client = get_next_api_client()
    resp = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    raw_terms = resp.text.strip().lower().split(",")
    
    # Clean, deduplicate, and limit number of terms
    output_terms: List[str] = []
    seen_terms_set: Set[str] = set()
    for t_str in raw_terms:
        cleaned_term = t_str.strip()
        if cleaned_term and cleaned_term not in seen_terms_set:
            seen_terms_set.add(cleaned_term)
            output_terms.append(cleaned_term)
        if len(output_terms) >= max_terms:
            break
    return output_terms

def generate_tech_terms(title: str, max_terms: int = 10) -> List[str]:
    """
    Extracts technique-centric or methodology phrases relevant to the title using Gemini.
    """
    prompt = f"""You are an expert academic research assistant.
Your task: From a research paper title, extract exactly {max_terms} **technique or methodology phrases**—the “how” of the work.
• If relevant and applicable, only include specific algorithm, architecture, or analysis method names (e.g. “saliency maps”, “grad-cam”, “message passing neural networks”), rather than generic words for these.
• Do NOT include domain words, datasets, or application contexts.
• Include exactly one technical or methodology term present in the title itself.
• Return exactly {max_terms} items as a comma-separated, lowercase list with no extra commentary.
Example for title “Explainable AI Techniques for Medical Image Segmentation”:
saliency maps, grad-cam, shap values, surrogate models, counterfactual explanations
Research Title: {title}"""
    
    active_client = get_next_api_client()
    resp = active_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    raw_terms = resp.text.strip().lower().split(",")

    # Clean, deduplicate, and limit number of terms
    output_terms: List[str] = []
    seen_terms_set: Set[str] = set()
    for t_str in raw_terms:
        cleaned_term = t_str.strip()
        if cleaned_term and cleaned_term not in seen_terms_set:
            seen_terms_set.add(cleaned_term)
            output_terms.append(cleaned_term)
        if len(output_terms) >= max_terms:
            break
    return output_terms
