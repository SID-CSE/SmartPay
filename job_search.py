"""
job_search.py -- Finds real, currently-posted job listings similar to a profile.

This is DIFFERENT from external_lookup.py, which searches for a salary FIGURE when
a company is unknown. This module searches for actual JOB LISTINGS -- real postings
a candidate could look at or apply to -- to complement the salary estimate with
market comparables ("here's what's actually posted right now that looks like this").

Same backend as external_lookup.py (Tavily, free tier, see that file's docstring for
setup) -- reuses the same TAVILY_API_KEY, no separate signup needed.
"""
import os
import re

try:
    from tavily import TavilyClient
    _TAVILY_SDK_AVAILABLE = True
except ImportError:
    _TAVILY_SDK_AVAILABLE = False


def _get_api_key():
    try:
        import streamlit as st
        if "TAVILY_API_KEY" in st.secrets:
            return st.secrets["TAVILY_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TAVILY_API_KEY")


JOB_SITES = ['naukri.com', 'linkedin.com/jobs', 'indeed.co.in', 'glassdoor.co.in']


def _clean_snippet(text, max_len=200):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text[:max_len] + ('...' if len(text) > max_len else '')


def _guess_company_from_result(title, url, snippet):
    """Best-effort company name extraction from a job listing title (e.g. 'Backend Developer at Zoho - Naukri.com')."""
    m = re.search(r'\bat\s+([A-Z][\w&.\- ]{1,40}?)(?:\s*[-|]|\s+in\s+|\s*$)', title or '')
    if m:
        return m.group(1).strip()
    return None


def search_similar_jobs(profile: dict, max_results: int = 5) -> dict:
    """
    Searches for real, currently-posted job listings similar to the given profile.

    Returns:
        {
            "available": bool,
            "jobs": [ {"title": str, "company": str|None, "url": str, "snippet": str}, ... ],
            "message": str,   # always present, human-readable status
        }
    """
    if not _TAVILY_SDK_AVAILABLE:
        return {"available": False, "jobs": [], "message": "tavily-python isn't installed. Run: pip install tavily-python"}

    api_key = _get_api_key()
    if not api_key:
        return {
            "available": False, "jobs": [],
            "message": (
                "Live job search isn't configured. Get a free Tavily API key at https://app.tavily.com "
                "(no card needed, 1,000 free searches/month) and set TAVILY_API_KEY -- see external_lookup.py "
                "for the two ways to provide it. This reuses the same key as the salary web-lookup feature."
            ),
        }

    title = (profile.get('title') or '').strip()
    location = (profile.get('location') or '').strip() or 'India'
    exp_min = profile.get('experience_min', 0)
    exp_max = profile.get('experience_max', exp_min)

    site_filter = ' OR '.join(f'site:{s}' for s in JOB_SITES)
    query = f'"{title}" jobs {location} {exp_min}-{exp_max} years experience ({site_filter})'

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", max_results=max_results)
    except Exception as e:
        return {"available": False, "jobs": [], "message": f"Job search failed ({type(e).__name__})."}

    jobs = []
    for r in response.get('results', []):
        job_title = r.get('title', '') or title
        url = r.get('url', '')
        snippet = _clean_snippet(r.get('content', ''))
        company = _guess_company_from_result(job_title, url, snippet)
        jobs.append({"title": job_title, "company": company, "url": url, "snippet": snippet})

    if not jobs:
        return {"available": False, "jobs": [], "message": f"No similar postings found for '{title}' in {location} right now."}

    return {"available": True, "jobs": jobs, "message": f"Found {len(jobs)} similar posting(s) currently live."}
