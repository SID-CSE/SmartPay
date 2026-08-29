"""
external_lookup.py -- Live web-search salary lookup via Tavily.

Backend: Tavily (https://tavily.com) -- chosen because, as of August 2026:
  - 1,000 free API credits/month, recurring (resets monthly), NO credit card required
  - Purpose-built for AI apps: returns a clean synthesized "answer" plus source URLs,
    instead of raw HTML you'd have to scrape/parse yourself
  - Bing Search API was retired Aug 2025; Brave's free tier was withdrawn late 2025;
    Google Custom Search is closed to new signups and shutting down entirely Jan 2027 --
    Tavily is the most viable genuinely-free option left as of this writing.

SETUP (takes ~2 minutes):
  1. Go to https://app.tavily.com and sign up (email or Google/GitHub, no card needed)
  2. Copy your API key from the dashboard (starts with "tvly-")
  3. Provide it to this app via EITHER:
       a) Environment variable:  export TAVILY_API_KEY="tvly-..."
       b) Streamlit secrets:     add TAVILY_API_KEY = "tvly-..." to .streamlit/secrets.toml
  4. That's it -- no code changes needed. The app auto-detects the key and activates
     live lookup; if no key is found, it cleanly falls back to "unavailable" (same
     behavior as before, nothing breaks).

Each unknown-company lookup costs 1 Tavily credit (basic search). At 1,000 free
credits/month, that's up to 1,000 "we don't recognize this company" lookups per month
at zero cost -- comfortably enough for personal/small-scale use.
"""
import os
import re

try:
    from tavily import TavilyClient
    _TAVILY_SDK_AVAILABLE = True
except ImportError:
    _TAVILY_SDK_AVAILABLE = False


def _get_api_key():
    """Checks Streamlit secrets first (works on Streamlit Cloud), then env var (works everywhere else)."""
    try:
        import streamlit as st
        if "TAVILY_API_KEY" in st.secrets:
            return st.secrets["TAVILY_API_KEY"]
    except Exception:
        pass  # not running inside Streamlit, or no secrets.toml configured -- fine, fall through
    return os.environ.get("TAVILY_API_KEY")


# Matches "10-15 LPA" / "10 to 15 lakh" style (single unit suffix after the range)
_LPA_RANGE_RE = re.compile(
    r'(?:\u20b9\s*)?(\d+(?:\.\d+)?)\s*(?:-|to|\u2013)\s*(?:\u20b9\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakh|lac|l\b)',
    re.IGNORECASE,
)
# Matches "6L-9L" / "6L to 9L" style (unit attached to each number individually)
_LPA_RANGE_PERNUM_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:l|lpa|lakh|lac)\s*(?:-|to|\u2013)\s*(\d+(?:\.\d+)?)\s*(?:l|lpa|lakh|lac)\b',
    re.IGNORECASE,
)
_LPA_SINGLE_RE = re.compile(
    r'(?:\u20b9\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakh(?:s)?(?:\s+per\s+annum)?|lac(?:s)?)',
    re.IGNORECASE,
)


def _extract_lpa_range(text):
    """Pulls a plausible (low_lpa, high_lpa) out of free text. Returns None if nothing found."""
    if not text:
        return None
    for pattern in (_LPA_RANGE_RE, _LPA_RANGE_PERNUM_RE):
        m = pattern.search(text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if 0.5 <= lo <= 500 and 0.5 <= hi <= 500:  # sanity bounds for an India annual salary in LPA
                return lo, hi
    singles = [float(m.group(1)) for m in _LPA_SINGLE_RE.finditer(text) if 0.5 <= float(m.group(1)) <= 500]
    if singles:
        val = singles[0]
        return round(val * 0.85, 1), round(val * 1.15, 1)
    return None


def try_external_lookup(profile):
    """
    Attempts a live Tavily search for an unknown company/title combo.

    Returns:
        {
            "available": bool,          # True only if a real figure was found
            "low_lpa": float | None,
            "high_lpa": float | None,
            "source": str | None,       # domain the figure was drawn from, if identifiable
            "message": str,             # always present, human-readable status
        }
    """
    if not _TAVILY_SDK_AVAILABLE:
        return {
            "available": False, "low_lpa": None, "high_lpa": None, "source": None,
            "message": "tavily-python isn't installed. Run: pip install tavily-python",
        }

    api_key = _get_api_key()
    if not api_key:
        return {
            "available": False, "low_lpa": None, "high_lpa": None, "source": None,
            "message": (
                "Live web lookup isn't configured yet. Get a free Tavily API key at "
                "https://app.tavily.com (no card needed, 1,000 free searches/month) and set "
                "TAVILY_API_KEY as an environment variable or in .streamlit/secrets.toml."
            ),
        }

    company = profile.get('company_name', '').strip()
    title = profile.get('title', '').strip()
    location = profile.get('location', '').strip() or 'India'
    exp_min = profile.get('experience_min', 0)
    exp_max = profile.get('experience_max', exp_min)

    query = (
        f'"{company}" "{title}" salary India {location} {exp_min}-{exp_max} years experience LPA '
        f'site:ambitionbox.com OR site:glassdoor.co.in OR site:naukri.com OR site:payscale.com'
    )

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",       # 1 credit per call -- keeps the free 1,000/month going far
            include_answer=True,        # Tavily's own synthesized answer, easiest to parse
            max_results=5,
        )
    except Exception as e:
        return {
            "available": False, "low_lpa": None, "high_lpa": None, "source": None,
            "message": f"Web search failed ({type(e).__name__}). Falling back to role/experience/location-only estimate.",
        }

    candidates = []
    if response.get('answer'):
        candidates.append((response['answer'], None))
    for r in response.get('results', []):
        snippet = r.get('content', '')
        candidates.append((snippet, r.get('url')))

    for text, url in candidates:
        found = _extract_lpa_range(text)
        if found:
            low_lpa, high_lpa = found
            source = None
            if url:
                for domain in ['ambitionbox.com', 'glassdoor.co.in', 'naukri.com', 'payscale.com']:
                    if domain in url:
                        source = domain
                        break
            return {
                "available": True, "low_lpa": low_lpa, "high_lpa": high_lpa,
                "source": source or "web_search",
                "message": f"Found a web-sourced estimate for '{company}' / '{title}'"
                           + (f" via {source}" if source else "") + ".",
            }

    return {
        "available": False, "low_lpa": None, "high_lpa": None, "source": None,
        "message": (
            f"Searched the web for '{company}' / '{title}' but couldn't extract a reliable salary figure "
            "from the results. Falling back to role/experience/location-only estimate."
        ),
    }
