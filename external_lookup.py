import os
import re
from urllib.parse import urlparse

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


_RANGE_PATTERNS = (
    re.compile(r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakh|lac|l\b)", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:l|lpa|lakh|lac)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*(?:l|lpa|lakh|lac)\b", re.IGNORECASE),
)
_SINGLE_PATTERN = re.compile(r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakh(?:s)?(?:\s+per\s+annum)?|lac(?:s)?)", re.IGNORECASE)


def _extract_lpa_range(text):
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            low, high = sorted((float(match.group(1)), float(match.group(2))))
            if 0.5 <= low <= high <= 500:
                return low, high
    values = [float(match.group(1)) for match in _SINGLE_PATTERN.finditer(text or "")]
    if values and 0.5 <= values[0] <= 500:
        return round(values[0] * 0.85, 1), round(values[0] * 1.15, 1)
    return None


def try_external_lookup(profile):
    unavailable = {"available": False, "low_lpa": None, "high_lpa": None, "source": None}
    if not _TAVILY_SDK_AVAILABLE:
        return {**unavailable, "message": "Live salary lookup is unavailable because tavily-python is not installed."}
    api_key = _get_api_key()
    if not api_key:
        return {**unavailable, "message": "Live salary lookup is not configured."}

    company = (profile.get("company_name") or "").strip()
    title = (profile.get("title") or "").strip()
    location = (profile.get("location") or "India").strip()
    exp_min = profile.get("experience_min", 0)
    exp_max = profile.get("experience_max", exp_min)
    query = f'"{company}" "{title}" salary India {location} {exp_min}-{exp_max} years experience LPA'
    try:
        response = TavilyClient(api_key=api_key).search(query=query, search_depth="basic", include_answer=True, max_results=5)
    except Exception as error:
        return {**unavailable, "message": f"Live salary lookup failed ({type(error).__name__})."}

    candidates = [(response.get("answer", ""), None)]
    candidates.extend((result.get("content", ""), result.get("url")) for result in response.get("results", []))
    for text, url in candidates:
        salary_range = _extract_lpa_range(text)
        if salary_range:
            source = urlparse(url).netloc if url else "web search"
            return {"available": True, "low_lpa": salary_range[0], "high_lpa": salary_range[1], "source": source,
                    "message": f"Found a web-sourced estimate for '{company}' / '{title}'."}
    return {**unavailable, "message": "No reliable salary figure was found in live search results."}
