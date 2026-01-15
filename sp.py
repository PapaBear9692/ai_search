import os
import re
import json
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from serpapi import GoogleSearch
from google import genai

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")  # <-- ADD THIS

if not SERPAPI_API_KEY:
    raise RuntimeError("SERPAPI_API_KEY is not set in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")
if not GOOGLE_PLACES_API_KEY:
    raise RuntimeError("GOOGLE_PLACES_API_KEY is not set in .env (Google Places API key)")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

DEFAULT_LOCATION = "Dhaka, Bangladesh"  # SerpAPI location bias
DEFAULT_PLACES_REGION = "bd"            # Places region bias (BD)
DEFAULT_PLACES_LANGUAGE = "en"          # You can set to "bn" if you want


# ============================================================
# HELPERS: SEARCH
# ============================================================

def web_search(query: str, num_results: int = 20):
    """Normal Google search via SerpAPI; returns organic results list."""
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
        "location": DEFAULT_LOCATION,
    }
    search = GoogleSearch(params)
    res = search.get_dict()
    return res.get("organic_results", [])


# ============================================================
# HELPERS: HTML FETCH / PARSE
# ============================================================

def fetch_html(url: str, timeout: int = 12) -> str | None:
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def get_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def guess_contact_urls(root_url: str, html: str, max_links: int = 3):
    """Find likely contact/about/support page links from a homepage."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip().lower()
        href = a["href"].strip()
        if any(k in text for k in ["contact", "contact us", "about","about us", "support", "help"]):
            candidates.append(urljoin(root_url, href))

    # unique preserve order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_links:
            break

    return out or [root_url]


def regex_contacts(text: str):
    email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    phone_re = r"\+?\d[\d\s\-()]{7,}"
    emails = sorted(set(re.findall(email_re, text)))
    phones = sorted(set(re.findall(phone_re, text)))
    return emails, phones


def normalize_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return (url or "").lower()


def safe_domain_for_query(url: str | None) -> str | None:
    if not url:
        return None
    try:
        d = urlparse(url).netloc.lower()
        if d.startswith("www."):
            d = d[4:]
        return d or None
    except Exception:
        return None


# ============================================================
# GEMINI: COMPANY/EMAIL/PHONE EXTRACTION (NO ADDRESS)
# ============================================================

def extract_contacts_with_gemini(text: str, fallback_website: str | None):
    """
    Use Gemini to extract company_name, emails, phones.
    Address will be fetched using Google Places API.
    """
    text = text[:25000]
    prompt = f"""
You are a data extraction assistant.

From the following web page text, extract business contact information and return ONLY valid JSON.

JSON schema:
{{
  "company_name": string or null,
  "website": string or null,
  "emails": string[],
  "phones": string[]
}}

Guidelines:
- Extract only what clearly belongs to the company.
- "emails" should contain only email addresses.
- "phones" should contain only phone numbers.
- Do NOT guess. If unsure, return null or [].
- If you can confidently identify the official website, include it; otherwise null.

Text:
\"\"\"{text}\"\"\"
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    raw = (response.text or "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            data = {}
        else:
            data = json.loads(match.group(0))

    def as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)]

    website = data.get("website") or fallback_website

    return {
        "company_name": data.get("company_name"),
        "website": website,
        "emails": as_list(data.get("emails")),
        "phones": as_list(data.get("phones")),
    }


def merge_contacts(llm_data, text_emails, text_phones):
    emails = sorted(set((llm_data.get("emails") or []) + (text_emails or [])))
    phones = sorted(set((llm_data.get("phones") or []) + (text_phones or [])))
    return {
        "company_name": llm_data.get("company_name"),
        "website": llm_data.get("website"),
        "emails": emails,
        "phones": phones,
    }


# ============================================================
# GOOGLE PLACES API: ADDRESS LOOKUP
# ============================================================

def places_find_place(query_text: str, timeout: int = 12) -> dict | None:
    """
    Uses Find Place From Text to get a place_id for a company.
    Docs: https://developers.google.com/maps/documentation/places/web-service/search-find-place
    """
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query_text,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address,types,geometry",
        "key": GOOGLE_PLACES_API_KEY,
        "region": DEFAULT_PLACES_REGION,
        "language": DEFAULT_PLACES_LANGUAGE,
    }

    try:
        r = httpx.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        return candidates[0]
    except Exception:
        return None


def places_details(place_id: str, timeout: int = 12) -> dict | None:
    """
    Uses Place Details to fetch formatted_address and other useful fields.
    Docs: https://developers.google.com/maps/documentation/places/web-service/details
    """
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,international_phone_number,formatted_phone_number,website,url",
        "key": GOOGLE_PLACES_API_KEY,
        "language": DEFAULT_PLACES_LANGUAGE,
    }

    try:
        r = httpx.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        result = data.get("result")
        if not result:
            return None
        return result
    except Exception:
        return None


def get_address_via_places(company_name: str | None, website: str | None):
    """
    Strategy:
    - Build a query like: "<company_name> <domain>" if possible
    - FindPlaceFromText -> place_id -> PlaceDetails -> formatted_address
    """
    if not company_name and not website:
        return []

    domain = safe_domain_for_query(website)
    # Build query text (helps disambiguation)
    if company_name and domain:
        q = f"{company_name} {domain}"
    elif company_name:
        q = company_name
    else:
        q = domain

    cand = places_find_place(q)
    if not cand:
        return []

    # Sometimes find-place already includes formatted_address; still use details for better accuracy
    place_id = cand.get("place_id")
    if not place_id:
        fa = cand.get("formatted_address")
        return [fa] if fa else []

    det = places_details(place_id)
    if not det:
        fa = cand.get("formatted_address")
        return [fa] if fa else []

    addr = det.get("formatted_address")
    return [addr] if addr else []


# ============================================================
# MAIN PIPELINE
# ============================================================

def find_vendors(query: str, per_query_results: int = 20):
    """
    Pipeline (as you requested):
    1) Normal Google search -> list of websites (SerpAPI organic_results)
    2) For each website: fetch HTML, find contact-ish page, extract company/email/phone (Gemini + regex)
    3) Use Google Places API to find address (not from HTML)
    4) Print merged result
    """
    organic = web_search(query, per_query_results)
    vendors = []

    for result in organic:
        url = result.get("link")
        if not url:
            continue

        html = fetch_html(url)
        if not html:
            continue

        # find best "contact" page (or fall back to homepage)
        contact_urls = guess_contact_urls(url, html, max_links=3)

        best_contact = None
        for cu in contact_urls:
            ch = fetch_html(cu)
            if ch:
                best_contact = (cu, ch)
                break

        if not best_contact:
            best_contact = (url, html)

        source_url, source_html = best_contact
        text = get_text_from_html(source_html)

        emails_rx, phones_rx = regex_contacts(text)

        llm_data = extract_contacts_with_gemini(text, fallback_website=url)
        merged = merge_contacts(llm_data, emails_rx, phones_rx)

        # ADDRESS via Places API
        merged["addresses"] = get_address_via_places(
            company_name=merged.get("company_name"),
            website=merged.get("website"),
        )

        merged["source_url"] = source_url
        vendors.append(merged)

    # deduplicate by domain
    dedup = {}
    for v in vendors:
        dom = normalize_domain(v.get("website") or v.get("source_url") or "")
        if dom and dom not in dedup:
            dedup[dom] = v

    return list(dedup.values())


if __name__ == "__main__":
    while True:
        q = input("Enter vendor/product query: ").strip()
        if not q:
            continue

        results = find_vendors(q, per_query_results=20)

        for v in results:
            print("====")
            print("Company:", v.get("company_name"))
            print("Website:", v.get("website"))
            print("Source URL:", v.get("source_url"))
            print("Emails:", ", ".join(v.get("emails", [])))
            print("Phones:", ", ".join(v.get("phones", [])))
            print("Addresses:")
            for a in v.get("addresses", []):
                print("  -", a)
