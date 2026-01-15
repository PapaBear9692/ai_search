import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import tldextract
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from perplexity import Perplexity
from google import genai


# ----------------------------
# Config
# ----------------------------
load_dotenv()
PPLX_KEY = os.getenv("PERPLEXITY_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not PPLX_KEY:
    raise RuntimeError("Set PERPLEXITY_API_KEY in .env")
if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

HEADERS = {"User-Agent": "VendorContactFinder/1.0 (+contact: you@example.com)"}

# IMPORTANT: We still keep CONTACT_HINTS for broad matching,
# but scoring will prioritize "contact" > "imprint/legal" > "sales/team" > "locations" > "about"
CONTACT_HINTS = (
    "contact", "contact-us", "kontakt",
    "impressum", "imprint", "legal",
    "sales", "team", "staff", "people",
    "locations", "location", "store-locator", "where-to-buy", "find-us", "findus",
    "about", "about-us", "unternehmen", "ueber-uns", "über-uns",
)

# IMPORTANT: Try CONTACT first, then imprint/legal, then sales/team, then locations, then about
COMMON_FALLBACK_PATHS = (
    "/contact", "/contact-us", "/kontakt",
    "/impressum", "/imprint", "/legal",
    "/sales", "/team", "/staff", "/people",
    "/where-to-buy", "/locations", "/location", "/store-locator",
    "/about", "/about-us", "/ueber-uns", "/über-uns",
)

MAX_HTML_CHARS_PER_SITE = 120_000     # cost control
THREADS = 12                          # crawl parallelism
GEMINI_BATCH_SIZE = 3                 # 2-3 websites per call
POLITE_DELAY = 0.15                   # keep low but non-zero


# ----------------------------
# Utility: safe JSON parse
# ----------------------------
def parse_json_loose(raw: str) -> Any:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m_obj = re.search(r"\{.*\}", raw, re.DOTALL)
        if m_obj:
            return json.loads(m_obj.group(0))
        m_arr = re.search(r"\[.*\]", raw, re.DOTALL)
        if m_arr:
            return json.loads(m_arr.group(0))
        raise ValueError("Model output is not valid JSON:\n" + raw[:2000])


def normalize_candidates(discovery: Any) -> List[Dict[str, str]]:
    if isinstance(discovery, dict):
        candidates = discovery.get("candidates", [])
    elif isinstance(discovery, list):
        candidates = discovery
    else:
        candidates = []

    out = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = (c.get("business_name") or "").strip()
        site = (c.get("website") or "").strip()
        if name and site:
            out.append({"business_name": name, "website": site})
    return out


def normalize_gemini_vendors(extracted: Any) -> List[Dict[str, Any]]:
    if isinstance(extracted, dict):
        vendors = extracted.get("vendors", [])
    elif isinstance(extracted, list):
        vendors = extracted
    else:
        vendors = []

    cleaned = []
    for v in vendors:
        if not isinstance(v, dict):
            continue
        cleaned.append({
            "business_name": v.get("business_name"),
            "website": v.get("website"),
            "source_page": v.get("source_page"),
            "phone_numbers": v.get("phone_numbers") or [],
            "emails": v.get("emails") or [],
            "addresses": v.get("addresses") or [],
        })
    return cleaned


# ----------------------------
# Perplexity: discovery
# ----------------------------
pplx_client = Perplexity(api_key=PPLX_KEY)

SYSTEM_PROMPT_DISCOVERY = """
You are a sourcing agent.

Task:
Given a user query about vendors/wholesalers/distributors/manufacturers,
return ONLY valid JSON listing candidate companies with their official websites.

Output format (either is acceptable):
A) { "candidates": [ { "business_name": "...", "website": "..." } ] }
OR
B) [ { "business_name": "...", "website": "..." } ]

Rules:
- Prefer official company websites (not directories, marketplaces, Wikipedia, LinkedIn, Facebook).
- Avoid: alibaba, indiamart, yellowpages, yelp, thomasnet, crunchbase, wikipedia, linkedin, facebook, instagram.
- Focus on companies/corporations, not individuals.
- Return up to 10 candidates.
- JSON ONLY. No extra keys. No explanation.
"""

def discover_candidates(user_query: str, max_candidates: int = 10) -> List[Dict[str, str]]:
    prompt = f"""
User query: {user_query}
Return up to {max_candidates} candidate companies with official websites.
JSON only.
"""
    print(f"[DISCOVER] Query: {user_query}")
    resp = pplx_client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_DISCOVERY},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content
    parsed = parse_json_loose(raw)
    candidates = normalize_candidates(parsed)[:max_candidates]
    print(f"[DISCOVER] Found {len(candidates)} candidate websites")
    return candidates


# ----------------------------
# Crawler helpers
# ----------------------------
def same_reg_domain(base_url: str, other_url: str) -> bool:
    b = tldextract.extract(base_url)
    o = tldextract.extract(other_url)
    return (b.domain, b.suffix) == (o.domain, o.suffix)

def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            return None
        ctype = r.headers.get("Content-Type", "")
        if ctype and ("text/html" not in ctype and "application/xhtml+xml" not in ctype):
            return None
        return r.text
    except Exception:
        return None

def clean_and_truncate_html(html: str, max_chars: int = MAX_HTML_CHARS_PER_SITE) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    cleaned = str(soup)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned

def extract_internal_links(base_url: str, html: str, limit: int = 250) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        if abs_url.startswith("http") and same_reg_domain(base_url, abs_url):
            links.append(abs_url)
        if len(links) >= limit:
            break
    return list(dict.fromkeys(links))

def score_contact_link(url: str) -> Tuple[int, int]:
    """
    Prioritize:
      0) contact
      1) impressum/imprint/legal
      2) sales/team/staff/people
      3) locations/where-to-buy/store-locator
      4) about
      5) everything else
    """
    path = (urlparse(url).path or "").lower()

    if any(k in path for k in ["contact", "contact-us", "kontakt"]):
        bucket = 0
    elif any(k in path for k in ["impressum", "imprint", "legal"]):
        bucket = 1
    elif any(k in path for k in ["sales", "team", "staff", "people"]):
        bucket = 2
    elif any(k in path for k in ["where-to-buy", "locations", "location", "store-locator", "find-us", "findus"]):
        bucket = 3
    elif any(k in path for k in ["about", "about-us", "ueber-uns", "über-uns", "unternehmen"]):
        bucket = 4
    else:
        bucket = 5

    # Shorter paths generally better; also prefer non-empty paths
    return (bucket, len(path))

def pick_best_contact_page(home_url: str, home_html: str) -> Optional[str]:
    links = extract_internal_links(home_url, home_html)
    links_sorted = sorted(links, key=score_contact_link)

    # Choose first strongly relevant link
    for u in links_sorted[:120]:
        path = (urlparse(u).path or "").lower()
        if any(h in path for h in CONTACT_HINTS):
            return u
    return None

def try_fallback_paths(home_url: str) -> Optional[str]:
    parsed = urlparse(home_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for p in COMMON_FALLBACK_PATHS:
        candidate = base + p
        html = fetch_html(candidate)
        if html:
            return candidate
        time.sleep(POLITE_DELAY)
    return None


@dataclass
class CrawledSite:
    business_name: str
    website: str
    source_page: str
    html: str

def normalize_website(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        return "https://" + url
    return url

def crawl_one_site(business_name: str, website: str) -> Optional[CrawledSite]:
    website = normalize_website(website)

    print(f"[CRAWL] Start: {business_name} -> {website}")
    home_html = fetch_html(website)
    if not home_html:
        print(f"[CRAWL] FAIL homepage fetch: {website}")
        return None

    # 1) try best prioritized page found in homepage links
    best_url = pick_best_contact_page(website, home_html)
    if best_url:
        print(f"[CRAWL] Selected from links: {best_url}")
        page_html = fetch_html(best_url)
        if page_html:
            cleaned = clean_and_truncate_html(page_html)
            print(f"[CRAWL] OK page fetch: {best_url} (chars={len(cleaned)})")
            return CrawledSite(
                business_name=business_name,
                website=website,
                source_page=best_url,
                html=cleaned
            )
        else:
            print(f"[CRAWL] FAIL page fetch (selected): {best_url}")

    # 2) fallback paths (contact first)
    fallback = try_fallback_paths(website)
    if fallback:
        print(f"[CRAWL] Selected fallback: {fallback}")
        fb_html = fetch_html(fallback)
        if fb_html:
            cleaned = clean_and_truncate_html(fb_html)
            print(f"[CRAWL] OK fallback fetch: {fallback} (chars={len(cleaned)})")
            return CrawledSite(
                business_name=business_name,
                website=website,
                source_page=fallback,
                html=cleaned
            )
        else:
            print(f"[CRAWL] FAIL fallback fetch: {fallback}")

    # 3) homepage fallback
    cleaned_home = clean_and_truncate_html(home_html)
    print(f"[CRAWL] Using homepage fallback (chars={len(cleaned_home)})")
    return CrawledSite(
        business_name=business_name,
        website=website,
        source_page=website,
        html=cleaned_home
    )

def crawl_sites_parallel(candidates: List[Dict[str, str]], threads: int = THREADS) -> List[CrawledSite]:
    crawled: List[CrawledSite] = []
    print(f"[CRAWL] Parallel crawling with {threads} threads...")
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(crawl_one_site, c["business_name"], c["website"]) for c in candidates]
        for fut in as_completed(futures):
            item = fut.result()
            if item:
                crawled.append(item)
    print(f"[CRAWL] Completed. Successfully crawled {len(crawled)} sites.")
    return crawled


# ----------------------------
# Gemini batch extraction
# ----------------------------
gemini_client = genai.Client(api_key=GEMINI_KEY)

GEMINI_PROMPT = """
You are a strict information extraction engine.

You will receive 2-3 website HTML documents. For each item, extract:
- business_name
- phone_numbers
- emails
- addresses

CRITICAL:
- Extract ONLY what is present in the HTML. Do NOT guess.
- If unknown, use null or [].
- Return JSON ONLY.

Schema (return exactly this shape):
{
  "vendors": [
    {
      "business_name": string | null,
      "website": string | null,
      "source_page": string | null,
      "phone_numbers": string[],
      "emails": string[],
      "addresses": string[]
    }
  ]
}

Rules:
- business_name: prefer official legal/company name in HTML; else business_name_hint; else null.
- Also extract obfuscated emails like "info (at) domain (dot) com" if present.
- Exclude placeholder emails like example@domain.com.
- Deduplicate phones/emails/addresses.
"""

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def gemini_extract_batch(batch: List[CrawledSite]) -> List[Dict[str, Any]]:
    print(f"[GEMINI] Extracting batch of {len(batch)} site(s)...")
    items = []
    for s in batch:
        items.append({
            "business_name_hint": s.business_name,
            "website": s.website,
            "source_page": s.source_page,
            "html": s.html
        })

    user_payload = {"items": items}
    user_text = GEMINI_PROMPT + "\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False)

    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": user_text}]}],
    )

    raw = (resp.text or "").strip()
    parsed = parse_json_loose(raw)
    vendors = normalize_gemini_vendors(parsed)
    print(f"[GEMINI] Batch done. Extracted {len(vendors)} vendor record(s).")
    return vendors


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(user_query: str, max_candidates: int = 10) -> Dict[str, Any]:
    # 1) Discover
    candidates = discover_candidates(user_query, max_candidates=max_candidates)
    if not candidates:
        return {"vendors": []}

    for i, c in enumerate(candidates, start=1):
        print(f"[DISCOVER] {i}. {c['business_name']} -> {c['website']}")

    # 2) Crawl in parallel
    crawled = crawl_sites_parallel(candidates, threads=THREADS)
    if not crawled:
        return {"vendors": []}

    # 3) Gemini extraction in batches (2-3 per call)
    all_vendors: List[Dict[str, Any]] = []
    batches = chunk_list(crawled, GEMINI_BATCH_SIZE)
    print(f"[PIPELINE] Sending {len(crawled)} crawled pages to Gemini in {len(batches)} batch(es)...")

    for idx, batch in enumerate(batches, start=1):
        print(f"[PIPELINE] Gemini batch {idx}/{len(batches)}")
        vendors = gemini_extract_batch(batch)
        all_vendors.extend(vendors)
        time.sleep(0.2)  # avoid hammering API

    return {"vendors": all_vendors}


# ----------------------------
# Minimal CLI (no beautification)
# ----------------------------
if __name__ == "__main__":
    while True:
        q = input("Describe the product/service vendors you need: ").strip()
        if not q:
            continue

        try:
            out = run_pipeline(q, max_candidates=10)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Error:", str(e))
