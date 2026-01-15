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

# Higher priority keywords for finding the right page
CONTACT_PRIORITY = ["contact", "contact-us", "kontakt", "get-in-touch", "reach-us", "connect", "call-us", "support"]
LEGAL_PRIORITY = ["impressum", "imprint", "legal", "privacy", "terms"]
SALES_PRIORITY = ["sales", "request-quote", "quote", "inquiry", "enquiry", "team", "customer-service"]
LOC_PRIORITY = ["where-to-buy", "locations", "location", "store-locator", "find-us", "findus"]
ABOUT_PRIORITY = ["about", "about-us", "ueber-uns", "über-uns", "unternehmen"]

CONTACT_HINTS = tuple(CONTACT_PRIORITY + ABOUT_PRIORITY + LEGAL_PRIORITY + SALES_PRIORITY + LOC_PRIORITY)

COMMON_FALLBACK_PATHS = (
    "/contact", "/contact-us", "/kontakt", "/get-in-touch", "/reach-us", "/support",
    "/impressum", "/imprint", "/legal",
    "/sales", "/request-quote", "/quote", "/team",
    "/where-to-buy", "/locations", "/location", "/store-locator",
    "/about", "/about-us", "/ueber-uns", "/über-uns",
)

MAX_HTML_CHARS_PER_SITE = 25000      # reduce slightly for speed; still enough for contact pages
THREADS = 12                          # crawl parallelism
GEMINI_BATCH_SIZE = 4                 # 2-3 websites per call
GEMINI_WORKERS = 3                    # parallel gemini calls (keep small to avoid rate limits)
POLITE_DELAY = 0.12                   # small delay
REQUEST_TIMEOUT = 10


# ----------------------------
# Utility: safe JSON parse
# ----------------------------
def parse_json_loose(raw: str) -> Any:
    """
    Robust JSON extractor:
    - Handles text before/after JSON
    - Handles multiple JSON blocks
    - Safely extracts the FIRST valid JSON object or array
    """
    if not raw:
        raise ValueError("Empty response")

    raw = raw.strip()

    # 1) Try direct parse first (fast path)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Find first JSON object or array by balanced braces
    def extract_first_json(s: str) -> Optional[str]:
        stack = []
        start = None

        for i, ch in enumerate(s):
            if ch in "{[":
                if not stack:
                    start = i
                stack.append(ch)

            elif ch in "}]":
                if not stack:
                    continue
                stack.pop()
                if not stack and start is not None:
                    return s[start:i + 1]

        return None

    candidate = extract_first_json(raw)
    if not candidate:
        raise ValueError("No valid JSON found in response:\n" + raw[:1000])

    try:
        return json.loads(candidate)
    except Exception as e:
        raise ValueError(
            "Extracted JSON but failed to parse:\n"
            + candidate[:1000]
        ) from e


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

def discover_candidates(user_query: str, max_candidates: int = 20) -> List[Dict[str, str]]:
    # Run TWO Perplexity discovery queries:
    # 1) normal query
    # 2) query that explicitly uses exclusion operators (-site:) to avoid common directories/marketplaces
    def _pplx_discover(user_prompt: str) -> List[Dict[str, str]]:
        resp = pplx_client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DISCOVERY},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content
        parsed = parse_json_loose(raw)
        return normalize_candidates(parsed)

    prompt1 = f"""
User query: {user_query}
Return up to {max_candidates} candidate companies with official websites.
JSON only.
"""

    # Exclusion-operator query (2nd pass). Perplexity generally understands -site: exclusions in the query string.
    exclude_sites = [
        "alibaba.com", "aliexpress.com", "indiamart.com", "thomasnet.com", "yelp.com",
        "yellowpages.com", "crunchbase.com", "wikipedia.org", "linkedin.com", "facebook.com",
        "instagram.com", "x.com", "twitter.com", "amazon.com", "ebay.com", "made-in-china.com",
        "global.sources.com", "globalsources.com", "kompass.com"
    ]
    exclude_clause = " ".join([f"-site:{d}" for d in exclude_sites])

    prompt2 = f"""
User query: {user_query}

Important:
- Treat the following as a search/exploration query string and apply exclusion operators:
  {exclude_clause}

Return up to {max_candidates} candidate companies with official websites.
JSON only.
"""

    print(f"[DISCOVER] query='{user_query}'")
    c1 = _pplx_discover(prompt1)
    print(f"[DISCOVER] pass1 candidates={len(c1)}")

    c2 = _pplx_discover(prompt2)
    print(f"[DISCOVER] pass2 candidates={len(c2)} (with -site: exclusions)")

    # Merge + dedupe (keep pass1 ordering first)
    merged: List[Dict[str, str]] = []
    seen = set()

    def _key(item: Dict[str, str]) -> str:
        site = (item.get("website") or "").strip().lower()
        # normalize common trailing slashes
        site = site.rstrip("/")
        return site

    for c in (c1 + c2):
        if not c.get("business_name") or not c.get("website"):
            continue
        k = _key(c)
        if not k or k in seen:
            continue
        seen.add(k)
        merged.append({"business_name": c["business_name"], "website": c["website"]})
        if len(merged) >= max_candidates:
            break

    print(f"[DISCOVER] candidates={len(merged)}")
    for i, c in enumerate(merged, 1):
        print(f"  {i}. {c['business_name']} -> {c['website']}")
    return merged


# ----------------------------
# Local extraction (cheap backfill)
# ----------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+|00)\s?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}(?:[\s\-\.]?\d{1,4})?")

def normalize_phone(p: str) -> Optional[str]:
    cleaned = re.sub(r"[^\d+]", "", p)
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 7 or len(digits) > 18:
        return None
    return cleaned

def deobfuscate_text(s: str) -> str:
    s = re.sub(r"\s*\[\s*at\s*\]\s*|\s*\(\s*at\s*\)\s*|\s+at\s+", "@", s, flags=re.I)
    s = re.sub(r"\s*\[\s*dot\s*\]\s*|\s*\(\s*dot\s*\)\s*|\s+dot\s+", ".", s, flags=re.I)
    return s

def extract_local_contacts_from_soup(soup: BeautifulSoup) -> Dict[str, List[str]]:
    # visible text
    text = soup.get_text(" ", strip=True)
    text = deobfuscate_text(text)

    emails = set(e.lower() for e in EMAIL_RE.findall(text))
    phones = set()

    for m in PHONE_RE.findall(text):
        n = normalize_phone(m)
        if n:
            phones.add(n)

    # mailto / tel
    for a in soup.select('a[href^="mailto:"]'):
        e = a.get("href", "").replace("mailto:", "").split("?")[0].strip().lower()
        if e:
            emails.add(e)

    for a in soup.select('a[href^="tel:"]'):
        p = a.get("href", "").replace("tel:", "").split("?")[0].strip()
        n = normalize_phone(p)
        if n:
            phones.add(n)

    # remove common placeholders
    emails = {e for e in emails if "example.com" not in e and "email.com" not in e}

    return {"emails": sorted(emails), "phone_numbers": sorted(phones)}


# ----------------------------
# Crawler helpers
# ----------------------------
def same_reg_domain(base_url: str, other_url: str) -> bool:
    b = tldextract.extract(base_url)
    o = tldextract.extract(other_url)
    return (b.domain, b.suffix) == (o.domain, o.suffix)

def fetch_html(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
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

def clean_html_to_evidence(html: str, max_chars: int = MAX_HTML_CHARS_PER_SITE) -> Tuple[str, Dict[str, List[str]]]:
    """
    Cost/time reducer:
    - remove scripts/styles/iframes
    - keep only text-ish content
    - truncate
    - also do local extraction from the cleaned soup
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    local = extract_local_contacts_from_soup(soup)

    # Build "evidence" string (faster for Gemini than full HTML)
    # include hrefs to help Gemini see mailto/tel even if not visible
    hrefs = []
    for a in soup.select("a[href]"):
        h = a.get("href")
        if h and (h.startswith("mailto:") or h.startswith("tel:")):
            hrefs.append(h)
        if len(hrefs) >= 40:
            break

    text = soup.get_text("\n", strip=True)
    text = deobfuscate_text(text)
    text = re.sub(r"\n{2,}", "\n", text)

    evidence = "LINK_HINTS:\n" + "\n".join(hrefs) + "\n\nPAGE_TEXT:\n" + text
    evidence = re.sub(r"[ \t]+", " ", evidence).strip()

    if len(evidence) > max_chars:
        evidence = evidence[:max_chars]

    return evidence, local

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
      0) contact/support/get-in-touch
      1) imprint/legal
      2) sales/team/quote
      3) locations/where-to-buy
      4) about
      5) everything else
    """
    path = (urlparse(url).path or "").lower()

    if any(k in path for k in CONTACT_PRIORITY):
        bucket = 0
    elif any(k in path for k in LEGAL_PRIORITY):
        bucket = 1
    elif any(k in path for k in SALES_PRIORITY):
        bucket = 2
    elif any(k in path for k in LOC_PRIORITY):
        bucket = 3
    elif any(k in path for k in ABOUT_PRIORITY):
        bucket = 4
    else:
        bucket = 5

    return (bucket, len(path))

def pick_top_pages(home_url: str, home_html: str, top_k: int = 3) -> List[str]:
    """
    Return up to top_k internal pages ranked by score.
    """
    links = extract_internal_links(home_url, home_html)
    ranked = sorted(links, key=score_contact_link)

    picked = []
    for u in ranked:
        path = (urlparse(u).path or "").lower()
        if any(h in path for h in CONTACT_HINTS):
            if u not in picked:
                picked.append(u)
        if len(picked) >= top_k:
            break
    return picked

def try_fallback_paths(home_url: str) -> List[str]:
    parsed = urlparse(home_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    out = []
    for p in COMMON_FALLBACK_PATHS:
        out.append(base + p)
    return out

@dataclass
class CrawledSite:
    business_name: str
    website: str
    source_page: str
    evidence: str
    local_emails: List[str]
    local_phones: List[str]
    extra_pages: List[str]  # for 2nd-pass retry


def normalize_website(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        return "https://" + url
    return url

def fetch_best_page_evidence(website: str, business_name: str) -> Optional[CrawledSite]:
    website = normalize_website(website)

    print(f"[CRAWL] {business_name} -> {website}")

    home_raw = fetch_html(website)
    if not home_raw:
        print(f"[CRAWL]   FAIL homepage")
        return None

    # Find best candidates from internal links
    top_pages = pick_top_pages(website, home_raw, top_k=3)

    # Build a list of candidates to try in order (best link pages first, then fallback paths, then homepage)
    candidates = []
    candidates.extend(top_pages)
    candidates.extend(try_fallback_paths(website))
    candidates.append(website)

    tried = set()
    extra_pages = []

    for u in candidates:
        if u in tried:
            continue
        tried.add(u)

        time.sleep(POLITE_DELAY)
        raw = fetch_html(u)
        if not raw:
            continue

        evidence, local = clean_html_to_evidence(raw)
        # Keep other high-ranked pages for retry (but don’t fetch now)
        extra_pages = [x for x in top_pages if x != u][:2]
        print(f"[CRAWL]   picked: {u}  (evidence_chars={len(evidence)}, local_emails={len(local['emails'])}, local_phones={len(local['phone_numbers'])})")

        return CrawledSite(
            business_name=business_name,
            website=website,
            source_page=u,
            evidence=evidence,
            local_emails=local["emails"],
            local_phones=local["phone_numbers"],
            extra_pages=extra_pages
        )

    # fallback: use homepage evidence even if link selection failed (should not hit often)
    evidence, local = clean_html_to_evidence(home_raw)
    print(f"[CRAWL]   fallback homepage (evidence_chars={len(evidence)})")
    return CrawledSite(
        business_name=business_name,
        website=website,
        source_page=website,
        evidence=evidence,
        local_emails=local["emails"],
        local_phones=local["phone_numbers"],
        extra_pages=[]
    )

def crawl_sites_parallel(candidates: List[Dict[str, str]], threads: int = THREADS) -> List[CrawledSite]:
    print(f"[CRAWL] parallel threads={threads}")
    crawled: List[CrawledSite] = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(fetch_best_page_evidence, c["website"], c["business_name"]) for c in candidates]
        for fut in as_completed(futures):
            item = fut.result()
            if item:
                crawled.append(item)
    print(f"[CRAWL] done ok={len(crawled)}")
    return crawled


# ----------------------------
# Gemini batch extraction (parallel)
# ----------------------------
gemini_client = genai.Client(api_key=GEMINI_KEY)

GEMINI_PROMPT = """
You are a strict information extraction engine.

You will receive 2-3 website documents. Each item contains:
- business_name_hint
- website
- source_page
- evidence (link hints + page text)

Extract for each item:
- business_name
- phone_numbers
- emails
- addresses

CRITICAL:
- Extract ONLY what is present in the evidence. Do NOT guess.
- If unknown, use null or [].
- Return JSON ONLY.

Schema:
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
- business_name: prefer official legal/company name in evidence; else business_name_hint; else null.
- Extract obfuscated emails like "info (at) domain (dot) com" if present.
- Exclude placeholder emails like example@domain.com.
- Deduplicate phones/emails/addresses.
"""

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def gemini_extract_batch(batch: List[CrawledSite]) -> List[Dict[str, Any]]:
    items = []
    for s in batch:
        items.append({
            "business_name_hint": s.business_name,
            "website": s.website,
            "source_page": s.source_page,
            "evidence": s.evidence
        })

    user_payload = {"items": items}
    user_text = GEMINI_PROMPT + "\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False)

    resp = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": user_text}]}],
    )

    raw = (resp.text or "").strip()
    parsed = parse_json_loose(raw)
    return normalize_gemini_vendors(parsed)

def gemini_extract_all_parallel(crawled: List[CrawledSite]) -> List[Dict[str, Any]]:
    batches = chunk_list(crawled, GEMINI_BATCH_SIZE)
    print(f"[GEMINI] batches={len(batches)} batch_size={GEMINI_BATCH_SIZE} workers={GEMINI_WORKERS}")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=GEMINI_WORKERS) as ex:
        fut_map = {ex.submit(gemini_extract_batch, b): b for b in batches}
        for fut in as_completed(fut_map):
            vendors = fut.result()
            results.extend(vendors)

    print(f"[GEMINI] extracted_records={len(results)}")
    return results


# ----------------------------
# Backfill + retry
# ----------------------------
def build_site_index(crawled: List[CrawledSite]) -> Dict[str, CrawledSite]:
    # key by canonical domain (or website string)
    return {c.website: c for c in crawled}

def backfill_vendor(v: Dict[str, Any], site: CrawledSite) -> Dict[str, Any]:
    # Fill phones/emails from local extraction if Gemini missed
    if not v.get("emails"):
        v["emails"] = site.local_emails
    if not v.get("phone_numbers"):
        v["phone_numbers"] = site.local_phones

    # Ensure website/source_page are filled
    if not v.get("website"):
        v["website"] = site.website
    if not v.get("source_page"):
        v["source_page"] = site.source_page

    return v

def needs_retry(v: Dict[str, Any]) -> bool:
    return (not v.get("emails")) and (not v.get("phone_numbers"))

def retry_failed_sites(failed_sites: List[CrawledSite]) -> List[CrawledSite]:
    """
    For each failed site:
    - fetch ONE additional page (best extra page, else first fallback contact)
    - merge evidence with existing evidence (truncate)
    - update local contacts union
    """
    updated: List[CrawledSite] = []

    for site in failed_sites:
        # choose 1 extra page to try
        extra = None
        if site.extra_pages:
            extra = site.extra_pages[0]
        else:
            # first fallback is usually /contact
            fallbacks = try_fallback_paths(site.website)
            extra = fallbacks[0] if fallbacks else None

        if not extra:
            updated.append(site)
            continue

        print(f"[RETRY] {site.business_name} -> fetching extra page: {extra}")
        raw = fetch_html(extra)
        if not raw:
            print(f"[RETRY]   FAIL extra page")
            updated.append(site)
            continue

        evidence2, local2 = clean_html_to_evidence(raw)

        # merge evidence (keep short)
        merged = (site.evidence + "\n\n----- EXTRA_PAGE -----\n\n" + evidence2).strip()
        if len(merged) > MAX_HTML_CHARS_PER_SITE:
            merged = merged[:MAX_HTML_CHARS_PER_SITE]

        # union local contacts
        emails = sorted(set(site.local_emails) | set(local2["emails"]))
        phones = sorted(set(site.local_phones) | set(local2["phone_numbers"]))

        updated.append(CrawledSite(
            business_name=site.business_name,
            website=site.website,
            source_page=extra,           # update source to the extra page
            evidence=merged,
            local_emails=emails,
            local_phones=phones,
            extra_pages=site.extra_pages[1:] if site.extra_pages else []
        ))

        time.sleep(POLITE_DELAY)

    return updated


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(user_query: str, max_candidates: int = 20) -> Dict[str, Any]:
    # 1) Discover
    candidates = discover_candidates(user_query, max_candidates=max_candidates)
    if not candidates:
        return {"vendors": []}

    # 2) Crawl in parallel
    crawled = crawl_sites_parallel(candidates, threads=THREADS)
    if not crawled:
        return {"vendors": []}

    # 3) Gemini extraction (parallel batches)
    vendors = gemini_extract_all_parallel(crawled)

    # 4) Backfill from local extraction
    idx = build_site_index(crawled)
    filled: List[Dict[str, Any]] = []
    retry_targets: List[CrawledSite] = []

    # Match returned vendors to crawled sites by website (best effort)
    for v in vendors:
        site = idx.get(v.get("website") or "", None)
        if not site:
            # If Gemini didn't return website, try to match by business_name hint
            # (fallback: keep as-is)
            filled.append(v)
            continue

        v2 = backfill_vendor(v, site)
        filled.append(v2)

        if needs_retry(v2):
            retry_targets.append(site)

    # 5) Retry only failed (phones+emails both empty)
    if retry_targets:
        print(f"[RETRY] targets={len(retry_targets)} -> second pass (only misses)")
        updated_sites = retry_failed_sites(retry_targets)

        # Gemini again only for those (parallel batches)
        retry_results = gemini_extract_all_parallel(updated_sites)

        # Backfill again and overwrite matching websites in filled list
        retry_idx = {s.website: s for s in updated_sites}
        retry_map = {}
        for v in retry_results:
            site = retry_idx.get(v.get("website") or "", None)
            if site:
                retry_map[site.website] = backfill_vendor(v, site)

        # apply overwrite
        for i, old in enumerate(filled):
            w = old.get("website")
            if w in retry_map:
                filled[i] = retry_map[w]

    # 6) Final output
    return {"vendors": filled}


# ----------------------------
# Minimal CLI (no beautification)
# ----------------------------
if __name__ == "__main__":
    while True:
        q = "Find the contact information of " + input("Describe the product/service vendors you need: ").strip()
        if not q:
            continue
        try:
            out = run_pipeline(q, max_candidates=20)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Error:", str(e))
