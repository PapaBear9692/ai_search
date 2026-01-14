# fast_one_call_gemini.py
# Speed-optimized: fail-fast timeouts, smaller downloads, skip extra page if homepage already has contact signals,
# parallel fetch, single Gemini call, no debug/json dumps.

import os
import re
import json
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any, Optional

import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types


# -----------------------------
# CONFIG (speed-focused)
# -----------------------------
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

MAX_SERP_RESULTS = 20
MAX_WORKERS = 12

# Fail fast (connect, read)
TIMEOUT = (5, 8)

# Download less (we only need footer/header + signal lines)
MAX_HTML_BYTES = 250_000

# Only allow one extra contact/about page
KEYWORDS = ["contact", "contact-us", "contactus", "about", "about-us", "aboutus", "company", "team", "who-we-are"]
FALLBACK_PATHS = ["/contact", "/contact-us", "/about", "/about-us", "/company"]

# Keep one-call payload compact
MAX_SNIPPET_CHARS_PER_PAGE = 6000
MAX_SITE_CHARS = 10_000
MAX_TOTAL_INPUT_CHARS = 160_000  # overall input budget to keep a single call safe


# -----------------------------
# Regex helpers
# -----------------------------
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().\-]{6,}\d)")
ADDR_HINT_RE = re.compile(
    r"\b(address|office|location|registered|hq|headquarters|"
    r"street|road|avenue|suite|zip|postal|city|country|"
    r"building|floor|unit|block|district|province|state|"
    r"straße|strasse|weg|platz|allee|gasse|haus|plz)\b",
    re.IGNORECASE
)


# -----------------------------
# Basics
# -----------------------------
def load_keys():
    load_dotenv()
    if not os.getenv("SERPAPI_API_KEY"):
        raise RuntimeError("Missing SERPAPI_API_KEY in .env")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")


def dom(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (urlparse(url).netloc or url).lower()


def root(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme or 'https'}://{p.netloc}"


# -----------------------------
# SerpAPI
# -----------------------------
def serpapi_links(session: requests.Session, query: str, n: int = 20) -> List[str]:
    params = {"engine": "google", "q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": min(n, 20)}
    r = session.get(SERPAPI_ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    links = []
    for item in (data.get("organic_results") or []):
        u = item.get("link")
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            links.append(u)

    seen, out = set(), []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:n]


# -----------------------------
# Fetching (fast)
# -----------------------------
def fetch_html(session: requests.Session, url: str) -> Optional[str]:
    try:
        with session.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            chunks, total = [], 0
            for chunk in r.iter_content(65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_HTML_BYTES:
                    break
                chunks.append(chunk)
            return b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        return None


def best_about_contact_url(base: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_dom = dom(base)

    best_score, best_url = 0, None
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        text = (a.get_text(" ", strip=True) or "").lower()
        href_l = href.lower()

        u = urljoin(base, href)
        if not u.startswith(("http://", "https://")):
            continue
        if dom(u) != base_dom or u == base:
            continue

        score = 0
        for kw in KEYWORDS:
            if kw in href_l:
                score += 3
            if kw in text:
                score += 2

        if score > best_score:
            best_score, best_url = score, u

    if best_url:
        return best_url

    # one fallback guess only
    for p in FALLBACK_PATHS:
        u = urljoin(base, p)
        if u != base:
            return u
    return None


# -----------------------------
# Snippet extraction (fast + small)
# -----------------------------
def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def signal_lines(soup: BeautifulSoup, window: int = 2, max_lines: int = 140) -> str:
    lines = _lines(soup.get_text("\n"))
    keep = set()

    for i, ln in enumerate(lines):
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or ADDR_HINT_RE.search(ln):
            for j in range(max(0, i - window), min(len(lines), i + window + 1)):
                keep.add(j)

    picked = [lines[i] for i in sorted(keep)]
    if not picked:
        picked = lines[:40]
    return "\n".join(picked[:max_lines])


def snippet(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # header/footer are usually where contact lives
    header = soup.find("header")
    footer = soup.find("footer")

    parts = []

    if header:
        parts.append(f"--- HEADER ({url}) ---\n" + "\n".join(_lines(header.get_text("\n")))[:2500])
    if footer:
        parts.append(f"--- FOOTER ({url}) ---\n" + "\n".join(_lines(footer.get_text("\n")))[:5000])

    parts.append(f"--- SIGNAL ({url}) ---\n" + signal_lines(soup))

    out = "\n\n".join(parts).strip()
    return out[:MAX_SNIPPET_CHARS_PER_PAGE]


# -----------------------------
# Site processing (homepage + maybe 1 extra page)
# -----------------------------
def fetch_site_payload(session: requests.Session, seed_url: str) -> Dict[str, Any]:
    r = root(seed_url)
    d = dom(r)

    payload = {"domain": d, "pages": []}

    home_html = fetch_html(session, r)
    if not home_html:
        return payload  # empty pages -> will be skipped

    home_snip = snippet(home_html, r)
    payload["pages"].append({"url": r, "snippet": home_snip})

    # Speedup: if homepage already has email/phone, don't fetch extra page.
    if EMAIL_RE.search(home_snip) or PHONE_RE.search(home_snip):
        return payload

    extra = best_about_contact_url(r, home_html)
    if extra:
        extra_html = fetch_html(session, extra)
        if extra_html:
            extra_snip = snippet(extra_html, extra)
            remaining = max(0, MAX_SITE_CHARS - len(home_snip))
            payload["pages"].append({"url": extra, "snippet": extra_snip[:remaining]})

    # enforce per-site cap
    combined_len = sum(len(p["snippet"]) for p in payload["pages"])
    if combined_len > MAX_SITE_CHARS and payload["pages"]:
        # trim last page snippet
        overflow = combined_len - MAX_SITE_CHARS
        payload["pages"][-1]["snippet"] = payload["pages"][-1]["snippet"][:-overflow]

    return payload


def trim_sites_to_budget(sites: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    out, total = [], 0
    for s in sites:
        s_text = json.dumps(s, ensure_ascii=False)
        if total + len(s_text) > max_chars:
            break
        out.append(s)
        total += len(s_text)
    return out


# -----------------------------
# One Gemini call for all sites
# -----------------------------
def gemini_one_call(client: genai.Client, sites: List[Dict[str, Any]], model: str = "gemini-3-flash-preview") -> List[Dict[str, Any]]:
    prompt = f"""
You are doing BUSINESS contact discovery.

Input is JSON of multiple websites, each with:
- domain
- pages: list of objects with url and high-signal snippet text (header/footer + lines near emails/phones/addresses)

Task:
For EACH domain, output ONE best row as JSON with exactly these keys:
{{
  "domain": string,
  "company_name": string|null,
  "emails": string[],
  "phones": string[],
  "address": string|null,
  "country": string|null,
  "best_source_url": string|null,
  "confidence": number,   // 0.0 to 1.0
  "notes": string|null
}}

Rules:
- Use ONLY provided snippets. Do NOT invent.
- If missing, use null or empty arrays.
- Prefer business emails (info@, sales@, contact@).
- best_source_url should point to the page where the best contact details are found.
- Output MUST be JSON only and match this top-level structure:
{{ "rows": [ ... ] }}

INPUT JSON:
{json.dumps(sites, ensure_ascii=False)}
""".strip()

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )

    raw = (resp.text or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"rows": []}

    rows = data.get("rows", [])
    if not isinstance(rows, list):
        return []

    def as_list(x):
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    cleaned = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        conf_raw = r.get("confidence")
        try:
            conf = float(conf_raw) if str(conf_raw).strip() else 0.0
        except Exception:
            conf = 0.0

        cleaned.append({
            "domain": r.get("domain"),
            "company_name": r.get("company_name") if isinstance(r.get("company_name"), (str, type(None))) else None,
            "emails": as_list(r.get("emails")),
            "phones": as_list(r.get("phones")),
            "address": r.get("address") if isinstance(r.get("address"), (str, type(None))) else None,
            "country": r.get("country") if isinstance(r.get("country"), (str, type(None))) else None,
            "best_source_url": r.get("best_source_url") if isinstance(r.get("best_source_url"), (str, type(None))) else None,
            "confidence": max(0.0, min(1.0, conf)),
            "notes": r.get("notes") if isinstance(r.get("notes"), (str, type(None))) else None,
        })

    return cleaned


# -----------------------------
# Main
# -----------------------------
def main():
    load_keys()
    query = input("Enter your business discovery query: ").strip()
    if not query:
        print("No query entered.")
        return

    session = requests.Session()

    # 1) Search
    links = serpapi_links(session, query, n=MAX_SERP_RESULTS)

    # 2) Dedupe by domain
    uniq = {}
    for u in links:
        d = dom(u)
        if d not in uniq:
            uniq[d] = u
    seeds = list(uniq.values())[:MAX_SERP_RESULTS]
    print(f"\nSERP links: {len(links)} | Unique domains: {len(seeds)}\n")

    # 3) Fetch snippets in parallel
    site_payloads: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_site_payload, session, u) for u in seeds]
        for i, fut in enumerate(as_completed(futures), start=1):
            payload = fut.result()
            if payload.get("pages"):
                site_payloads.append(payload)
            print(f"[fetch {i}/{len(seeds)}] {payload.get('domain')} pages={len(payload.get('pages', []))}")

    # 4) Enforce global input budget for one call
    site_payloads = trim_sites_to_budget(site_payloads, MAX_TOTAL_INPUT_CHARS)
    print(f"\nSites sent to Gemini (budgeted): {len(site_payloads)}\n")

    # 5) One Gemini call
    client = genai.Client()
    rows = gemini_one_call(client, site_payloads)

    # 6) Output final JSON
    out_file = "contacts_table.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"query": query, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_file} | Rows: {len(rows)}")


if __name__ == "__main__":
    main()
