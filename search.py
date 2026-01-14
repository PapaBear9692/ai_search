import os
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse

import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from google import genai
from google.genai import types


SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

MAX_SERP_RESULTS = 20

# Only allow ABOUT/CONTACT-type pages
ABOUT_CONTACT_KEYWORDS = [
    "contact", "contact-us", "contactus", "get-in-touch", "getintouch",
    "about", "about-us", "aboutus", "company", "who-we-are", "whoweare",
    "team"
]

# Optional fallback guesses (still only about/contact)
FALLBACK_PATHS = [
    "/contact", "/contact-us", "/about", "/about-us", "/company", "/team"
]

MAX_FETCH_PAGES_PER_SITE = 2  # homepage + ONE extra about/contact page
MAX_HTML_BYTES = 1_500_000

# Keep prompt compact and high-signal
MAX_SNIPPET_CHARS_PER_PAGE = 12_000
MAX_TOTAL_SNIPPET_CHARS_PER_SITE = 28_000

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().\-]{6,}\d)")
ADDRESS_HINT_RE = re.compile(
    r"\b(address|office|head\s*office|location|registered|hq|headquarters|"
    r"street|st\.|road|rd\.|avenue|ave\.|suite|zip|postal|city|country|"
    r"building|floor|unit|block|district|province|state|"
    r"straße|strasse|weg|platz|allee|gasse|haus|plz|postfach)\b",
    re.IGNORECASE
)


def load_keys_from_env() -> None:
    load_dotenv()
    if not os.getenv("SERPAPI_API_KEY"):
        raise RuntimeError("Missing SERPAPI_API_KEY in .env")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")


def serpapi_search_top_links(query: str, num_results: int = 20) -> List[str]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": min(num_results, 20),
    }
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    organic = data.get("organic_results", []) or []
    links = []
    for item in organic:
        u = item.get("link")
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            links.append(u)

    # dedupe preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:num_results]


def registrable_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (urlparse(url).netloc or url).lower()


def normalize_root(url: str) -> str:
    p = urlparse(url)
    scheme = p.scheme or "https"
    netloc = p.netloc
    return f"{scheme}://{netloc}"


def fetch_html(url: str, timeout: int = 25) -> Optional[str]:
    try:
        with requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_HTML_BYTES:
                    break
                chunks.append(chunk)
            return b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        return None


def is_same_site(candidate_url: str, base_domain: str) -> bool:
    return registrable_domain(candidate_url) == base_domain


def discover_about_contact_link(base_url: str, html: str) -> Optional[str]:
    """
    From the homepage, find the BEST internal link that looks like About/Contact.
    Returns 1 URL (best) or None.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_domain = registrable_domain(base_url)

    scored: List[Tuple[int, str]] = []

    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        text = (a.get_text(" ", strip=True) or "").lower()
        href_low = href.lower()

        abs_url = urljoin(base_url, href)
        if not abs_url.startswith(("http://", "https://")):
            continue
        if not is_same_site(abs_url, base_domain):
            continue

        score = 0
        for kw in ABOUT_CONTACT_KEYWORDS:
            if kw in href_low:
                score += 3
            if kw in text:
                score += 2

        if score > 0:
            scored.append((score, abs_url))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate and return the top candidate
    seen: Set[str] = set()
    for _, u in scored:
        if u not in seen and u != base_url:
            return u
        seen.add(u)

    return None


def add_fallback_urls(root_url: str) -> List[str]:
    return [urljoin(root_url, p) for p in FALLBACK_PATHS]


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_section_text(soup: BeautifulSoup, selector: str, limit_chars: int) -> str:
    node = soup.select_one(selector)
    if not node:
        return ""
    for tag in node(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()
    return clean_text(node.get_text("\n"))[:limit_chars]


def high_signal_lines(all_text: str, window: int = 2, max_lines: int = 130) -> str:
    lines = [ln.strip() for ln in all_text.splitlines() if ln.strip()]
    keep = set()

    for i, ln in enumerate(lines):
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or ADDRESS_HINT_RE.search(ln):
            for j in range(max(0, i - window), min(len(lines), i + window + 1)):
                keep.add(j)

    picked = [lines[i] for i in sorted(keep)]
    if not picked:
        picked = lines[:50]
    return "\n".join(picked[:max_lines])


def extract_high_signal_snippet(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    header_text = extract_section_text(soup, "header", 6000)
    footer_text = extract_section_text(soup, "footer", 9000)

    # Global text but only keep signal lines
    full_text = clean_text(soup.get_text("\n"))
    signal_from_full = high_signal_lines(full_text, window=2, max_lines=160)

    parts = []
    if header_text:
        parts.append(f"--- HEADER ({page_url}) ---\n{header_text}")
    if footer_text:
        parts.append(f"--- FOOTER ({page_url}) ---\n{footer_text}")
    if signal_from_full:
        parts.append(f"--- SIGNAL LINES ({page_url}) ---\n{signal_from_full}")

    snippet = "\n\n".join(parts).strip()
    return snippet[:MAX_SNIPPET_CHARS_PER_PAGE]


def pick_one_extra_page(root: str, home_html: str) -> Optional[str]:
    """
    Choose ONE extra page:
    - best discovered about/contact link, else
    - first working fallback path
    """
    found = discover_about_contact_link(root, home_html)
    if found:
        return found

    # Try fallbacks quickly (we'll only actually fetch one that works later)
    for u in add_fallback_urls(root):
        if u != root:
            return u
    return None

def gemini_extract_one_company(
    client: genai.Client,
    domain: str,
    source_pages: List[Dict[str, str]],
    model: str = "gemini-3-flash-preview",
) -> Dict[str, Any]:
    """
    Runtime LLM call. We avoid response_schema because google-genai does not
    accept JSON Schema unions like ["string","null"].
    """
    prompt = f"""
You are extracting BUSINESS contact information for a company.

Domain: {domain}

You will receive snippets from the homepage and ONE extra page (about/contact).
Return ONE best record as STRICT JSON with exactly these keys:

{{
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
- Use ONLY the snippet text provided.
- Do NOT invent details.
- If missing, use null or empty arrays.
- Prefer business emails (info@, sales@, contact@).
- Output MUST be JSON only (no markdown, no extra text).

PAGES JSON:
{json.dumps(source_pages, ensure_ascii=False)}
""".strip()

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    raw = (resp.text or "").strip()

    # Parse JSON safely
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes models wrap JSON with extra text; try to extract first JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # Normalize + fill defaults
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    out = {
        "company_name": data.get("company_name") if isinstance(data.get("company_name"), (str, type(None))) else None,
        "emails": _as_list(data.get("emails")),
        "phones": _as_list(data.get("phones")),
        "address": data.get("address") if isinstance(data.get("address"), (str, type(None))) else None,
        "country": data.get("country") if isinstance(data.get("country"), (str, type(None))) else None,
        "best_source_url": data.get("best_source_url") if isinstance(data.get("best_source_url"), (str, type(None))) else None,
        "confidence": float(data.get("confidence")) if isinstance(data.get("confidence"), (int, float, str)) and str(data.get("confidence")).strip() != "" else 0.0,
        "notes": data.get("notes") if isinstance(data.get("notes"), (str, type(None))) else None,
    }

    # clamp confidence
    if out["confidence"] < 0.0:
        out["confidence"] = 0.0
    if out["confidence"] > 1.0:
        out["confidence"] = 1.0

    return out


def process_site_runtime(root: str, client: genai.Client) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
    - final_row: extracted company record (runtime LLM)
    - debug_site: debug payload containing fetched URLs/snippets
    """
    domain = registrable_domain(root)

    debug_site = {
        "domain": domain,
        "root": root,
        "status": None,
        "pages": [],   # [{"url":..., "snippet":...}]
        "chosen_extra_url": None,
        "errors": [],
    }

    home_html = fetch_html(root)
    time.sleep(0.6)
    if not home_html:
        debug_site["status"] = "home_fetch_failed"
        final_row = {
            "domain": domain,
            "company_name": None,
            "emails": [],
            "phones": [],
            "address": None,
            "country": None,
            "best_source_url": None,
            "confidence": 0.0,
            "notes": "Homepage fetch failed",
        }
        return final_row, debug_site

    home_snip = extract_high_signal_snippet(home_html, root)
    debug_site["pages"].append({"url": root, "snippet": home_snip})

    extra_url = pick_one_extra_page(root, home_html)
    debug_site["chosen_extra_url"] = extra_url

    source_pages = [{"url": root, "snippet": home_snip}]
    total_chars = len(home_snip)

    # Fetch only ONE extra page and only if it looks like about/contact
    if extra_url:
        extra_html = fetch_html(extra_url)
        time.sleep(0.6)
        if extra_html:
            extra_snip = extract_high_signal_snippet(extra_html, extra_url)
            # cap
            remaining = max(0, MAX_TOTAL_SNIPPET_CHARS_PER_SITE - total_chars)
            extra_snip = extra_snip[:remaining]
            debug_site["pages"].append({"url": extra_url, "snippet": extra_snip})
            source_pages.append({"url": extra_url, "snippet": extra_snip})
        else:
            debug_site["errors"].append("Extra page fetch failed")

    debug_site["status"] = "ok"

    # Runtime LLM call with snippets
    record = gemini_extract_one_company(
        client=client,
        domain=domain,
        source_pages=source_pages,
    )
    record["domain"] = domain
    return record, debug_site


def main():
    load_keys_from_env()

    query = input("Enter your business discovery query: ").strip()
    if not query:
        print("No query entered.")
        return

    # 1) Search
    links = serpapi_search_top_links(query, num_results=MAX_SERP_RESULTS)
    print(f"\nGot {len(links)} search results.\n")

    # 2) Deduplicate by domain so you don't hit the same site 5 times
    unique = {}
    for u in links:
        d = registrable_domain(u)
        if d not in unique:
            unique[d] = u
    seed_urls = list(unique.values())[:MAX_SERP_RESULTS]

    # 3) Runtime processing + LLM extraction
    client = genai.Client()

    results: List[Dict[str, Any]] = []
    debug_sites: List[Dict[str, Any]] = []

    for i, u in enumerate(seed_urls, start=1):
        root = normalize_root(u)
        print(f"[{i}/{len(seed_urls)}] Processing: {root}")

        final_row, debug_site = process_site_runtime(root, client=client)
        results.append(final_row)
        debug_sites.append(debug_site)

        # slow down a bit across domains
        time.sleep(0.8)

    # 4) Export DEBUG JSON (NOT used as context; just for inspection)
    debug_payload = {
        "query": query,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "sites": debug_sites
    }
    debug_file = f"debug_snippets_{int(time.time())}.json"
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_payload, f, ensure_ascii=False, indent=2)

    # 5) Export FINAL results JSON
    out_file = "contacts_table.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"query": query, "rows": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved FINAL JSON: {out_file}")
    print(f"Saved DEBUG JSON: {debug_file}")

    print("\n=== First 10 rows ===")
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
