# chatgpt_like_rank_then_contact.py
# - Handles BOTH intents:
#   A) "top/best/leading/largest companies in X"  -> RANK THEN CONTACT (ChatGPT-like)
#      * Gemini call #1: extract top company names from list/report sources
#      * Then: find official sites + extract contact info deterministically (no extra LLM call)
#   B) "find contact of <company>" / narrow business discovery -> CONTACT ONLY
#      * No ranking; goes straight to official site + contact extraction (no LLM required)
#
# - SerpAPI for search
# - Fetch a few high-signal pages per company (homepage + contact/about/legal + optional /directory)
# - Extract contacts via:
#   * mailto: / tel:
#   * schema.org JSON-LD (Organization)
#   * signal lines near address/phone/email hints (footer/header included naturally)
#
# Requirements:
#   .env must include:
#     SERPAPI_API_KEY
#     GEMINI_API_KEY (or GOOGLE_API_KEY)

import os
import re
import json
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from start import genai
from google.genai import types


# -----------------------------
# CONFIG
# -----------------------------
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = (5, 12)                # (connect, read)
MAX_HTML_BYTES = 900_000         # cap per page
MAX_WORKERS = 14

# How many companies to output in rank mode
TOP_N = 10

# Rank-mode search (list/report sources)
RANK_SERP_PAGES = 3              # 0,10,20
RANK_SERP_PAGE_SIZE = 10
RANK_FETCH_PAGES = 8             # fetch this many list pages as evidence

# Contact-mode search (official site discovery)
OFFICIAL_SERP_PAGES = 2
OFFICIAL_SERP_PAGE_SIZE = 10

# Contact pages to try per official domain
MAX_PAGES_PER_COMPANY = 4        # homepage + best contact-like + optional legal + optional /directory
TRY_DIRECTORY = True

KEYWORDS = [
    "contact", "contact-us", "contactus",
    "about", "about-us", "aboutus",
    "imprint", "impressum", "legal", "privacy",
    "company", "team", "who-we-are",
    "support", "customer-service",
    "locations", "office", "find-us",
]
FALLBACK_PATHS = [
    "/contact", "/contact-us", "/about", "/about-us",
    "/imprint", "/impressum", "/legal", "/privacy",
    "/company", "/support"
]

# Basic filtering
BAD_DOMAINS_DEFAULT = {
    "facebook.com", "m.facebook.com", "linkedin.com", "twitter.com", "x.com",
    "youtube.com", "instagram.com", "tiktok.com",
    "alibaba.com", "aliexpress.com", "amazon.com", "ebay.com",
    "indiamart.com", "made-in-china.com", "globalsources.com",
    "tradeindia.com", "ecplaza.net", "exportersindia.com",
    "yellowpages.com", "yelp.com", "justdial.com",
    "mapquest.com",
}
# In RANK mode we ALLOW Wikipedia (helps a lot for “top companies” recall)
RANK_ALLOW_DOMAINS = {"wikipedia.org"}

BAD_EXTENSIONS = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")

# Extraction regex
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().\-]{6,}\d)")
ADDR_HINT_RE = re.compile(
    r"\b(address|office|location|registered|hq|headquarters|"
    r"street|road|avenue|suite|zip|postal|city|country|"
    r"building|floor|unit|block|district|province|state|"
    r"straße|strasse|weg|platz|allee|gasse|haus|plz)\b",
    re.IGNORECASE
)

# -----------------------------
# Utilities
# -----------------------------
def load_keys() -> None:
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
    scheme = p.scheme or "https"
    return f"{scheme}://{p.netloc}"


def normalize_url(u: Any) -> Optional[str]:
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not u.startswith(("http://", "https://")):
        return None
    return u


def is_bad_url(u: str, rank_mode: bool = False) -> bool:
    ul = u.lower()
    if ul.endswith(BAD_EXTENSIONS):
        return True

    d = dom(u)
    if rank_mode and d in RANK_ALLOW_DOMAINS:
        return False

    if d in BAD_DOMAINS_DEFAULT:
        return True

    noisy_bits = ["/login", "/signin", "/sign-in", "/watch", "/posts", "/share", "/photo", "/videos"]
    if any(b in ul for b in noisy_bits):
        return True

    return False


def add_directory_path(url: str) -> str:
    return urljoin(url if url.endswith("/") else (url + "/"), "directory")


def safe_dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -----------------------------
# Intent detection (no LLM needed)
# -----------------------------
RANK_WORDS = {
    "best", "top", "largest", "leading", "biggest", "rank", "ranking", "list",
    "companies", "company", "manufacturers", "suppliers", "brands"
}
def is_rank_query(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in RANK_WORDS)


# -----------------------------
# HTTP Fetch
# -----------------------------
def fetch_html(session: requests.Session, url: str) -> Optional[str]:
    try:
        with session.get(url, headers=HTTP_HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True) as r:
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


# -----------------------------
# SerpAPI helpers
# -----------------------------
def serpapi_search(session: requests.Session, q: str, num: int, start: int) -> List[Dict[str, Any]]:
    params = {
        "engine": "google",
        "q": q,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num,
        "start": start,
    }
    r = session.get(SERPAPI_ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in (data.get("organic_results") or []):
        link = normalize_url(item.get("link"))
        if not link:
            continue
        out.append({
            "link": link,
            "title": item.get("title") or "",
            "snippet": item.get("snippet") or ""
        })
    return out


def pick_rank_sources(results: List[Dict[str, Any]], want: int) -> List[str]:
    """
    Prefer list/report pages, allow Wikipedia, avoid social/marketplaces.
    """
    def score(r: Dict[str, Any]) -> int:
        u = (r["link"] or "").lower()
        t = (r.get("title") or "").lower()
        s = (r.get("snippet") or "").lower()
        sc = 0

        # good cues for ranking/list sources
        for kw in ["top", "best", "largest", "leading", "list", "ranking", "companies", "pharmaceutical", "industry"]:
            if kw in t or kw in s:
                sc += 2

        # wikipedia can be useful in rank mode
        if "wikipedia.org" in u:
            sc += 3

        # discourage obvious directory/listing spam
        for bad in ["yellowpages", "yelp", "justdial", "directory", "listing", "profile", "marketplace"]:
            if bad in u or bad in t or bad in s:
                sc -= 4

        # short paths can be official, but in rank mode we want list articles too
        if u.count("/") <= 3:
            sc += 1

        return sc

    ranked = sorted(results, key=score, reverse=True)
    urls = []
    for r in ranked:
        u = r["link"]
        if not is_bad_url(u, rank_mode=True):
            urls.append(u)
    return safe_dedupe_preserve_order(urls)[:want]


def pick_official_site_candidates(results: List[Dict[str, Any]], want: int) -> List[str]:
    """
    Prefer likely official websites (shorter URLs, "official" cues; avoid directories/social).
    Unique by domain.
    """
    def score(r: Dict[str, Any]) -> int:
        u = (r["link"] or "").lower()
        t = (r.get("title") or "").lower()
        s = (r.get("snippet") or "").lower()
        sc = 0

        for kw in ["official", "contact", "about", "imprint", "impressum"]:
            if kw in t or kw in s:
                sc += 3
        # discourage big platforms
        for bad in ["wikipedia", "linkedin", "facebook", "instagram", "youtube", "twitter", "x.com"]:
            if bad in u:
                sc -= 5

        # shorter paths tend to be homepages
        if u.count("/") <= 3:
            sc += 2
        return sc

    ranked = sorted(results, key=score, reverse=True)

    seen_domains = set()
    picked = []
    for r in ranked:
        u = r["link"]
        if is_bad_url(u, rank_mode=False):
            continue
        d = dom(u)
        if d in seen_domains:
            continue
        seen_domains.add(d)
        picked.append(u)
        if len(picked) >= want:
            break
    return picked


# -----------------------------
# Rank mode: fetch sources and ask Gemini for top company names
# -----------------------------
def html_to_readable_text(html: str, url: str, max_chars: int = 25_000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # keep first N lines; list pages usually have names early
    out = f"URL: {url}\n" + "\n".join(lines[:1200])
    return out[:max_chars]


def gemini_extract_top_companies(client: genai.Client, sources: List[str], country_hint: str, industry_hint: str, top_n: int) -> List[str]:
    """
    Gemini call #1 (only in rank mode): from list/report sources, output company names.
    """
    prompt = f"""
You are identifying the TOP companies from multiple web sources.

Task:
- Extract the top {top_n} company names for this query: "{industry_hint} companies in {country_hint}".
- Use ONLY the provided source text.
- Prefer widely-known/large/leading companies that are repeatedly mentioned across sources.
- Output JSON ONLY:
{{ "companies": [ "Company A", "Company B", ... ] }}

Rules:
- Do NOT invent names not present in sources.
- Deduplicate names (case/spacing).
- If sources conflict, prefer names that appear in multiple sources.
- Keep names as official as possible (e.g., "Square Pharmaceuticals Ltd." not "Square Pharma").
SOURCES:
{json.dumps(sources, ensure_ascii=False)}
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )

    raw = resp.text or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    companies = data.get("companies", [])
    if not isinstance(companies, list):
        return []
    cleaned = []
    for c in companies:
        if isinstance(c, str):
            c2 = c.strip()
            if c2:
                cleaned.append(c2)
    return cleaned[:top_n]


def infer_country_and_industry(user_query: str) -> Tuple[str, str]:
    """
    Lightweight heuristic:
    - country: last token-ish after 'in <country>' if present
    - industry: rough noun phrase (fallback: 'companies')
    """
    ql = user_query.lower()
    country = ""
    industry = "companies"

    # try "in <country>"
    m = re.search(r"\bin\s+([a-z\s]+)$", ql)
    if m:
        country = m.group(1).strip().title()

    # industry: pull a keyword if present
    for k in ["pharmaceutical", "pharma", "garments", "software", "bank", "telecom", "cement", "steel"]:
        if k in ql:
            industry = "pharmaceutical" if k in ["pharmaceutical", "pharma"] else k
            break

    if not country:
        country = "the specified country"
    return country, industry


# -----------------------------
# Contact extraction: evidence from a few pages, deterministic parsing
# -----------------------------
def extract_mailto_tel(soup: BeautifulSoup) -> Tuple[Set[str], Set[str]]:
    emails, phones = set(), set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        hl = href.lower()
        if hl.startswith("mailto:"):
            e = href.split(":", 1)[1].split("?")[0].strip()
            if e:
                emails.add(e)
        elif hl.startswith("tel:"):
            p = href.split(":", 1)[1].strip()
            if p:
                phones.add(p)
    return emails, phones


def parse_jsonld_for_org(raw_html: str) -> Dict[str, Any]:
    """
    Parse schema.org JSON-LD blocks. Return best organization-ish fields found.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    best = {"name": None, "emails": set(), "phones": set(), "address": None}

    for tag in soup.find_all("script"):
        t = (tag.get("type") or "").lower()
        if "ld+json" not in t:
            continue
        content = (tag.string or tag.get_text() or "").strip()
        if not content:
            continue

        # Sometimes multiple JSON objects / invalid JSON. Try best effort.
        try:
            data = json.loads(content)
        except Exception:
            continue

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                typ = obj.get("@type") or obj.get("type")
                if isinstance(typ, list):
                    typ = " ".join([str(x) for x in typ])
                typ_s = str(typ).lower() if typ else ""

                is_org = any(x in typ_s for x in ["organization", "corporation", "localbusiness", "pharmacy", "company"])
                if is_org:
                    nm = obj.get("name")
                    if isinstance(nm, str) and nm.strip() and not best["name"]:
                        best["name"] = nm.strip()

                    em = obj.get("email")
                    if isinstance(em, str) and em.strip():
                        best["emails"].add(em.strip())
                    if isinstance(em, list):
                        for e in em:
                            if isinstance(e, str) and e.strip():
                                best["emails"].add(e.strip())

                    tel = obj.get("telephone") or obj.get("phone")
                    if isinstance(tel, str) and tel.strip():
                        best["phones"].add(tel.strip())
                    if isinstance(tel, list):
                        for p in tel:
                            if isinstance(p, str) and p.strip():
                                best["phones"].add(p.strip())

                    addr = obj.get("address")
                    addr_str = address_to_string(addr)
                    if addr_str and not best["address"]:
                        best["address"] = addr_str

                # recurse
                for v in obj.values():
                    walk(v)

            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

    return {
        "name": best["name"],
        "emails": best["emails"],
        "phones": best["phones"],
        "address": best["address"],
    }


def address_to_string(addr: Any) -> Optional[str]:
    if isinstance(addr, str):
        a = addr.strip()
        return a if a else None
    if isinstance(addr, dict):
        parts = []
        for k in ["streetAddress", "postOfficeBoxNumber", "addressLocality", "addressRegion", "postalCode", "addressCountry"]:
            v = addr.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        joined = ", ".join(parts).strip()
        return joined if joined else None
    if isinstance(addr, list):
        # take first good
        for it in addr:
            s = address_to_string(it)
            if s:
                return s
    return None


def get_title_or_og_name(raw_html: str) -> Optional[str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)[:120]
    og = soup.find("meta", attrs={"property": "og:site_name"})
    if og and og.get("content"):
        return str(og["content"]).strip()[:120]
    return None


def signal_address_from_text(raw_html: str) -> Optional[str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    # pick a small window around address hints
    for i, ln in enumerate(lines):
        if ADDR_HINT_RE.search(ln):
            window = lines[i:i+4]
            joined = " | ".join(window).strip()
            # avoid huge
            return joined[:220]
    return None


def best_contact_like_url(base: str, html: str) -> Optional[str]:
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
        if dom(u) != base_dom:
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

    for p in FALLBACK_PATHS:
        u = urljoin(base, p)
        if dom(u) == base_dom:
            return u
    return None


def collect_company_pages(session: requests.Session, official_home: str) -> List[Tuple[str, str]]:
    """
    Return list of (url, html) pages to mine.
    """
    pages: List[Tuple[str, str]] = []
    base = root(official_home)

    # homepage
    home_html = fetch_html(session, base)
    if home_html:
        pages.append((base, home_html))
    else:
        return pages

    # /directory (optional)
    if TRY_DIRECTORY and len(pages) < MAX_PAGES_PER_COMPANY:
        dir_url = add_directory_path(base)
        dir_html = fetch_html(session, dir_url)
        if dir_html:
            pages.append((dir_url, dir_html))

    # best contact-like from homepage
    if len(pages) < MAX_PAGES_PER_COMPANY:
        extra = best_contact_like_url(base, home_html)
        if extra:
            extra_html = fetch_html(session, extra)
            if extra_html:
                pages.append((extra, extra_html))

    # one more fallback path if still small
    if len(pages) < MAX_PAGES_PER_COMPANY:
        for p in FALLBACK_PATHS:
            u = urljoin(base, p)
            if all(u != x[0] for x in pages):
                h = fetch_html(session, u)
                if h:
                    pages.append((u, h))
                    break

    return pages[:MAX_PAGES_PER_COMPANY]


def extract_contacts_from_pages(pages: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Deterministic extraction across pages, pick best source.
    """
    all_emails: Set[str] = set()
    all_phones: Set[str] = set()
    company_name: Optional[str] = None
    address: Optional[str] = None

    best_source_url: Optional[str] = None
    best_score = -1
    notes = []

    for url, html in pages:
        # HTML parsing
        soup = BeautifulSoup(html, "html.parser")

        # mailto/tel
        emails_mt, phones_tel = extract_mailto_tel(soup)

        # regex find in raw text too (sometimes no mailto)
        text = soup.get_text("\n")
        emails_rx = set(EMAIL_RE.findall(text))
        phones_rx = set(PHONE_RE.findall(text))

        # schema.org
        org = parse_jsonld_for_org(html)

        # page-level aggregation
        page_emails = set()
        page_phones = set()
        page_emails |= emails_mt | emails_rx | set(org["emails"])
        page_phones |= phones_tel | phones_rx | set(org["phones"])

        # clean obvious junk phones (very short)
        page_phones = {p.strip() for p in page_phones if isinstance(p, str) and len(p.strip()) >= 7}
        page_emails = {e.strip() for e in page_emails if isinstance(e, str) and "@" in e}

        # address signals
        page_addr = org.get("address") or signal_address_from_text(html)

        # name signals
        page_name = org.get("name") or get_title_or_og_name(html)

        # score this page as a source
        score = 0
        if page_emails: score += 6
        if page_phones: score += 4
        if page_addr: score += 2
        if page_name: score += 1

        if score > best_score:
            best_score = score
            best_source_url = url

        # global agg
        all_emails |= page_emails
        all_phones |= page_phones
        if page_addr and not address:
            address = page_addr
        if page_name and (not company_name):
            company_name = page_name

    # confidence
    if all_emails and all_phones:
        confidence = "high"
    elif all_emails or all_phones:
        confidence = "medium"
    else:
        confidence = "low"
        notes.append("No email/phone found; site may be JS-rendered or hiding contact info.")

    # lightly prefer business-y emails if many
    preferred_emails = sorted(all_emails, key=lambda e: (0 if any(e.lower().startswith(p) for p in ("info@", "sales@", "contact@", "support@")) else 1, e.lower()))
    preferred_phones = sorted(all_phones)

    # normalize company name if it's just a generic title
    if company_name:
        # trim common separators
        company_name = company_name.split("|")[0].split("–")[0].split("-")[0].strip() or company_name

    return {
        "company_name": company_name,
        "emails": preferred_emails,
        "phones": preferred_phones,
        "address": address,
        "best_source_url": best_source_url,
        "confidence": confidence,
        "notes": " ".join(notes) if notes else "",
    }


# -----------------------------
# Official site discovery per company name
# -----------------------------
def find_official_site(session: requests.Session, company_name: str, country_hint: str) -> Optional[str]:
    """
    Use SerpAPI to find official site candidates for a company name.
    """
    q_variants = [
        f'"{company_name}" official website {country_hint}',
        f'"{company_name}" contact {country_hint}',
        f'{company_name} {country_hint} official site',
    ]

    gathered: List[Dict[str, Any]] = []
    for q in q_variants:
        for page in range(OFFICIAL_SERP_PAGES):
            start = page * OFFICIAL_SERP_PAGE_SIZE
            try:
                gathered.extend(serpapi_search(session, q, OFFICIAL_SERP_PAGE_SIZE, start))
            except Exception:
                continue

    candidates = pick_official_site_candidates(gathered, want=5)
    return candidates[0] if candidates else None


# -----------------------------
# Printing
# -----------------------------
def print_rank_list(companies: List[str]) -> None:
    print("\n========== TOP COMPANIES ==========\n")
    for i, c in enumerate(companies, 1):
        print(f"{i}. {c}")
    print("-" * 70)


def print_contact_rows(rows: List[Dict[str, Any]]) -> None:
    print("\n========== CONTACT RESULTS ==========\n")
    for i, r in enumerate(rows, 1):
        print(f"{i}. {r.get('company_name') or 'Unknown'}")
        print(f"   Website     : {r.get('website')}")
        print(f"   Emails      : {r.get('emails')}")
        print(f"   Phones      : {r.get('phones')}")
        print(f"   Address     : {r.get('address')}")
        print(f"   Best Source : {r.get('best_source_url')}")
        print(f"   Confidence  : {r.get('confidence')}")
        if r.get("notes"):
            print(f"   Notes       : {r.get('notes')}")
        print("-" * 70)


# -----------------------------
# MAIN
# -----------------------------
def main():
    load_keys()
    user_query = input("Enter your query: ").strip()
    if not user_query:
        return

    session = requests.Session()
    client = genai.Client()

    rank_mode = is_rank_query(user_query)
    country_hint, industry_hint = infer_country_and_industry(user_query)

    if rank_mode:
        # -------------------------
        # RANK THEN CONTACT (ChatGPT-like)
        # -------------------------
        # 1) Find ranking/list sources
        rank_queries = [
            user_query,
            f"top {industry_hint} companies in {country_hint}",
            f"largest {industry_hint} companies in {country_hint} list",
            f"leading {industry_hint} manufacturers in {country_hint}",
            f"{industry_hint} industry {country_hint} top companies",
        ]

        all_rank_results: List[Dict[str, Any]] = []
        for q in rank_queries:
            for page in range(RANK_SERP_PAGES):
                start = page * RANK_SERP_PAGE_SIZE
                try:
                    all_rank_results.extend(serpapi_search(session, q, RANK_SERP_PAGE_SIZE, start))
                except Exception:
                    continue

        rank_sources_urls = pick_rank_sources(all_rank_results, want=RANK_FETCH_PAGES)
        if not rank_sources_urls:
            print("\nNo rank sources found. Try a broader query.\n")
            return

        print(f"\n[rank] Selected {len(rank_sources_urls)} ranking sources.\n")

        # 2) Fetch ranking sources and convert to readable text
        texts: List[str] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_html, session, u) for u in rank_sources_urls]
            for idx, fut in enumerate(as_completed(futures), start=1):
                html = fut.result()
                u = rank_sources_urls[idx - 1] if idx - 1 < len(rank_sources_urls) else ""
                if html:
                    texts.append(html_to_readable_text(html, u))
                    print(f"[rank fetch {idx}/{len(rank_sources_urls)}] OK")
                else:
                    print(f"[rank fetch {idx}/{len(rank_sources_urls)}] FAIL")

        if not texts:
            print("\nCould not fetch any ranking source pages.\n")
            return

        # 3) Gemini call #1: extract top company names
        companies = gemini_extract_top_companies(
            client=client,
            sources=texts,
            country_hint=country_hint,
            industry_hint=industry_hint,
            top_n=TOP_N
        )

        if not companies:
            print("\nGemini couldn't extract company names from the sources. Try adjusting the query.\n")
            return

        print_rank_list(companies)

        # 4) For each company: find official site, fetch pages, extract contacts deterministically
        rows: List[Dict[str, Any]] = []

        def company_worker(name: str) -> Dict[str, Any]:
            site = find_official_site(session, name, country_hint=country_hint)
            if not site:
                return {
                    "company_name": name,
                    "website": None,
                    "emails": [],
                    "phones": [],
                    "address": None,
                    "best_source_url": None,
                    "confidence": "low",
                    "notes": "Could not confidently find an official website via search.",
                }
            pages = collect_company_pages(session, site)
            extracted = extract_contacts_from_pages(pages) if pages else {
                "company_name": name, "emails": [], "phones": [], "address": None,
                "best_source_url": None, "confidence": "low", "notes": "Could not fetch pages from official site."
            }
            # Prefer the company name we started with if the page title is generic
            if extracted.get("company_name") and len(extracted["company_name"]) < 3:
                extracted["company_name"] = name
            if not extracted.get("company_name"):
                extracted["company_name"] = name

            extracted["website"] = root(site)
            return extracted

        print("\n[contact] Finding official sites + extracting contacts...\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(company_worker, c) for c in companies]
            for idx, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                print(f"[contact {idx}/{len(companies)}] {r.get('company_name')} -> {r.get('confidence')}")
                rows.append(r)

        # 5) Print contacts
        # keep same order as companies
        name_to_row = {r["company_name"]: r for r in rows}
        ordered_rows = []
        for c in companies:
            # row might have slightly different name from extraction; fallback by prefix match
            exact = None
            for r in rows:
                if r.get("company_name") == c:
                    exact = r
                    break
            ordered_rows.append(exact if exact else next((r for r in rows if (r.get("company_name") or "").lower().startswith(c.lower()[:8])), name_to_row.get(c, {"company_name": c})))

        print_contact_rows(ordered_rows)
        return

    # -------------------------
    # CONTACT ONLY
    # -------------------------
    # If the user didn't ask for a "top/best" list, treat it as targeted discovery.
    # We'll try to find 20 candidate official domains and extract contacts from each.
    print("\n[mode] CONTACT ONLY\n")

    # Build search queries to find official sites
    negative = (
        "-linkedin -facebook -instagram -youtube -alibaba -indiamart "
        "-made-in-china -globalsources -yellowpages -yelp -justdial -pdf"
    )
    q_variants = [
        f"{user_query} official website {negative}",
        f"{user_query} contact email phone address {negative}",
        f"{user_query} headquarters address phone {negative}",
    ]

    all_results: List[Dict[str, Any]] = []
    for q in q_variants:
        for page in range(OFFICIAL_SERP_PAGES):
            start = page * OFFICIAL_SERP_PAGE_SIZE
            try:
                all_results.extend(serpapi_search(session, q, OFFICIAL_SERP_PAGE_SIZE, start))
            except Exception:
                continue

    seeds = pick_official_site_candidates(all_results, want=WANT_SITES)
    print(f"\nSelected {len(seeds)} candidate sites.\n")
    if not seeds:
        print("No sites found. Try refining your query.")
        return

    rows: List[Dict[str, Any]] = []

    def seed_worker(seed: str) -> Dict[str, Any]:
        pages = collect_company_pages(session, seed)
        extracted = extract_contacts_from_pages(pages) if pages else {
            "company_name": None, "emails": [], "phones": [], "address": None,
            "best_source_url": None, "confidence": "low", "notes": "Could not fetch pages."
        }
        extracted["website"] = root(seed)
        return extracted

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(seed_worker, s) for s in seeds]
        for idx, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            print(f"[site {idx}/{len(seeds)}] {r.get('website')} -> {r.get('confidence')}")
            rows.append(r)

    print_contact_rows(rows)


if __name__ == "__main__":
    main()
