import os
import re
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from serpapi import GoogleSearch
from google import genai

# ---------- config & clients ----------

load_dotenv()  # loads .env from current directory [web:75][web:81]

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPAPI_API_KEY:
    raise RuntimeError("SERPAPI_API_KEY is not set in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)  # [web:79][web:82]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# ---------- helpers ----------

def web_search(query: str, num_results: int = 20):
    """Search the web using SerpAPI (Google engine) and return organic results."""  # [web:72][web:63]
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
        "location": "Dhaka, Bangladesh",
    }
    search = GoogleSearch(params)
    res = search.get_dict()
    return res.get("organic_results", [])


def fetch_html(url: str, timeout: int = 10) -> str | None:
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def get_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def guess_contact_urls(root_url: str, html: str, max_links: int = 3):
    """Find likely 'contact' page links from the homepage."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").lower()
        href = a["href"]
        if any(k in text for k in ["contact", "contact us", "about us", "support", "help"]):
            full = urljoin(root_url, href)
            candidates.append(full)
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


def extract_contacts_with_gemini(text: str):
    """Use Gemini to extract contact info as JSON-like structure."""
    # truncate to keep prompt manageable
    text = text[:12000]

    prompt = f"""
You are a data extraction assistant. Provide only relevant information according to user query.

From the following web page text, extract business contact information and return ONLY valid JSON.

JSON schema:
{{
  "company_name": string or null,
  "website": string or null,
  "emails": string[],
  "phones": string[],
  "addresses": string[]
}}

Guidelines:
- Use information that clearly refers to the company (not customers or partners).
- "emails" should contain only email addresses.
- "phones" should contain only phone numbers.
- "addresses" should be full postal addresses if possible.
- If you are unsure about a field, set it to null or [].

Text:
\"\"\"{text}\"\"\"
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",  # fast & cheap, adjust if needed [web:76][web:79]
        contents=prompt,
    )
    raw = response.text.strip()

    # Gemini will usually output valid JSON, but be defensive
    import json
    try:
        data = json.loads(raw)
    except Exception:
        # try to extract JSON block with a fallback
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {
                "company_name": None,
                "website": None,
                "emails": [],
                "phones": [],
                "addresses": [],
            }
        data = json.loads(match.group(0))

    # normalize
    def as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [str(v)]

    return {
        "company_name": data.get("company_name"),
        "website": data.get("website"),
        "emails": as_list(data.get("emails")),
        "phones": as_list(data.get("phones")),
        "addresses": as_list(data.get("addresses")),
    }


def merge_contacts(llm_data, text_emails, text_phones, fallback_website):
    emails = sorted(set(llm_data.get("emails", []) + text_emails))
    phones = sorted(set(llm_data.get("phones", []) + text_phones))
    website = llm_data.get("website") or fallback_website
    return {
        "company_name": llm_data.get("company_name"),
        "website": website,
        "emails": emails,
        "phones": phones,
        "addresses": llm_data.get("addresses", []),
    }


def normalize_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url.lower()


# ---------- main pipeline ----------

def find_vendors(query: str, per_query_results: int = 5):
    organic = web_search(query, per_query_results)
    vendors = []

    for result in organic:
        url = result.get("link")
        if not url:
            continue

        html = fetch_html(url)
        if not html:
            continue

        contact_urls = guess_contact_urls(url, html)

        best_contact = None
        for cu in contact_urls:
            ch = fetch_html(cu)
            if ch:
                best_contact = (cu, ch)
                break

        if not best_contact:
            best_contact = (url, html)

        contact_url, contact_html = best_contact
        text = get_text_from_html(contact_html)
        emails_rx, phones_rx = regex_contacts(text)
        llm_data = extract_contacts_with_gemini(text)
        merged = merge_contacts(llm_data, emails_rx, phones_rx, fallback_website=url)
        merged["source_url"] = contact_url
        vendors.append(merged)

    # deduplicate by domain
    dedup = {}
    for v in vendors:
        dom = normalize_domain(v.get("website") or v.get("source_url", ""))
        if dom and dom not in dedup:
            dedup[dom] = v

    return list(dedup.values())


if __name__ == "__main__":
    while True:
        q = input("Enter vendor/product query: ").strip()
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
