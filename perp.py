import os
import json
from dotenv import load_dotenv
from perplexity import Perplexity
import re

load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    raise RuntimeError("Set PERPLEXITY_API_KEY in .env")

client = Perplexity(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert data extraction agent.

Task:
Given a user query about product/service vendors, do web search
and return ONLY valid JSON with a list of businesses and their contacts.

JSON schema:
{
  "vendors": [
    {
      "business_name": string,
      "website": string | null,
      "phone_numbers": string[],
      "emails": string[],
      "addresses": string[]
    }
  ]
}

Rules:
- Focus on companies, limited/LLC/corporations (not individuals).
- Only include vendors that actually provide the requested product/service.
- Phone numbers and emails must be valid and usable.
- Addresses should be as complete as possible (street, city, country).
- If something is unknown, use null or [].
- No extra keys, no comments, no explanation text. JSON ONLY.
"""

def find_vendors(user_query: str, max_vendors: int = 10):
    prompt = f"""
User query: {user_query}

Return up to {max_vendors} high-quality vendors that match this query,
using the JSON schema in the system instructions.
"""

    resp = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except Exception:

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("Model output is not valid JSON:\n" + raw)
        data = json.loads(m.group(0))

    return data

if __name__ == "__main__":
    while True:
        q = input("Describe the product/service vendors you need: ").strip()
        result = find_vendors(q, max_vendors=10)

        print(json.dumps(result, indent=2, ensure_ascii=False))

        print("\n=== VENDORS ===")
        for v in result.get("vendors", []):
            print("-----")
            print("Name:", v.get("business_name"))
            print("Website:", v.get("website"))
            print("Phones:", ", ".join(v.get("phone_numbers", [])))
            print("Emails:", ", ".join(v.get("emails", [])))
            print("Addresses:")
            for a in v.get("addresses", []):
                print("  -", a)
