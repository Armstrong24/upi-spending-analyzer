"""Optional: ask an LLM for a short, friendly spending summary.

Only used when the user pastes an API key in the sidebar. Only aggregated
numbers are sent (no transaction descriptions), and the app works fully
offline without it.
"""
import json

import requests


def build_prompt(kpi: dict, categories: dict, budget_rows: list, recurring: list, flagged: int) -> str:
    facts = {"kpis": kpi, "spend_by_category_inr": categories, "budget_503020": budget_rows,
             "recurring_payments": recurring, "unusual_payments": flagged}
    return ("You are a personal finance coach for an Indian college student. Using only these numbers, "
            "write 5 short bullet points: what stands out, one risk, and three concrete saving actions. "
            "Use Rs for money.\n" + json.dumps(facts, default=str))


def ask_llm(prompt: str, api_key: str, model: str = "gpt-4o-mini",
            base_url: str = "https://api.openai.com/v1") -> str:
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.4}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
