import os
import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
print("key loaded:", bool(key), "| model:", model)

r = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": model, "messages": [{"role": "user", "content": "Say hello in 3 words"}]},
    timeout=30,
)
print(r.status_code)
print(r.text[:500])