import os
import requests

class OpenRouterClient:
    def __init__(self, model):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt, temperature=0.7):
        headers = { "Authorization": f"Bearer {self.api_key}" }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        resp = requests.post(self.url, json=payload, headers=headers)
        return resp.json()["choices"][0]["message"]["content"]
