"""Small, explicit HTTP client for Azure OpenAI deployments."""

import httpx


class AzureOpenAIClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        embedding_deployment: str,
        chat_deployment: str,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._embedding_deployment = embedding_deployment
        self._chat_deployment = chat_deployment

    async def embed(self, text: str) -> list[float]:
        response = await self._request(
            f"/openai/deployments/{self._embedding_deployment}/embeddings",
            {"input": text},
            timeout=90,
        )
        data = response.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0].get("embedding"), list):
            raise RuntimeError("Azure OpenAI returned an invalid embedding response.")
        return data[0]["embedding"]

    async def generate(self, prompt: str) -> str:
        response = await self._request(
            f"/openai/deployments/{self._chat_deployment}/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": "You are a careful incident commander. Use only supplied evidence."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        choices = response.get("choices")
        recommendation = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(recommendation, str) or not recommendation.strip():
            raise RuntimeError("Azure OpenAI returned an empty recommendation.")
        return recommendation.strip()

    async def _request(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self._endpoint, timeout=timeout) as client:
                response = await client.post(
                    path,
                    params={"api-version": self._api_version},
                    headers={"api-key": self._api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(f"Azure OpenAI request failed: {error}") from error
