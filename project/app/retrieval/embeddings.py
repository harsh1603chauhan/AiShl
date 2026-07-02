from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32).tolist()


class OpenAIEmbedder:
    def __init__(self, model_name: str, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=[text])
        return response.data[0].embedding


class HashingEmbedder:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for index, token in enumerate(text.lower().split()):
            position = abs(hash(token)) % self.dimensions
            vector[position] += 1.0 / (index + 1)
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embeddings_provider == "openai" and settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_embeddings_model, settings.openai_api_key)
    if settings.embeddings_provider == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(settings.embeddings_model)
        except Exception:
            return HashingEmbedder()
    return HashingEmbedder()
