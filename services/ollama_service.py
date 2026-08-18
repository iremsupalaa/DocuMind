"""Yerel Ollama üzerinden embedding üretme işlemleri."""

import json
from typing import List
from urllib.request import Request, urlopen

from core.config import EMBEDDING_DIM, EMBEDDING_MODEL, OLLAMA_URL


def ollama_embeddings(texts: List[str]) -> List[List[float]]:
    """Metinler için pgvector ile uyumlu embedding vektörleri döndürür."""
    if not texts:
        return []

    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": texts,
        "dimensions": EMBEDDING_DIM,
        "keep_alive": "30m",
    }).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise ValueError("Embedding sayısı metin sayısıyla eşleşmedi.")
    if any(len(vector) != EMBEDDING_DIM for vector in embeddings):
        raise ValueError(f"Embedding boyutu {EMBEDDING_DIM} olmalıdır.")
    return embeddings
