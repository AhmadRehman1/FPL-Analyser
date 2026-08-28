"""Thin client for a local Ollama server (https://ollama.com), used only by
scripts/narrate_ml_results.py to turn the Phase-0 ML experiment's real numeric results into a
plain-English draft paragraph.

Not a project dependency: Ollama itself is a separate local application (`ollama serve`), not a
pip package, and nothing in research/ml/experiment.py or the production pipeline imports this
module or calls a network endpoint on its own. `requests` is already a hard dependency
(requirements.txt), so no new package is needed to use this.

Deliberately narrow: one blocking, non-streaming chat call. No retry/backoff, no model
management (pulling/listing models) -- this is a local convenience tool run by hand, not a
production service with an SLA.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

DEFAULT_HOST = "http://localhost:11434"
# phi4-mini (Microsoft, 3.8B, ~2.5GB at Ollama's default Q4_K_M quantization) rather than a
# 7-8B model: this script's job is narrating numbers it is already given, not open-ended
# reasoning, so a small model is enough -- and it fully fits a 4GB-VRAM laptop GPU (e.g. an
# RTX 3050 Ti) with headroom for context, instead of spilling into slower CPU offload the way
# a Q4 7-8B model (~4.5-5GB) would on the same card. See research/ml/README.md's "Running
# locally with Ollama" section for the hardware reasoning and a larger-model alternative.
DEFAULT_MODEL = "phi4-mini"
DEFAULT_TIMEOUT_SECONDS = 120


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached or the request fails -- always with
    an actionable message (start the server / pull the model), never a raw traceback, since this
    runs interactively on a user's laptop."""


@dataclass
class OllamaResponse:
    text: str
    model: str


def is_reachable(host: str = DEFAULT_HOST, timeout_seconds: float = 3.0) -> bool:
    """True if a local Ollama server responds at all. Used to fail fast with a clear message
    before spending time building a prompt."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout_seconds)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = 0.2,
) -> OllamaResponse:
    """One non-streaming chat completion against a local Ollama server's native /api/chat
    endpoint. Low default temperature: this is narrating real numbers, not creative writing --
    the numbers themselves come from the prompt, verbatim, so the model's only job is prose."""
    if not is_reachable(host):
        raise OllamaUnavailableError(
            f"Cannot reach a local Ollama server at {host}. Start it with `ollama serve` "
            f"(or the Ollama desktop app), and make sure a model is pulled, e.g. "
            f"`ollama pull {model}`."
        )
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Ollama request failed ({type(exc).__name__}: {exc}). If the model isn't pulled "
            f"yet, run `ollama pull {model}` first."
        ) from exc
    payload = resp.json()
    message = payload.get("message", {})
    text = message.get("content", "")
    if not text:
        raise OllamaUnavailableError(
            f"Ollama returned an empty response for model {model!r} -- check `ollama list` "
            "shows it pulled and `ollama serve` is running the expected version."
        )
    return OllamaResponse(text=text, model=model)
