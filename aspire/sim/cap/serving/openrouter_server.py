import asyncio
import contextlib
import itertools
import json
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

GEMINI_CONCURRENCY_LIMIT = 3

logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "openrouter/google/gemini-2.5-pro-preview"
    messages: list[Message]
    temperature: float | None = 0.2
    max_tokens: int | None = 256
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _load_api_keys(key_file: str) -> list[str]:
    """Load API keys from a protected file, falling back to the environment."""
    path = Path(key_file)
    if not path.exists():
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            return [api_key]
        raise FileNotFoundError(
            f"Key file not found: {key_file}. Set OPENROUTER_API_KEY or pass --key-file."
        )
    keys = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def create_app(api_keys: list[str], base_url: str, async_client: bool = True) -> FastAPI:
    default_headers = {
        "HTTP-Referer": "https://github.com/nvidia-gear/ASPIRE",
        "X-Title": "ASPIRE",
    }

    if async_client:
        clients = [AsyncOpenAI(api_key=k, base_url=base_url, default_headers=default_headers, timeout=250, max_retries=0) for k in api_keys]
    else:
        clients = [OpenAI(api_key=k, base_url=base_url, default_headers=default_headers, timeout=250, max_retries=0) for k in api_keys]
    _client_cycle = itertools.cycle(enumerate(clients, start=1))

    app = FastAPI(title="OpenRouter Proxy", version="1.0.0")

    if async_client:
        _gemini_sem: dict = {"sem": None}

        def _get_gemini_sem() -> asyncio.Semaphore:
            if _gemini_sem["sem"] is None:
                _gemini_sem["sem"] = asyncio.Semaphore(GEMINI_CONCURRENCY_LIMIT)
            return _gemini_sem["sem"]

        @app.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            key_idx = None
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                # Strip the "openrouter/" prefix if present so OpenRouter sees the
                # native model identifier (e.g. "google/gemini-2.5-pro-preview").
                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                is_gemini = "gemini" in client_kwargs.get("model", "").lower()
                gate = _get_gemini_sem() if is_gemini else contextlib.nullcontext()

                if request.stream:
                    client_kwargs["stream"] = True
                    key_idx, client = next(_client_cycle)
                    print(f"[key={key_idx}] model={client_kwargs.get('model', '?')}", flush=True)
                    async with gate:
                        response = await client.chat.completions.create(**client_kwargs)

                    async def event_stream():
                        async for chunk in response:
                            data = chunk.model_dump_json()
                            yield f"data: {data}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(event_stream(), media_type="text/event-stream")

                client_kwargs["stream"] = False
                key_idx, client = next(_client_cycle)
                print(f"[key={key_idx}] model={client_kwargs.get('model', '?')}", flush=True)
                async with gate:
                    response = await client.chat.completions.create(**client_kwargs)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except Exception as e:
                print(f"[ERROR] key={key_idx} model={request.model}: {e}", flush=True)
                raise HTTPException(status_code=500, detail=str(e))

    else:

        @app.post("/chat/completions", response_model=ChatCompletionResponse)
        def chat_completions(request: ChatCompletionRequest):
            key_idx = None
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                client_kwargs["stream"] = False

                key_idx, client = next(_client_cycle)
                print(f"[key={key_idx}] model={client_kwargs.get('model', '?')}", flush=True)
                response = client.chat.completions.create(**client_kwargs)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except Exception as e:
                print(f"[ERROR] key={key_idx} model={request.model}: {e}", flush=True)
                raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main(
    key_file: str = ".openrouterkey",
    host: str = "127.0.0.1",
    port: int = 8111,
    base_url: str = "https://openrouter.ai/api/v1/",
    async_client: bool = True,
):
    """
    Start the OpenRouter Proxy Server.

    Reads API keys from --key-file (one per line), or OPENROUTER_API_KEY when
    the file does not exist. The loopback-only default prevents other hosts
    from using the configured provider credentials.
    All keys are used in round-robin rotation across requests.
    """
    keys = _load_api_keys(key_file)
    print(f"Loaded {len(keys)} API key(s) (round-robin)", flush=True)

    app = create_app(api_keys=keys, base_url=base_url, async_client=async_client)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
