#!/usr/bin/env python3
"""
The Substitution v1.0.0 — API Server
Brandon Michael Jeanpierre Corporation d/b/a The Black Flag
Organization: theblackflagbmjc

FastAPI-based REST API server for The Substitution model.
Exposes inference endpoints for integration with enterprise systems.

Usage:
    # Start server
    python inference/api_server.py

    # Custom host/port
    python inference/api_server.py --host 0.0.0.0 --port 8080

    # With specific model
    python inference/api_server.py --model_path ./output/the-substitution-v1/merged_model

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible chat completion
    POST /v1/completions       — Raw text completion
    GET  /v1/models            — List available models
    GET  /health               — Health check
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api_server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api_server")

# --- System Prompt ---
DEFAULT_SYSTEM_PROMPT = (
    "You are The Substitution, an enterprise systems engineering assistant created by "
    "Brandon Michael Jeanpierre Corporation d/b/a The Black Flag. "
    "You are a precise, thorough technical advisor specializing in infrastructure architecture, "
    "systems administration, software engineering, networking, defensive and offensive cybersecurity, "
    "exploits, database operations, data analysis, enterprise documentation, and security governance."
)

# --- Globals ---
model = None
tokenizer = None
device = None
model_name = "the-substitution-v1.0.0"


# =============================================================================
# REQUEST / RESPONSE SCHEMAS (OpenAI-compatible)
# =============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="the-substitution", description="Model identifier")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Streaming not yet supported")
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)


class CompletionRequest(BaseModel):
    model: str = Field(default="the-substitution")
    prompt: str = Field(..., description="Input text to complete")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    uptime_seconds: float


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_inference_model(model_path: str = None, adapter_path: str = None):
    """Load model and tokenizer for serving."""
    global model, tokenizer, device

    config_path = PROJECT_ROOT / "training" / "training_config.yaml"
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
    base_model_name = train_config["model"]["base_model"]

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        logger.info("MPS (Apple Silicon) device")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("CPU device")

    if model_path and Path(model_path).exists():
        logger.info(f"Loading merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if device == "cuda" else {"": device},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if device == "cuda" else {"": device},
        )

        if adapter_path is None:
            adapter_path = str(
                PROJECT_ROOT / train_config["training"]["output_dir"] / "final_adapter"
            )

        if Path(adapter_path).exists():
            logger.info(f"Loading adapter: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Adapter loaded")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {total_params:,}. Device: {device}. Dtype: {dtype}")


# =============================================================================
# INFERENCE
# =============================================================================

def run_chat_inference(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run chat completion inference."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": request.repetition_penalty,
    }

    if request.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = request.temperature
        generation_kwargs["top_p"] = request.top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    response_ids = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    completion_tokens = len(response_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def run_completion_inference(request: CompletionRequest) -> CompletionResponse:
    """Run raw text completion inference."""
    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=3584)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if request.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = request.temperature
        generation_kwargs["top_p"] = request.top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    response_ids = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    completion_tokens = len(response_ids)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[
            CompletionChoice(
                index=0,
                text=response_text,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global startup_time
    startup_time = time.time()
    # Model loading is handled in main() before uvicorn starts
    logger.info("API server ready")
    yield
    logger.info("API server shutting down")


app = FastAPI(
    title="The Substitution API",
    description="Enterprise Systems Engineering LLM — Brandon Michael Jeanpierre Corporation d/b/a The Black Flag",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model=model_name,
        device=device or "unknown",
        uptime_seconds=round(time.time() - startup_time, 1),
    )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_name,
                created=int(startup_time),
                owned_by="theblackflagbmjc",
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return run_chat_inference(request)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Raw text completion endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return run_completion_inference(request)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="The Substitution API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model_path", type=str, default=None, help="Path to merged model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("The Substitution v1.0.0 — API Server")
    logger.info("Brandon Michael Jeanpierre Corporation d/b/a The Black Flag")
    logger.info("=" * 60)

    # Load model before starting server
    load_inference_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
