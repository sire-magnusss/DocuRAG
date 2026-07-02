"""FastAPI server for LLMSearch."""

import os
import gc

import torch


# This is a temporary solution due to incompatimbility of ChromaDB with latest version of Protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from llmsearch.chroma import VectorStoreChroma
from functools import lru_cache
from typing import Any, List

import langchain
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from loguru import logger

# import llmsearch.database.crud as crud
from llmsearch.config import Config, ResponseModel, get_doc_with_model_config

# from llmsearch.database.config import get_local_session
from llmsearch.process import get_and_parse_response
from llmsearch.ranking import get_relevant_documents
from llmsearch.utils import LLMBundle, get_llm_bundle

from llmsearch.embeddings import (
    EmbeddingsHashNotExistError,
    update_embeddings,
)

# from sqlalchemy.orm import Session


load_dotenv()
langchain.debug = False


# ---------------------------------------------------------------------------
# Config / bundle loading
# ---------------------------------------------------------------------------

def read_config() -> Config:
    """Reads the configuration from environment variables and config files."""
    rag_config_file = os.environ["FASTAPI_RAG_CONFIG"]
    llm_config_file = os.environ["FASTAPI_LLM_CONFIG"]

    if not rag_config_file or not llm_config_file:
        raise SystemError(
            "Set 'FASTAPI_RAG_CONFIG' and 'FASTAPI_LLM_CONFIG' environment variable to point to a model config file."
        )

    logger.info(f"Loading configuration from {rag_config_file}")
    conf = get_doc_with_model_config(rag_config_file, llm_config_file)
    return conf


@lru_cache()
def get_config() -> Config:
    """Loads and caches the configuration."""
    return read_config()


@lru_cache()
def get_cached_llm_bundle() -> LLMBundle:
    """Loads and caches the LLM bundle."""
    config = get_config()
    logger.info("Loading LLM...")
    bundle = get_llm_bundle(config)
    return bundle


def get_llm_bundle_cached() -> LLMBundle:
    """Provides the cached LLM bundle."""
    return get_cached_llm_bundle()


# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Enforce API key auth when LLMSEARCH_API_KEY environment variable is set.

    If the variable is not set (or empty), authentication is disabled so
    existing deployments are unaffected. When the variable IS set every
    request must supply a matching ``X-Api-Key`` header.
    """
    expected_key = os.environ.get("LLMSEARCH_API_KEY", "")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Supply it via the X-Api-Key header.",
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api_app = FastAPI()
mcp = FastApiMCP(
    api_app,
    name="pyLLMSearch MCP Server",
    description="pyLLMSearch MCP Server",
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=["rag_retrieve_chunks", "rag_generate_answer", "rag_generate_answer_simple", "rag_update_index"],
)

mcp.mount()

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_cache_folder(cache_folder_root: str):
    """Set temporary cache folder for HF models and transformers"""

    sentence_transformers_home = cache_folder_root
    transformers_cache = os.path.join(cache_folder_root, "transformers")
    hf_home = os.path.join(cache_folder_root, "hf_home")

    logger.info(f"Setting SENTENCE_TRANSFORMERS_HOME folder: {sentence_transformers_home}")
    logger.info(f"Setting TRANSFORMERS_CACHE folder: {transformers_cache}")
    logger.info(f"Setting HF_HOME: {hf_home}")
    logger.info(f"Setting MODELS_CACHE_FOLDER: {cache_folder_root}")

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = sentence_transformers_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_HOME"] = hf_home
    os.environ["MODELS_CACHE_FOLDER"] = cache_folder_root


def unload_model(llm_bundle: LLMBundle):
    """Unloads llm_bundle from the state to free up the GPU memory"""

    llm_bundle.store = None  # type: ignore
    llm_bundle.chain = None  # type: ignore
    llm_bundle.reranker = None
    llm_bundle.hyde_chain = None
    llm_bundle.multiquery_chain = None

    get_cached_llm_bundle.cache_clear()
    gc.collect()

    llm_bundle = None  # type: ignore
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api_app.get("/")
def test():
    """Test endpoint to check if the API is running."""
    return {"message": "Welcome to LLMSearch API"}


@api_app.put("/update", operation_id="rag_update_index")
def update_index(
    llm_bundle: LLMBundle = Depends(get_llm_bundle_cached),
    config: Config = Depends(get_config),
    _auth: None = Depends(verify_api_key),
) -> Any:
    """Updates the index with the latest documents."""

    set_cache_folder(str(config.cache_folder))

    vs = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path), config=config)
    try:
        logger.debug("Updating embeddings")
        stats = update_embeddings(config, vs)
    except EmbeddingsHashNotExistError as exc:
        raise HTTPException(
            status_code=500,
            detail="Couldn't find hash files. Please re-create the index using current version of the app.",
        ) from exc
    else:
        return stats
    finally:
        logger.debug("Cleaning memory and re-Loading model...")

        vs.unload()

        vs = None  # type: ignore

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        unload_model(llm_bundle)


@api_app.get("/llm", response_model=ResponseModel, operation_id="rag_generate_answer")
async def llmsearch(
    question: str = Query(..., min_length=1, max_length=2000, description="The question to answer"),
    label: str = Query(default="", description="Optional document label to filter results"),
    llm_bundle: LLMBundle = Depends(get_llm_bundle_cached),
    _auth: None = Depends(verify_api_key),
) -> Any:
    """Retrieves answer to the question from the embedded documents, using semantic search."""
    if label and (label not in get_config().embeddings.labels):
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label}' doesn't exist. Use GET /labels to get a list of labels.",
        )

    output = get_and_parse_response(
        query=question,
        llm_bundle=llm_bundle,
        config=get_config(),
        label=label,
    )
    return output.model_dump()


@api_app.get("/rag_text_response", operation_id="rag_generate_answer_simple")
async def llmsearch_simple(
    question: str = Query(..., min_length=1, max_length=2000, description="The question to answer"),
    label: str = Query(default="", description="Optional document label to filter results"),
    llm_bundle: LLMBundle = Depends(get_llm_bundle_cached),
    _auth: None = Depends(verify_api_key),
) -> str:
    """Retrieves answer to the question from the embedded documents, using semantic search."""
    if label and (label not in get_config().embeddings.labels):
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label}' doesn't exist. Use GET /labels to get a list of labels.",
        )

    output = get_and_parse_response(
        query=question,
        llm_bundle=llm_bundle,
        config=get_config(),
        label=label,
    )
    return output.response


@api_app.get("/rag_chunks", operation_id="rag_retrieve_chunks")
async def semanticsearch(
    question: str = Query(..., min_length=1, max_length=2000, description="The question to retrieve relevant chunks for"),
    _auth: None = Depends(verify_api_key),
):
    """Retrieves chunks of information relevant to the question from the embedded documents, using semantic search."""
    docs = get_relevant_documents(
        original_query=question,
        queries=[question],
        llm_bundle=get_llm_bundle_cached(),
        config=get_config().semantic_search,
        label="",
    )
    return {"sources": docs}


@api_app.get("/labels")
async def labels(_auth: None = Depends(verify_api_key)) -> List[str]:
    """Returns a list of labels for the embeddings."""
    return get_config().embeddings.labels


def main():
    """Main function to run the FastAPI app."""
    uvicorn.run(api_app, host="0.0.0.0", port=8000)


# Refresh mcp server
mcp.setup_server()

if __name__ == "__main__":
    main()
