import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    model: str = "gpt-4.1"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    api_version: Optional[str] = None
    temperature: float = 0.0

    # Azure specific
    use_azure: bool = field(
        default_factory=lambda: os.getenv("USE_AZURE_OPENAI", "False").lower() == "true"
    )
    azure_deployment: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    azure_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    azure_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_api_version: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Gemini specific
    use_gemini: bool = field(
        default_factory=lambda: os.getenv("USE_GEMINI", "False").lower() == "true"
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )

    # Azure specific
    use_azure: bool = field(
        default_factory=lambda: os.getenv("USE_AZURE_OPENAI", "False").lower() == "true"
    )
    azure_deployment: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    azure_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    azure_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_api_version: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Gemini specific
    use_gemini: bool = field(
        default_factory=lambda: os.getenv("USE_GEMINI", "False").lower() == "true"
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )


@dataclass
class PathRAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    working_dir: str = "./PathRAG_cache"

    # Vector DB / KV Storage
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"

    # Chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity Extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Node Embedding (Node2Vec)
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # Batching / Async
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16

    vector_db_storage_cls_kwargs: Dict[str, Any] = field(default_factory=dict)
    addon_params: Dict[str, Any] = field(default_factory=dict)

    enable_llm_cache: bool = True

    embedding_cache_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )

    log_level: str = "INFO"
