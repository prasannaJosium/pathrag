import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast
import numpy as np

from .config import PathRAGConfig
from .llm_langchain import (
    get_langchain_llm,
    get_langchain_embeddings,
    langchain_llm_complete,
    langchain_embedding,
)

from .operate import (
    chunking_by_token_size,
    extract_entities,
    kg_query,
    find_and_link_disconnected_nodes,
)

from .utils import (
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
    wrap_embedding_func_with_attrs,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)

        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class PathRAG:
    config: PathRAGConfig = field(default_factory=PathRAGConfig)

    def __post_init__(self):
        # Setup working directory default if not customized but default value
        if self.config.working_dir == "./PathRAG_cache":
            self.config.working_dir = (
                f"./PathRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            )

        log_file = os.path.join("PathRAG.log")
        set_logger(log_file)
        logger.setLevel(self.config.log_level)

        logger.info(
            f"Logger initialized for working directory: {self.config.working_dir}"
        )

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.config.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.config.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.config.graph_storage
        ]

        if not os.path.exists(self.config.working_dir):
            logger.info(f"Creating working directory {self.config.working_dir}")
            os.makedirs(self.config.working_dir)

        # Initialize LangChain Models
        self.llm = get_langchain_llm(self.config.llm)
        self.embeddings = get_langchain_embeddings(self.config.embedding)

        # Create wrappers
        async def _embedding_func_wrapper(texts: list[str]) -> np.ndarray:
            return await langchain_embedding(self.embeddings, texts)

        wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.config.node2vec_params.get("dimensions", 1536),
            max_token_size=8191,
        )
        self.embedding_func = wrapper(_embedding_func_wrapper)

        self.llm_model_func = partial(langchain_llm_complete, self.llm)

        # Initial global config for response cache (before wrapping/limiting functions)
        global_config_clean = self.to_global_config()

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=global_config_clean,
                embedding_func=None,
            )
            if self.config.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(
            self.config.embedding_func_max_async
        )(self.embedding_func)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=global_config_clean,
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.config.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=global_config_clean,
                ),
                **self.config.llm.extra_kwargs,
            )
        )

        # We must ensure that when to_global_config is called later (e.g. in query),
        # it returns the wrapped llm_model_func.
        # Since to_global_config access self.llm_model_func dynamically, it will work.

    def to_global_config(self) -> dict:
        """Reconstructs the global configuration dictionary expected by storage and operation modules."""
        return {
            "working_dir": self.config.working_dir,
            "embedding_cache_config": self.config.embedding_cache_config,
            "kv_storage": self.config.kv_storage,
            "vector_storage": self.config.vector_storage,
            "graph_storage": self.config.graph_storage,
            "log_level": self.config.log_level,
            "chunk_token_size": self.config.chunk_token_size,
            "chunk_overlap_token_size": self.config.chunk_overlap_token_size,
            "tiktoken_model_name": self.config.tiktoken_model_name,
            "entity_extract_max_gleaning": self.config.entity_extract_max_gleaning,
            "entity_summary_to_max_tokens": self.config.entity_summary_to_max_tokens,
            "node_embedding_algorithm": self.config.node_embedding_algorithm,
            "node2vec_params": self.config.node2vec_params,
            "embedding_func": self.embedding_func,
            "embedding_batch_num": self.config.embedding_batch_num,
            "embedding_func_max_async": self.config.embedding_func_max_async,
            "llm_model_func": self.llm_model_func,
            "llm_model_name": self.config.llm.model,
            "llm_model_max_token_size": self.config.llm_model_max_token_size,
            "llm_model_max_async": self.config.llm_model_max_async,
            "llm_model_kwargs": self.config.llm.extra_kwargs,
            "vector_db_storage_cls_kwargs": self.config.vector_db_storage_cls_kwargs,
            "enable_llm_cache": self.config.enable_llm_cache,
            "addon_params": self.config.addon_params,
            "convert_response_to_json_func": convert_response_to_json,
        }

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
        }

    async def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return await loop.run_until_complete(await self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.config.chunk_overlap_token_size,
                        max_token_size=self.config.chunk_token_size,
                        tiktoken_model=self.config.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=self.to_global_config(),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)

            # Automatically link isolated nodes during ingestion
            await find_and_link_disconnected_nodes(
                self.chunk_entity_relation_graph,
                self.text_chunks,
                self.to_global_config(),
                self.relationships_vdb,
            )
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")

                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }

                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)

                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    async def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return await loop.run_until_complete(await self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode in ["hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self.to_global_config(),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=self.to_global_config(),
                ),
            )
            print("response all ready")
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_document(self, doc_id: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_document(doc_id))

    async def adelete_document(self, doc_id: str):
        from .prompt import GRAPH_FIELD_SEP

        chunks_to_delete = []
        if hasattr(self.text_chunks, "_data"):
            for chunk_id, chunk_data in self.text_chunks._data.items():
                if chunk_data.get("full_doc_id") == doc_id:
                    chunks_to_delete.append(chunk_id)
        else:
            all_keys = await self.text_chunks.all_keys()
            for chunk_id in all_keys:
                chunk_data = await self.text_chunks.get_by_id(chunk_id)
                if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                    chunks_to_delete.append(chunk_id)

        if not chunks_to_delete:
            logger.warning(f"No chunks found for doc {doc_id}")
            await self.full_docs.delete([doc_id])
            await self._insert_done()
            return

        logger.info(f"Deleting {len(chunks_to_delete)} chunks for doc {doc_id}")

        nodes_to_fully_delete = []
        edges_to_fully_delete = []
        nodes_to_update = []
        edges_to_update = []

        if hasattr(self.chunk_entity_relation_graph, "_graph"):
            graph = self.chunk_entity_relation_graph._graph
            for node, data in list(graph.nodes(data=True)):
                source_ids = data.get("source_id", "").split(GRAPH_FIELD_SEP)
                new_source_ids = [
                    sid for sid in source_ids if sid not in chunks_to_delete
                ]

                if not new_source_ids:
                    nodes_to_fully_delete.append(node)
                elif len(new_source_ids) < len(source_ids):
                    nodes_to_update.append((node, GRAPH_FIELD_SEP.join(new_source_ids)))

            for u, v, data in list(graph.edges(data=True)):
                source_ids = data.get("source_id", "").split(GRAPH_FIELD_SEP)
                new_source_ids = [
                    sid for sid in source_ids if sid not in chunks_to_delete
                ]

                if not new_source_ids:
                    edges_to_fully_delete.append((u, v))
                elif len(new_source_ids) < len(source_ids):
                    edges_to_update.append((u, v, GRAPH_FIELD_SEP.join(new_source_ids)))

        if nodes_to_fully_delete:
            entity_vdb_ids = [
                compute_mdhash_id(node, prefix="ent-") for node in nodes_to_fully_delete
            ]
            await self.entities_vdb.delete(entity_vdb_ids)
            for node in nodes_to_fully_delete:
                await self.chunk_entity_relation_graph.delete_node(node)

        if edges_to_fully_delete:
            rel_vdb_ids = [
                compute_mdhash_id(u + v, prefix="rel-")
                for u, v in edges_to_fully_delete
            ]
            await self.relationships_vdb.delete(rel_vdb_ids)
            if hasattr(self.chunk_entity_relation_graph, "_graph"):
                graph = self.chunk_entity_relation_graph._graph
                for u, v in edges_to_fully_delete:
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)

        for node, new_sid in nodes_to_update:
            if hasattr(self.chunk_entity_relation_graph, "_graph"):
                self.chunk_entity_relation_graph._graph.nodes[node]["source_id"] = (
                    new_sid
                )

        for u, v, new_sid in edges_to_update:
            if hasattr(self.chunk_entity_relation_graph, "_graph"):
                if self.chunk_entity_relation_graph._graph.has_edge(u, v):
                    self.chunk_entity_relation_graph._graph[u][v]["source_id"] = new_sid

        await self.chunks_vdb.delete(chunks_to_delete)
        await self.text_chunks.delete(chunks_to_delete)
        await self.full_docs.delete([doc_id])

        logger.info(
            f"Deleted doc {doc_id}: {len(chunks_to_delete)} chunks, {len(nodes_to_fully_delete)} nodes deleted, {len(edges_to_fully_delete)} edges deleted."
        )

        await self._insert_done()

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def retry_isolated_nodes(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aretry_isolated_nodes())

    async def aretry_isolated_nodes(self):
        await find_and_link_disconnected_nodes(
            self.chunk_entity_relation_graph,
            self.text_chunks,
            self.to_global_config(),
            self.relationships_vdb,
        )
        await self._insert_done()

    def retry_linking(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aretry_linking())

    async def aretry_linking(self):
        from .operate import find_and_link_disconnected_nodes

        await find_and_link_disconnected_nodes(
            self.chunk_entity_relation_graph,
            self.text_chunks,
            self.to_global_config(),
            self.relationships_vdb,
        )
        await self._insert_done()
