import logging
import os


from nv_ingest_client.util.vdb.adt_vdb import VDB
from datetime import timedelta
from functools import partial
from urllib.parse import urlparse
from nv_ingest_client.util.transport import infer_microservice
from nv_ingest_client.util.vdb.milvus import nv_rerank
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


def create_lancedb_results(results):
    """Transform NV-Ingest pipeline results into LanceDB ingestible rows.

    The NV-Ingest pipeline provides nested lists of record dictionaries. This
    helper extracts the inner `metadata` dict for each record, filters out
    entries without an embedding, and returns a list of dictionaries with the
    exact fields expected by the LanceDB table schema used in
    `LanceDB.create_index`.

    Parameters
    ----------
    results : list
        Nested list-of-lists containing record dicts in the NV-Ingest format.

    Returns
    -------
    list
        List of dictionaries with keys: `vector` (embedding list), `text`
        (string content), `metadata` (page number) and `source` (source id).

    Notes
    -----
    - The function expects each inner record to have a `metadata` mapping
        containing `embedding`, `content`, `content_metadata.page_number`, and
        `source_metadata.source_id`.
    - Records with `embedding is None` are skipped.
    """
    old_results = [res["metadata"] for result in results for res in result]
    lancedb_rows = []
    for result in old_results:
        if result["embedding"] is None:
            continue
        lancedb_rows.append(
            {
                "vector": result["embedding"],
                "text": result["content"],
                "metadata": result["content_metadata"]["page_number"],
                "source": result["source_metadata"]["source_id"],
            }
        )
    return lancedb_rows


class LanceDB(VDB):
    """LanceDB operator implementing the VDB interface.

    This class adapts NV-Ingest records to LanceDB, providing index creation,
    ingestion, and retrieval hooks. The implementation is intentionally small
    and focuses on the example configuration used in NV-Ingest evaluation
    scripts.
    """

    def __init__(
        self,
        uri=None,
        overwrite=True,
        table_name="nv-ingest",
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        **kwargs,
    ):
        """Initialize the LanceDB VDB operator.

        Parameters
        ----------
        uri: str, optional
            LanceDB connection URI (default is "lancedb" for local file-based
            storage).
        overwrite : bool, optional
            If True, existing tables will be overwritten during index creation.
            If False, new data will be appended to existing tables.
        table_name : str, optional
            Name of the LanceDB table to create/use (default is "nv-ingest").
        index_type : str, optional
            Type of vector index to create (default is "IVF_HNSW_SQ").
        metric : str, optional
            Distance metric for the vector index (default is "l2").
        num_partitions : int, optional
            Number of partitions for the vector index (default is 16).
        num_sub_vectors : int, optional
            Number of sub-vectors for the vector index (default is 256).
        **kwargs : dict
            Forwarded configuration options. This implementation does not
            actively consume specific keys, but passing parameters such as
            `uri`, `index_name`, or security options is supported by the
            interface pattern and may be used by future enhancements.
        """
        self.uri = uri or "lancedb"
        self.overwrite = overwrite
        self.table_name = table_name
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        super().__init__(**kwargs)

    def create_index(self, records=None, table_name="nv-ingest", **kwargs):
        """Create a LanceDB table and populate it with transformed records.

        This method connects to LanceDB, transforms NV-Ingest records using
        `create_lancedb_results`, builds a PyArrow schema that matches the
        expected table layout, and creates/overwrites a table named `bo`.

        Parameters
        ----------
        records : list, optional
            NV-Ingest records in nested list format (the same structure passed
            to `run`). If ``None``, an empty table will be created.

        table_name : str, optional
            Name of the LanceDB table to create (default is "nv-ingest").

        Returns
        -------
        table
            The LanceDB table object returned by `db.create_table`.
        """
        db = lancedb.connect(uri=self.uri)
        results = create_lancedb_results(records)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), 2048)),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("source", pa.string()),
            ]
        )
        table = db.create_table(
            table_name, data=results, schema=schema, mode="overwrite" if self.overwrite else "append"
        )
        return table

    def write_to_index(
        self,
        records,
        table=None,
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        **kwargs,
    ):
        """Create an index on the LanceDB table and wait for it to become ready.

        This function calls `table.create_index` with an IVF+HNSW+SQ index
        configuration used in NV-Ingest benchmarks. After requesting index
        construction it lists available indices and waits for each one to
        reach a ready state using `table.wait_for_index`.

        Parameters
        ----------
        records : list
            The original records being indexed (not used directly in this
            implementation but kept in the signature for consistency).
        table : object
            LanceDB table object returned by `create_index`.
        """
        table.create_index(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            # accelerator="cuda",
            vector_column_name="vector",
        )
        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

    # TODO: make this static, so we can call lancedb.retrieval
    # instead of calling for a table object, we can take the example from
    # lancedb_retrieval right now and decouple the table object from the
    # retrieval method

    # TODO: create example notebook for lancedb just like building_vdb_operator.ipynb and opensearch.ipynb

    # make sure send queries in as a list

    # TODO: add filtering and hybrid search to lancedb
    def retrieval(
        self,
        queries,
        table=None,
        embedding_endpoint="http://localhost:8012/v1",
        nvidia_api_key=None,
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        result_fields=["text", "metadata", "source"],
        top_k=10,
        **kwargs,
    ):
        """Run similarity search for a list of text queries.

        This method converts textual queries to embeddings by calling the
        transport helper `infer_microservice` (configured to use an NVIDIA
        embedding model in the example) and performs a vector search against
        the LanceDB `table`.

        Parameters
        ----------
        queries : list[str]
            Text queries to be embedded and searched.
        table : object
            LanceDB table object with a built vector index.
        embedding_endpoint : str, optional
            URL of the embedding microservice (default is
            "http://localhost:8012/v1").
        nvidia_api_key : str, optional
            NVIDIA API key for authentication with the embedding service. If
            ``None``, no authentication is used.
        model_name : str, optional
            Name of the embedding model to use (default is
            "nvidia/llama-3.2-nv-embedqa-1b-v2").
        result_fields : list, optional
            List of field names to retrieve from each hit document (default is
            `["text", "metadata", "source"]`).
        top_k : int, optional
            Number of top results to return per query (default is 10).

        Returns
        -------
        list[list[dict]]
            For each input query, a list of hit documents (each document is a
            dict with fields such as `text`, `metadata`, and `source`). The
            example limits each query to 20 results.
        """
        embed_model = partial(
            infer_microservice,
            model_name=model_name,
            embedding_endpoint=embedding_endpoint,
            nvidia_api_key=nvidia_api_key,
            input_type="query",
            output_names=["embeddings"],
            grpc=not ("http" in urlparse(embedding_endpoint).scheme),
        )
        results = []
        query_embeddings = embed_model(queries)
        for query_embed in query_embeddings:
            results.append(
                table.search([query_embed], vector_column_name="vector")
                .disable_scoring_autoprojection()
                .select(result_fields + ["_distance"])
                .limit(top_k)
                .to_list()
            )
        return results

    def run(self, records):
        """Orchestrate index creation and data ingestion.

        The `run` method is the public entry point used by NV-Ingest pipeline
        tasks. A minimal implementation first ensures the table exists by
        calling `create_index` and then kicks off index construction with
        `write_to_index`.

        Parameters
        ----------
        records : list
            NV-Ingest records to index.

        Returns
        -------
        list
            The original `records` list is returned unchanged to make the
            operator composable in pipelines.
        """
        table = self.create_index(records=records, table_name=self.table_name)
        self.write_to_index(
            records,
            table=table,
            index_type=self.index_type,
            metric=self.metric,
            num_partitions=self.num_partitions,
            num_sub_vectors=self.num_sub_vectors,
        )
        return records


def lancedb_retrieval(
    queries,
    table_path: str,
    table_name: str = "nv-ingest",
    embedding_endpoint: str = "http://localhost:8012/v1",
    nvidia_api_key: str = None,
    model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
    top_k: int = 10,
    refine_factor: int = 50,
    n_probe: int = 64,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_top_k: int = 100,
    **kwargs,
):
    """Standalone LanceDB retrieval function for harness compatibility.

    This function mirrors `nvingest_retrieval` from the Milvus module, providing
    a simple interface for recall evaluation against a LanceDB table.

    Parameters
    ----------
    queries : list[str]
        Text queries to search for.
    table_path : str
        Path to the LanceDB database directory (the `uri` used during ingestion).
    table_name : str, optional
        Name of the table within the LanceDB database (default is "nv-ingest").
    embedding_endpoint : str, optional
        URL of the embedding microservice (default is "http://localhost:8012/v1").
    nvidia_api_key : str, optional
        NVIDIA API key for authentication. If None, no auth is used.
    model_name : str, optional
        Name of the embedding model (default is "nvidia/llama-3.2-nv-embedqa-1b-v2").
    top_k : int, optional
        Number of results to return per query (default is 10).
    refine_factor : int, optional
        LanceDB search refine factor for accuracy (default is 2).
    n_probe : int, optional
        Number of partitions to probe during search (default is 32).
    nv_ranker : bool, optional
        Whether to apply NV reranker after retrieval (default is False).
    nv_ranker_endpoint : str, optional
        URL of the reranker microservice (e.g., "http://localhost:8020/v1/ranking").
    nv_ranker_model_name : str, optional
        Name of the reranker model (e.g., "nvidia/llama-3.2-nv-rerankqa-1b-v2").
    nv_ranker_top_k : int, optional
        Number of candidates to fetch before reranking (default is 50).
    **kwargs
        Additional keyword arguments (ignored, for API compatibility).

    Returns
    -------
    list[list[dict]]
        For each query, a list of result dicts with keys: `source`, `metadata`, `text`.
        Results are formatted to match Milvus output structure for recall scoring.
    """
    db = lancedb.connect(uri=table_path)
    table = db.open_table(table_name)

    embed_model = partial(
        infer_microservice,
        model_name=model_name,
        embedding_endpoint=embedding_endpoint,
        nvidia_api_key=nvidia_api_key,
        input_type="query",
        output_names=["embeddings"],
        grpc=not ("http" in urlparse(embedding_endpoint).scheme),
    )

    # When using reranker, fetch more candidates initially
    search_top_k = nv_ranker_top_k if nv_ranker else top_k

    results = []
    query_embeddings = embed_model(queries)
    for query_embed in query_embeddings:
        search_results = (
            table.search([query_embed], vector_column_name="vector")
            .disable_scoring_autoprojection()
            .select(["text", "metadata", "source", "_distance"])
            .limit(search_top_k)
            .refine_factor(refine_factor)
            .nprobes(n_probe)
            .to_list()
        )
        # Format results to match Milvus structure for recall scoring compatibility
        # Milvus returns: {"entity": {"source": {"source_id": ...}, "content_metadata": {"page_number": ...}}}
        # LanceDB stores: {"source": "...", "metadata": "page_num", "text": "..."}
        formatted = []
        for r in search_results:
            formatted.append(
                {
                    "entity": {
                        "source": {"source_id": r["source"]},
                        "content_metadata": {"page_number": r["metadata"]},
                        "text": r["text"],
                    }
                }
            )
        results.append(formatted)

    # Apply reranking if enabled
    if nv_ranker:
        debug_enabled = os.getenv("NV_INGEST_RERANK_DEBUG", "").lower() in {"1", "true", "yes"}
        debug_limit = int(os.getenv("NV_INGEST_RERANK_DEBUG_LIMIT", "0"))
        rerank_results = []
        for idx, (query, candidates) in enumerate(zip(queries, results)):
            if debug_enabled and (debug_limit == 0 or idx < debug_limit):
                print("=" * 60)
                print(f"[rerank debug] query[{idx}]: {query}")
                print(f"[rerank debug] candidates: {len(candidates)}")
                for c_idx, cand in enumerate(candidates[:5]):
                    text = cand.get("entity", {}).get("text", "")
                    source_id = cand.get("entity", {}).get("source", {}).get("source_id", "")
                    page_number = cand.get("entity", {}).get("content_metadata", {}).get("page_number", "")
                    text_preview = text[:200].replace("\n", " ")
                    print(f"  cand[{c_idx}] source={source_id} page={page_number} text='{text_preview}'")
            rerank_results.append(
                nv_rerank(
                    query,
                    candidates,
                    reranker_endpoint=nv_ranker_endpoint,
                    model_name=nv_ranker_model_name,
                    topk=top_k,
                )
            )
        results = rerank_results

    return results
