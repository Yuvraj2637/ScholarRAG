from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import settings
from src.ingestion.chunker import Chunk

COLLECTION_NAME = "transformer_papers_dense"

class DenseRetriever:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        persist_dir: Path = settings.CHROMA_DIR
    ):
        self.model_name = model_name
        self.persist_dir = str(persist_dir)
        self._embedder = None
        self._client = None
        self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64, force_reindex: bool = False) -> int:
        """Embed and index chunks in ChromaDB vector store."""
        current_count = self.collection.count()
        if current_count > 0 and not force_reindex:
            print(f"[INFO] Dense index already contains {current_count} vectors.")
            return current_count

        if force_reindex and current_count > 0:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

        print(f"[INFO] Indexing {len(chunks)} chunks into ChromaDB with {self.model_name}...")
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
                "section_title": c.section_title,
                "year": c.year,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "token_count": c.token_count,
                "primary_topic": c.primary_topic
            }
            for c in chunks
        ]

        # Batch embedding and insertion
        for i in range(0, len(chunks), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            embeddings = self.embedder.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas
            )

        print(f"[INFO] Successfully indexed {self.collection.count()} chunks in ChromaDB.")
        return self.collection.count()

    def search(self, query: str, top_k: int = settings.DENSE_TOP_K) -> List[Dict[str, Any]]:
        """Perform dense semantic search given a query."""
        # For BGE models, queries can benefit from instruction if using query format
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                # Cosine distance: similarity = 1 - distance
                dist = results["distances"][0][idx] if "distances" in results else 0.0
                score = 1.0 - dist

                hits.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "score": float(score),
                    "paper_id": meta.get("paper_id", ""),
                    "paper_title": meta.get("paper_title", ""),
                    "section_title": meta.get("section_title", ""),
                    "year": meta.get("year", 0),
                    "page_start": meta.get("page_start", 1),
                    "page_end": meta.get("page_end", 1),
                    "primary_topic": meta.get("primary_topic", ""),
                    "retriever": "dense"
                })

        return hits
