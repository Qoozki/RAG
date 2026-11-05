import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import config
from datetime import datetime, timezone
from lang_utils import normalize_text, is_arabic


def initialize_vector_store(force_recreate: bool = False):
    """
    Loads documents, splits into chunks, and initializes Qdrant.
    Recreates the collection if forced, if it doesn't exist, or if the
    embedding dimension changed (e.g. after switching embedding models).
    """
    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )
    embed_dim = len(embeddings.embed_query("dimension probe"))

    client = QdrantClient(path=config.QDRANT_PATH)
    collection_exists = client.collection_exists(config.COLLECTION_NAME)

    # 1. Load documents (PyMuPDF handles Arabic PDFs better than pypdf)
    print("Loading documents from directory...")
    loader = DirectoryLoader(
        config.DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        silent_errors=True,
    )
    docs = loader.load()
    for d in docs:
        d.page_content = normalize_text(d.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", "؟", "؛", "،", ". ", "? ", "! ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

    chunks_by_id = {}
    chunk_counter = {}  # per-file chunk numbering
    for idx, chunk in enumerate(splits):
        md = chunk.metadata
        path = md.get("file_path") or md.get("source") or ""
        file_name = os.path.basename(path) if path else "Unknown"

        try:
            modified_at = datetime.fromtimestamp(
                os.path.getmtime(path), tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
        except OSError:
            modified_at = None

        chunk_counter[file_name] = chunk_counter.get(file_name, 0) + 1

        md["chunk_id"] = idx                              # global id (used for neighbour expansion)
        md["file_name"] = file_name
        md["file_type"] = os.path.splitext(file_name)[1].lstrip(".").lower() or None
        md["page_number"] = md.get("page", 0) + 1         # human-friendly (1-based)
        md["total_pages"] = md.get("total_pages")
        md["language"] = "ar" if is_arabic(chunk.page_content) else "en"
        md["chunk_index_in_file"] = chunk_counter[file_name]
        md["char_count"] = len(chunk.page_content)
        md["modified_at"] = modified_at
        # PyMuPDF already supplies: title, author, creationDate, producer, ... (kept as-is)
        chunks_by_id[idx] = chunk

    # 2. Detect stale collection (different embedding size)
    if collection_exists and not force_recreate:
        vectors_cfg = client.get_collection(config.COLLECTION_NAME).config.params.vectors
        existing_size = getattr(vectors_cfg, "size", None)
        if existing_size != embed_dim:
            print(f"Embedding size changed ({existing_size} -> {embed_dim}); recreating collection...")
            force_recreate = True

    if force_recreate and collection_exists:
        print(f"Force-recreating collection '{config.COLLECTION_NAME}'...")
        client.delete_collection(config.COLLECTION_NAME)
        collection_exists = False

    if not collection_exists:
        print("Building new Qdrant collection and indexing documents...")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings,
        )
        if splits:
            vectorstore.add_documents(splits)
    else:
        print(f"Loading existing Qdrant collection '{config.COLLECTION_NAME}'...")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings,
        )

    return client, vectorstore, chunks_by_id