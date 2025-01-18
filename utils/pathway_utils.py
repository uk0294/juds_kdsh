from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.parsers import ParseUtf8
import pathway as pw
from PyPDF2 import PdfReader
import io

# Conference Mapping
conference_mapping = {
    "R006": "CVPR",
    "R007": "CVPR",
    "R008": "EMNLP",
    "R009": "EMNLP",
    "R0010": "KDD",
    "R0011": "KDD",
    "R0012": "NeurIPS",
    "R0013": "NeurIPS",
    "R0014": "TMLR",
    "R0015": "TMLR",
}

def add_conference_metadata(metadata):
    """Add conference metadata to labeled data."""
    if not isinstance(metadata, dict):
        metadata = {}
    name = metadata.get("name", "")
    metadata["conference"] = conference_mapping.get(name, "Unknown")
    return metadata

def parse_pdf(content: bytes) -> list[tuple[str, dict]]:
    """Parse PDF content."""
    try:
        reader = PdfReader(io.BytesIO(content))
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        return [(text, {})]
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return []

def conditional_parser(content: bytes) -> list[tuple[str, dict]]:
    """Try UTF-8 parsing or fallback to PDF parsing."""
    try:
        text = content.decode("utf-8")
        return [(text, {})]
    except UnicodeDecodeError:
        return parse_pdf(content)

def setup_vector_store(object_id, service_user_credentials, port, embedder, splitter):
    """Set up a Pathway vector store."""
    table = pw.io.gdrive.read(
        object_id=object_id,
        service_user_credentials_file=service_user_credentials,
        mode="static",
        with_metadata=True,
    )
    table = table.select(
        data=pw.this.data,
        _metadata=pw.apply(add_conference_metadata, pw.this._metadata),
    )
    vector_server = VectorStoreServer(
        table,
        parser=conditional_parser,
        embedder=embedder,
        splitter=splitter,
    )
    vector_server.run_server(host="127.0.0.1", port=port, threaded=True, with_cache=True)
    client = VectorStoreClient(host="127.0.0.1", port=port)
    return client
