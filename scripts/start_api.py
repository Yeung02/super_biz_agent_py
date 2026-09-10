"""Bootstrap the FastAPI service with a Milvus connection.

The app imports the LangChain Milvus vector store during module import, so the
PyMilvus default connection must exist before Uvicorn imports app.main.
"""

from app.config import config
from app.core.milvus_client import milvus_manager


def main() -> None:
    milvus_manager.connect()

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
