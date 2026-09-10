from __future__ import annotations

from langchain_core.documents import Document

from app.rag.models import RagContext
from app.rag.retriever import RetrievalResult
from app.tools import knowledge_tool


def test_retrieve_knowledge_uses_stage3b_context_when_enabled(monkeypatch) -> None:
    fake_retriever = _FakeRetriever()
    fake_context_builder = _FakeContextBuilder()

    monkeypatch.setattr(knowledge_tool.config, "new_rag_retriever_enabled", True)
    monkeypatch.setattr(knowledge_tool.config, "context_builder_enabled", True)
    monkeypatch.setattr(
        knowledge_tool,
        "RagRetriever",
        lambda **kwargs: fake_retriever,
        raising=False,
    )
    monkeypatch.setattr(
        knowledge_tool,
        "ContextBuilder",
        lambda: fake_context_builder,
        raising=False,
    )
    monkeypatch.setattr(
        knowledge_tool.vector_store_manager,
        "get_vector_store",
        _old_vector_store_must_not_be_used,
    )

    content, artifact = knowledge_tool.retrieve_knowledge.func("CPU high usage")

    assert content == "packed context"
    assert artifact == []
    assert fake_retriever.calls == ["CPU high usage"]
    assert fake_context_builder.calls == [("CPU high usage", [])]


def test_retrieve_knowledge_keeps_legacy_vector_store_path_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(knowledge_tool.config, "new_rag_retriever_enabled", False)
    monkeypatch.setattr(knowledge_tool.config, "context_builder_enabled", False)
    monkeypatch.setattr(
        knowledge_tool.vector_store_manager,
        "get_vector_store",
        lambda: _FakeVectorStore(),
    )

    content, artifact = knowledge_tool.retrieve_knowledge.func("CPU high usage")

    assert "CPU runbook" in content
    assert len(artifact) == 1
    assert artifact[0].page_content == "CPU runbook"


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.calls.append(query)
        return RetrievalResult(chunks=[])


class _FakeContextBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def build(self, query: str, chunks: object, budget: object) -> RagContext:
        _ = budget
        self.calls.append((query, chunks))
        return RagContext(context_text="packed context")


class _FakeVectorStore:
    def as_retriever(self, search_kwargs: dict[str, object]) -> object:
        _ = search_kwargs
        return _FakeLegacyRetriever()


class _FakeLegacyRetriever:
    def invoke(self, query: str) -> list[Document]:
        _ = query
        return [
            Document(
                page_content="CPU runbook",
                metadata={"_file_name": "cpu.md"},
            )
        ]


def _old_vector_store_must_not_be_used() -> object:
    raise AssertionError("legacy vector_store path should not be used")
