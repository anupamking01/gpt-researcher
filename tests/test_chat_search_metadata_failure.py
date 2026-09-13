from types import SimpleNamespace

import pytest

from backend.chat.chat import ChatAgentWithMemory


class SearchClient:
    def __init__(self):
        self.calls = 0

    def search(self, *, query, max_results):
        assert max_results == 5
        self.calls += 1

        if self.calls == 1:
            return {
                "results": [
                    {
                        "title": "First source",
                        "url": "https://example.com/first",
                        "content": "First result",
                    }
                ]
            }

        raise RuntimeError(f"search failed for {query}")


@pytest.mark.asyncio
async def test_failed_second_search_does_not_reuse_first_metadata(monkeypatch):
    async def fake_completion(**_kwargs):
        return "answer", [
            {
                "tool": "search_tool",
                "args": {"query": "first query"},
            },
            {
                "tool": "search_tool",
                "args": {"query": "second query"},
            },
        ]

    monkeypatch.setattr(
        "backend.chat.chat.create_chat_completion_with_tools",
        fake_completion,
    )

    agent = ChatAgentWithMemory.__new__(ChatAgentWithMemory)
    agent.config = SimpleNamespace(
        smart_llm_model="test-model",
        smart_llm_provider="test-provider",
        llm_kwargs={},
    )
    agent.tavily_client = SearchClient()
    agent.search_metadata = None

    response, metadata = await agent.process_chat_completion([])

    assert response == "answer"
    assert metadata[0]["search_metadata"] == {
        "query": "first query",
        "sources": [
            {
                "title": "First source",
                "url": "https://example.com/first",
                "content": "First result",
            }
        ],
    }
    assert metadata[1]["search_metadata"] == {
        "query": "second query",
        "sources": [],
        "error": "search failed for second query",
    }


def test_failed_search_before_any_success_records_current_query():
    class FailingClient:
        def search(self, *, query, max_results):
            assert max_results == 5
            raise RuntimeError(f"search failed for {query}")

    agent = ChatAgentWithMemory.__new__(ChatAgentWithMemory)
    agent.tavily_client = FailingClient()
    agent.search_metadata = None

    result = agent.quick_search("only query")

    assert result == {
        "error": "search failed for only query",
        "results": [],
    }
    assert agent.search_metadata == {
        "query": "only query",
        "sources": [],
        "error": "search failed for only query",
    }
