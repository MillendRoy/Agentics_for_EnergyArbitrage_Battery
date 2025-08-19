from typing import List, Optional

from crewai.tools import BaseTool
from memory_backend_client.api.default import (
    get_memory_dbs_list_get_memory_dbs_list_post,
    recall_recall_post,
)
from pydantic import BaseModel, Field

import agentics.core.globals as globals


class RetrievalToolInput(BaseModel):
    query: str


class Passage(BaseModel):
    id: Optional[str] = Field(None, description="The identifier of the passage")
    text: Optional[str] = Field(None, description="The full text of the passage")


class Passages(BaseModel):
    passages: List[Passage] = Field(
        [],
        description="List of relevant passages returned as a result of a search query",
    )


def retrieve_relevant_passages(
    query: str,
    memory_id: Optional[str] = "IBM_FINANCIAL_REPORT_2024",
    k: Optional[int] = 10,
) -> Passages:
    """Retrieve relevant passages from a corpus for a given query"""
    relevant_contexts = Passages()
    if memory_id and memory_id in get_memory_dbs_list_get_memory_dbs_list_post.sync(
        client=globals.memory_client
    ):
        results = recall_recall_post.sync(
            client=globals.memory_client,
            query=query,
            k=k,
            memory_id=memory_id,
            return_metadata=True,
        )

        for doc in results:
            relevant_contexts.passages.append(Passage(text=doc[0], id=doc[1]["pk"]))
        return relevant_contexts


class RetrievalTool(BaseTool):
    name: str = "Retrieval Tool"
    description: str = """Retrieve relevant passages from a corpus for a given query"""
    args_schema = RetrievalToolInput
    k: int = 10
    memory_id: str = "IBM_FINANCIAL_REPORT_2024"

    def __init__(
        self, *args, k: int = 10, memory_id: str = "IBM_FINANCIAL_REPORT_2024", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.memory_id = memory_id

    def _run(self, query: str) -> Passages:
        return retrieve_relevant_passages(query)


# retrieval_tool = StructuredTool.from_function(retrieve_relevant_passages)
# rt = RetrievalTool()
# print(rt.run({"query": "IBM"}))
