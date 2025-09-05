from agentics import Agentics as AG
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters # For Stdio Server
import os
from agentics.core.llm_connections import available_llms
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

import asyncio

load_dotenv()


server_params=StdioServerParameters(
    command="python3",
    args=[os.getenv("MCP_SERVER_PATH")],
    env={"UV_PYTHON": "3.12", **os.environ},
)
    

class SearchResult(BaseModel):
    title: Optional[str]
    text: Optional[str]
    source_url:Optional[str]

class WebSearchReport(BaseModel):
    """Detailed Report to answer the question using Aristotelian Narrative Structure"""
    short_answer:str
    introduction: str
    detailed_report: Optional[str]  = Field(None, description="Markdown document containing relevant background information for the question being answered.")
    conclusion:str
    relevant_references: list[SearchResult] = Field([], description="Relevant Snippets of text extracted from web search that supports the answers. Do not make up text, just copy from search results if relevant")


with MCPServerAdapter(server_params) as server_tools:
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in server_tools]}")
   
    results = asyncio.run(AG(atype=WebSearchReport,
                            tools = server_tools, 
                            max_iter=10,
                            verbose_agent=True,
                            description="Extract stock market price for the input day ",
                            llm=available_llms["watsonx"]) <<[input("AG>   Day\nUSER> ")])
    print(results.pretty_print())
