import os
from typing import Literal, List, Callable, Any

from langchain_core.tools import tool
from tavily import TavilyClient

from deepagent.prompts import TAVILY_SEARCH_DESCRIPTION


@tool(description=TAVILY_SEARCH_DESCRIPTION)
def internet_search(
    query: str,
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = 5,
    include_raw_content: bool = False
):
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_client.search(
        query=query,
        topic=topic,
        max_results=max_results,
        include_raw_content=include_raw_content
    )

TOOLS: List[Callable[..., Any]] = [internet_search]