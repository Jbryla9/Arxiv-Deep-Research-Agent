
"""Research Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

# -------------------- Imports --------------------
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.documents import Document
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from markdownify import markdownify
from tavily import TavilyClient
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages
#from prompts import SUMMARIZE_WEB_SEARCH
#from prompts import WRITE_TODOS_DESCRIPTION

#
from task_tool_arxiv import _create_task_tool
from state_arxiv import DeepAgentState



import os
from datetime import datetime
import uuid, base64

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal, List

from prompts_arxiv import SUMMARIZE_WEB_SEARCH
from state_arxiv import DeepAgentState

# Summarization model 
summarization_model = init_chat_model(model="openai:gpt-4o-mini")
tavily_client = TavilyClient()

class Summary(BaseModel):
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %d, %Y").replace(" 0", " ")

def run_tavily_search(
    search_query: str, 
    max_results: int = 1, 
    include_raw_content: bool = True, 
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content
    )
    return result

def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Summary object with filename and summary
    """
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_and_filename = structured_model.invoke([
            HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])
        return summary_and_filename
    except Exception:
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        )

def process_search_results(results: dict) -> List[dict]:
    """Process search results by summarizing content where available.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    HTTPX_CLIENT = httpx.Client(timeout=30.0)
    processed_results = []
    for result in results.get('results', []):
        url = result['url']
        try:
            response = HTTPX_CLIENT.get(url)
            if response.status_code == 200:
                raw_content = markdownify(response.text)
                summary_obj = summarize_webpage_content(raw_content)
            else:
                raw_content = result.get('raw_content', '')
                summary_obj = Summary(
                    filename="URL_error.md",
                    summary=result.get('content', 'Error reading URL; try another search.')
                )
        except (httpx.TimeoutException, httpx.RequestError):
            raw_content = result.get('raw_content', '')
            summary_obj = Summary(
                filename="connection_error.md",
                summary=result.get('content', 'Could not fetch URL (timeout/connection error). Try another search.')
            )
        # uniquify file names
        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"
        processed_results.append({
            'url': url,
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })
    return processed_results


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: int = 1
) -> Command:
    """Search the web and save detailed results to files while returning a minimal context summary.

    Performs a Tavily search, fetches full webpage content for each result,
    summarizes it, saves the raw content and summary to files in the virtual
    filesystem, and returns a brief overview to the agent.

    Args:
        query: The search query string.
        state: Injected agent state containing the current file system.
        tool_call_id: Injected tool call identifier for message responses.
        max_results: Maximum number of search results to return (default: 1)

    Returns:
        Command updating the agent's files with saved search results and
        adding a tool message containing a summary of what was saved.
    """
    search_results = run_tavily_search(query, max_results=max_results, include_raw_content=True)
    processed_results = process_search_results(search_results)
    files = state.get("files", {})
    saved_files = []
    summaries = []
    for result in processed_results:
        filename = result['filename']
        file_content = f"""# Search Result: {result['title']}

                **URL:** {result['url']}
                **Query:** {query}
                **Date:** {get_today_str()}

                ## Summary
                {result['summary']}

                ## Raw Content
                {result['raw_content'] if result['raw_content'] else 'No raw content available'}
                """
        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}

💡 Use read_file() to access full details when needed."""
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(summary_text, tool_call_id=tool_call_id)
            ],
        }
    )

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
