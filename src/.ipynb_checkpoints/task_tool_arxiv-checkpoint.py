from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langchain.agents import create_agent
from langgraph.types import Command
from deep_agents_from_scratch.state_arxiv import DeepAgentState

class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: Optional[List[str]]

def _create_task_tool(all_tools: List, subagents: List[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    # Build tool name mapping
    tools_by_name = {}
    for t in all_tools:
        if not isinstance(t, BaseTool):
            t = tool(t)  # ensure it's a BaseTool
        tools_by_name[t.name] = t

    # Create sub‑agents
    agents = {}
    for sub in subagents:
        if "tools" in sub and sub["tools"]:
            sub_tools = [tools_by_name[name] for name in sub["tools"] if name in tools_by_name]
        else:
            sub_tools = all_tools  # default to all tools
        agents[sub["name"]] = create_agent(
            model,
            tools=sub_tools,
            system_prompt=sub["prompt"],
            state_schema=state_schema
        )

    # Build description of available agents
    agent_descriptions = "\n".join([f"- {a['name']}: {a['description']}" for a in subagents])

    @tool(description=f"Delegate a task to a specialized sub‑agent with isolated context. Available agents:\n{agent_descriptions}")
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Delegate a task to a specialized sub-agent with isolated context.
        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        if subagent_type not in agents:
            return Command(
                update={
                    "messages": [
                        ToolMessage(f"Error: unknown subagent type '{subagent_type}'", tool_call_id=tool_call_id)
                    ]
                }
            )

        sub_agent = agents[subagent_type]

        # Create isolated sub‑state – copy only the files (so sub‑agent can write)
        sub_state = {
            "messages": [{"role": "user", "content": description}],
            "todos": [],                     # fresh todo list
            "files": state.get("files", {}),  # share files (optional)
            "web_search_count": 0,            # start fresh counter
        }

        # Run sub‑agent
        result = sub_agent.invoke(sub_state)

        # Merge files (sub‑agent’s new files override parent’s if same name)
        #new_files = {**state.get("files", {}), **result.get("files", {})}

        # Return results to parent agent via Command state update
        return Command(
            update={
                "files": result.get("files", {}),  # Merge any file changes
                "messages": [
                    # Sub-agent result becomes a ToolMessage in parent context
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task