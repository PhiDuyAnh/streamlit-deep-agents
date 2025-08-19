# 🤖 Simple Streamlit DeepAgent (with MCP Tools support) 

A simple Streamlit UI for LangChain's [Deep Agents](https://github.com/langchain-ai/deepagents/) where multiple agents can be implemented and switched between (in this case, a normal deep agent and a deep researcher). Different MCP Servers can also be integrated.

<img width="2559" height="1396" alt="image" src="https://github.com/user-attachments/assets/bf3599b4-22e1-4033-90c1-967f559ce029" />

## 🚀 Quickstart

1. Follow the instructions from this link to install uv package: https://github.com/astral-sh/uv

2. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/PhiDuyAnh/streamlit-deep-agents.git
cd streamlit-deep-agents
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
# or
uv pip install -r pyproject.toml
```

4. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings like LangSmith tracing):
```bash
cp .env.example .env
```

5. If you want to add memory to the agents on Streamlit and allow them to remember conversations, you need to make changes to the `create_deep_agent` function of the `deepagents` package to add a checkpointer:
```python
# .venv/lib/python3.12/site-packages/deepagents/graph.py
from langgraph.types import Checkpointer

def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    checkpointer: Optional[Checkpointer] = None # Add a checkpointer
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
    """
    prompt = instructions + base_prompt
    built_in_tools = [write_todos, write_file, read_file, ls, edit_file]
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + list(tools) + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        checkpointer=checkpointer # Add a checkpointer
    )
```

6. Launch Streamlit app:
```bash
uv run streamlit run app_st.py
```

## ⚙️ Others

### MCP Servers and Tools

Integrate with other MCP Servers and get their tools using the code in `src/deepagent/mcp_tools.py`.

### LangGraph Studio

Test your agents with LangGraph Studio using LangGraph CLI and the file `langgraph.json`.
```bash
uv run langgraph dev
```
