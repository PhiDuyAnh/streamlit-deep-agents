import asyncio, nest_asyncio
import uuid
import os
from datetime import UTC, datetime

import streamlit as st
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

from deepagent.prompts import (
    AGENT_INSTRUCTIONS,
    RESEARCH_INSTRUCTIONS,
    SUB_RESEARCH_PROMPT,
    SUB_CRITIQUE_PROMPT
)
from deepagent.tools import TOOLS
from deepagent.mcp_tools import get_mcp_tools

nest_asyncio.apply() # For Jupyter interactive window


@st.cache_resource
def get_checkpointer():
    """Caches the memory checkpointer."""
    return MemorySaver()


def format_instructions(instruction):
    """Format the prompts."""
    return instruction.format(current_date=datetime.now(tz=UTC).strftime("%Y-%m-%d"))


@st.cache_resource
def create_sub_agents():
    """Create sub agents for deep researcher."""
    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
        "prompt": format_instructions(SUB_RESEARCH_PROMPT),
        "tools": ["internet_search"]
    }

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report. Give this agent some infomration about how you want it to critique the report.",
        "prompt": format_instructions(SUB_CRITIQUE_PROMPT)
    }

    return research_sub_agent, critique_sub_agent


@st.cache_resource
def create_agent(mode: str, model_name: str):
    """Initialize the agents depending on the modes: Normal or Deep Research mode."""

    try:
        MCP_TOOLS = asyncio.run(get_mcp_tools())
    except Exception as e:
        raise ValueError(f"You may have not set up an MCP Server, please do so or remove MCP_TOOLS. The error message: {str(e)}")

    all_tools = TOOLS + MCP_TOOLS
    if mode == "Normal Mode":
        model = ChatOpenAI(model=model_name, api_key=os.environ["OPENAI_API_KEY"])
        agent = create_deep_agent(
            tools=all_tools,
            model=model,
            instructions=format_instructions(AGENT_INSTRUCTIONS),
            checkpointer=get_checkpointer()
        )
        return agent
    else:
        research_sub_agent, critique_sub_agent = create_sub_agents()
        model = ChatOpenAI(model=model_name, api_key=os.environ["OPENAI_API_KEY"])
        agent = create_deep_agent(
            tools=all_tools,
            model=model,
            subagents=[research_sub_agent, critique_sub_agent],
            instructions=format_instructions(RESEARCH_INSTRUCTIONS),
            checkpointer=get_checkpointer()
        )
        return agent


async def invoke_agent_responses(agent, prompt, config):
    """Stream agent responses and get the last message to display."""
    # last_msg = None
    # for event in agent.stream(
    #     {"messages": [{"role": "user", "content": prompt}]},
    #     config
    # ):
    #     for value in event.values():
    #         last_msg = value["messages"][-1].content
    
    # return last_msg
    return await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config
    )


async def main():
    # Page title
    st.set_page_config(
        page_title="DeepAgent",
        initial_sidebar_state="collapsed"
    )
    st.title("DeepAgent 🦜🔗")

    tab_names = ["Normal Mode", "Deep Research"]
    active_tab = st.radio("Mode 🚀", tab_names, index=0, horizontal=True)

    # Session states
    if "messages" not in st.session_state:
        st.session_state["messages"] = {name: [] for name in tab_names}
    for message in st.session_state["messages"][active_tab]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = {name: str(uuid.uuid4()) for name in tab_names}

    # Create the agents
    model_names = {
        "Normal Mode": os.environ["MODEL_NAME"],
        "Deep Research": os.environ["MODEL_NAME"]
    }

    agents = {name: create_agent(name, model_names[name]) for name in tab_names}

    # Clear history and memory
    st.markdown(
    """
    <style>
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
            margin-top: -1.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            padding-bottom: 0rem;
        }

        /* == Custom style for the sidebar title == */
        [data-testid="stMarkdownContainer"] > h2 {
            font-size: 2.0rem;
            font-weight: 650;    /* Adjust boldness (e.g., 500, 700) */
            margin-top: -1.0rem;     /* Reduce space ABOVE the title text */
            /* margin-bottom: 0rem;
            /* color: #FF4B4B;
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.write("")
        st.warning("Click this button to delete all chat history and reset the agents' memory.")
        reset = st.button("Delete all", type="primary", use_container_width=True)
        if reset:
            st.session_state["messages"] = {name: [] for name in tab_names}
            st.session_state["thread_id"] = {name: str(uuid.uuid4()) for name in tab_names}
            st.rerun()

    # Chat inputs
    loading = {
        "Normal Mode": "Thinking...",
        "Deep Research": "Researching, this may take a few minutes..."
    }
    prompt_key = f"chat_input_{active_tab}"
    prompt = st.chat_input("Ask anything", key=prompt_key)

    if prompt:
        st.session_state["messages"][active_tab].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(loading[active_tab]):
            config = {"configurable": {"thread_id": st.session_state["thread_id"][active_tab]}}
            agent = agents[active_tab]
            response = await invoke_agent_responses(agent, prompt, config)
            displayed_response = response["messages"][-1].content

        st.session_state["messages"][active_tab].append({"role": "assistant", "content": displayed_response})
        with st.chat_message("assistant"):
            st.markdown(displayed_response)

if __name__ == "__main__":
    # main()
    asyncio.run(main())
