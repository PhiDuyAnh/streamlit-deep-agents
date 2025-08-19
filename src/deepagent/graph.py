import os
from datetime import UTC, datetime

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from deepagent.prompts import AGENT_INSTRUCTIONS
from deepagent.tools import TOOLS

model = ChatOpenAI(
    model=os.environ["MODEL_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
)

agent = create_deep_agent(
    model=model,
    tools=TOOLS,
    instructions=AGENT_INSTRUCTIONS.format(current_date=datetime.now(tz=UTC).strftime("%Y-%m-%d"))
)