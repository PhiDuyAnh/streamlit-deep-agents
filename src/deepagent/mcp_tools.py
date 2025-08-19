from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "<mcp-server-name>": {
            "command": "",
            "args": [],
            "transport": "",
        },
    }
)


async def get_mcp_tools():
    return await client.get_tools()