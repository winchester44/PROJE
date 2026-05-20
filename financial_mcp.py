import asyncio
import os
import json
import argparse
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Load environment variables
load_dotenv()

API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
SERVER_URL = os.getenv("FINANCIAL_DATASETS_URL", "https://mcp.financialdatasets.ai/api")

async def call_financial_tool(tool_name, tool_args):
    """
    Calls a tool on the Financial Datasets MCP server.
    """
    if not API_KEY:
        return {"error": "FINANCIAL_DATASETS_API_KEY not found in .env file"}

    try:
        async with streamablehttp_client(
            SERVER_URL,
            headers={"X-API-KEY": API_KEY},
        ) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # List available tools if tool_name is "list"
                if tool_name == "list":
                    tools = await session.list_tools()
                    return {"tools": [t.name for t in tools]}
                
                result = await session.call_tool(tool_name, tool_args)
                
                # Check if result has content
                if hasattr(result, 'content'):
                    # The content is usually a list of TextContent or ImageContent
                    text_contents = [c.text for c in result.content if hasattr(c, 'text')]
                    return {"result": "\n".join(text_contents)}
                else:
                    return {"result": str(result)}

    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Financial Datasets MCP Client")
    parser.add_argument("tool", help="Tool name to call (or 'list' to see all tools)")
    parser.add_argument("--args", help="JSON string of arguments for the tool", default="{}")
    
    args = parser.parse_args()
    
    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON in --args"}))
        return

    result = asyncio.run(call_financial_tool(args.tool, tool_args))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
