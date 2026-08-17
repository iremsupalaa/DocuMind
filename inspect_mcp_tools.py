import json

from app import MCPClient, MCP_URL


mcp = MCPClient(MCP_URL)

try:
    mcp.connect()

    tools = mcp.list_tools()

    for tool in tools:
        print("\n" + "=" * 60)
        print("Araç adı:", tool["name"])
        print("Açıklama:", tool.get("description", "Açıklama yok"))
        print("Parametreler:")

        print(
            json.dumps(
                tool.get("inputSchema", {}),
                ensure_ascii=False,
                indent=2,
            )
        )

finally:
    mcp.close()
    