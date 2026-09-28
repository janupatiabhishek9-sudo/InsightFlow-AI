"""Knowledge MCP server: search_business_docs, get_kpi_definition, get_data_dictionary (access-level filtered).

Run:  python -m app.mcp_servers.knowledge_server
"""

from app.mcp_servers.common import main

if __name__ == "__main__":
    main("knowledge")
