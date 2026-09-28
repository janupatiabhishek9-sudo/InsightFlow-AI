"""Analytics MCP server: get_schema, profile_dataset, execute_sql, get_statistics (+ approval-only execute_write_sql).

Run:  python -m app.mcp_servers.analytics_server --dataset data/examples/sales.csv
"""

from app.mcp_servers.common import main

if __name__ == "__main__":
    main("analytics")
