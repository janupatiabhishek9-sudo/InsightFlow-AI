"""Python Analysis MCP server: run_analysis (sandboxed), create_chart, run_statistical_test.

Run:  python -m app.mcp_servers.python_server --dataset data/examples/sales.csv
"""

from app.mcp_servers.common import main

if __name__ == "__main__":
    main("python")
