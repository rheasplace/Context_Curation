"""Generate portable client snippets with this machine's absolute paths.

Writes new example files only, never edits an application's active configuration.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=root / "prepared_context")
    parser.add_argument("--db", type=Path, default=root / "local_vectordb")
    parser.add_argument("--output-dir", type=Path, default=root / "mcp-configs")
    parser.add_argument("--graph", type=Path, default=root / "local_graph.json")
    args = parser.parse_args()
    # Keep the virtualenv interpreter path: resolving its symlink loses the venv.
    command = str(Path(sys.executable).absolute())
    server_args = [str(root / "level5_mcp_server.py"), "--folder", str(args.folder.resolve()),
                   "--db", str(args.db.resolve()), "--graph", str(args.graph.resolve())]
    entry = {"command": command, "args": server_args}
    config = {"mcpServers": {"context-curation": entry}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("claude-code.json", "claude-desktop.json", "antigravity.json"):
        (args.output_dir / name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    toml = ("[mcp_servers.context-curation]\ncommand = " + json.dumps(command)
            + "\nargs = " + json.dumps(server_args) + "\n")
    (args.output_dir / "codex.toml").write_text(toml, encoding="utf-8")
    print(f"Wrote client examples to {args.output_dir}. Merge the matching server entry into your client settings.")
    print("No active client configuration was changed. Use this same Python environment to install requirements-mcp.txt.")


if __name__ == "__main__":
    main()
