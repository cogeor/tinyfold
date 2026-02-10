#!/usr/bin/env python
"""TinyFold light showcase server (stdlib-only).

Serves static files from web-light/static so the viewer is runnable right
after cloning without additional Python dependencies.
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class ShowcaseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, static_dir: Path, assets_dir: Path, **kwargs):
        self.static_dir = static_dir
        self.assets_dir = assets_dir
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = unquote(parsed.path)
        if clean_path.startswith("/assets/"):
            rel = clean_path.removeprefix("/assets/")
            return str((self.assets_dir / rel).resolve())
        return super().translate_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyFold Light Showcase Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5002, help="Port to listen on")
    args = parser.parse_args()

    static_dir = Path(__file__).parent / "static"
    assets_dir = Path(__file__).parent.parent / "assets"
    if not static_dir.exists():
        raise FileNotFoundError(f"Missing static directory: {static_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Missing assets directory: {assets_dir}")

    def handler(*h_args, **h_kwargs):
        return ShowcaseHandler(*h_args, static_dir=static_dir, assets_dir=assets_dir, **h_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"TinyFold Light Showcase: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
