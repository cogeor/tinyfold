"""
TinyFold Light Viewer - Minimal server for embedding in other applications.

No model loading, just serves the viewer UI that accepts coordinates via:
1. postMessage API (for iframe embedding)
2. window.tinyfold JS API (for same-origin usage)

Usage:
    python server.py [--port 5002]
"""
import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Paths
SCRIPT_DIR = Path(__file__).parent
WEB_DIR = SCRIPT_DIR.parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="TinyFold Light Viewer")

# Mount static files from main web directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the viewer in embed mode with injected config."""
    # Read the original HTML
    html_path = STATIC_DIR / "index.html"
    html = html_path.read_text()

    # Inject embed mode config before other scripts
    inject_script = """
    <script>
        window.TINYFOLD_MODE = 'embed';
    </script>
    <script src="/static/js/api.js"></script>
"""
    html = html.replace(
        '<script src="/static/js/api.js"></script>',
        inject_script
    )

    return html


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "mode": "light"}


def main():
    parser = argparse.ArgumentParser(description="TinyFold Light Viewer Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5002, help="Port to listen on")
    args = parser.parse_args()

    import uvicorn
    print(f"Starting TinyFold Light Viewer at http://{args.host}:{args.port}")
    print("API available at window.tinyfold or via postMessage")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
