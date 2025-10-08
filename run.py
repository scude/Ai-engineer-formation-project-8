"""Entry point for running the Flask development server."""
from __future__ import annotations

from app import create_app
from app.config import MAX_CONTENT_LENGTH

app = create_app({"MAX_CONTENT_LENGTH": MAX_CONTENT_LENGTH})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)