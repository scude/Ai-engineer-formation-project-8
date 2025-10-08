"""Entry point for running the Flask development server."""
from __future__ import annotations

from app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)