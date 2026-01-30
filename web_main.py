#!/usr/bin/env python3
"""
Vostok Web Interface Entry Point
================================

Launch the web interface for Vostok Climate Agent.

Usage:
    python web_main.py
    python web_main.py --host 0.0.0.0 --port 8080

The web interface provides:
- Browser-based chat with the Vostok climate agent
- Inline plot display for generated visualizations
- Persistent conversation history
- Access to ERA5 reanalysis data
"""

import os
import sys
import argparse

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Launch Vostok Climate Agent Web Interface"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Check for required environment variables
    from dotenv import load_dotenv
    load_dotenv()

    missing_keys = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        missing_keys.append("ARRAYLAKE_API_KEY")

    if missing_keys:
        print("WARNING: Missing environment variables:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nAdd these to your .env file for full functionality.")
        print()

    # Import and run
    import uvicorn

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ██╗   ██╗ ██████╗ ███████╗████████╗ ██████╗ ██╗  ██╗                   ║
║    ██║   ██║██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██║ ██╔╝                   ║
║    ██║   ██║██║   ██║███████╗   ██║   ██║   ██║█████╔╝                    ║
║    ╚██╗ ██╔╝██║   ██║╚════██║   ██║   ██║   ██║██╔═██╗                    ║
║     ╚████╔╝ ╚██████╔╝███████║   ██║   ╚██████╔╝██║  ██╗                   ║
║      ╚═══╝   ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝                   ║
║                                                                           ║
║                      Vostok Web Interface v1.0                            ║
║                 ─────────────────────────────────────                     ║
║                                                                           ║
║   Starting server at: http://{args.host}:{args.port:<24}              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
