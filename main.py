"""Entry point for running the app directly: python main.py"""

import argparse


def cli():
    parser = argparse.ArgumentParser(description="Moondream Chat")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve on (default: from config.yaml, or 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config.yaml, or 0.0.0.0)",
    )
    args = parser.parse_args()

    from src.app import main

    main(host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
