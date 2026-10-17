"""Run the private subscription broker with a systemd-scoped client credential."""

from __future__ import annotations
import os
from pathlib import Path


def main() -> None:
    """Read only the service credential and delegate to the application's CLI."""
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory:
        raise RuntimeError("The systemd credential directory is required")
    token = (Path(directory) / "broker-token").read_text().strip()
    if len(token) < 32 or not token.isascii() or any(character.isspace() for character in token):
        raise RuntimeError("The scoped broker credential has an invalid format")
    os.environ["DJ_REACHY_REMIX_SUBSCRIPTION_BROKER_TOKEN"] = token
    from dj_reachy_remix.subscription_broker import main as broker_main

    broker_main()


if __name__ == "__main__":
    main()
