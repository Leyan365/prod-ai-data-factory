"""Entrypoint for running the web interface via python -m training_data_bot.web."""

from .app import run_server

if __name__ == "__main__":
    run_server()
