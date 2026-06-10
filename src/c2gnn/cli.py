"""Command line entry points for common project tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from c2gnn.realtime.pipeline import RealtimePipeline, _api_alert_callback, _load_model

app = typer.Typer(help="Realtime C2 GNN detection utilities")


@app.command()
def demo(
    data: Annotated[Path, typer.Option(exists=True, help="Input .parquet/.binetflow file")],
    model: Annotated[Path, typer.Option(exists=True, help="Trained .pt model checkpoint")],
    model_type: Annotated[str | None, typer.Option(help="graphsage or gatv2")] = None,
    threshold: Annotated[float, typer.Option(help="Alert threshold")] = 0.7,
    window_size: Annotated[float, typer.Option(help="Sliding graph window in seconds")] = 60.0,
    realtime_factor: Annotated[float, typer.Option(help="Replay speed multiplier")] = 50.0,
    max_flows: Annotated[int | None, typer.Option(help="Optional flow cap for a short demo")] = None,
    api_url: Annotated[str | None, typer.Option(help="Optional Alert API base URL")] = None,
) -> None:
    """Replay CTU-13 flows through the realtime graph inference pipeline."""
    loaded_model = _load_model(model, model_type)
    callback = _api_alert_callback(api_url) if api_url else None
    pipeline = RealtimePipeline(
        data_path=data,
        model=loaded_model,
        window_size=window_size,
        threshold=threshold,
        realtime_factor=realtime_factor,
        alert_callback=callback,
        max_flows=max_flows,
    )
    pipeline.run_until_complete()


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI alert server."""
    from c2gnn.api.server import run_api_server

    run_api_server(host=host, port=port)


if __name__ == "__main__":
    app()
