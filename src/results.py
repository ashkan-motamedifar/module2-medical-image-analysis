import json
import platform
from datetime import datetime
from pathlib import Path

import torch


def save_run(tag, config, metrics, history=None, model=None, root="results"):
    """Writes results/<timestamp>_<tag>/ and returns the path. Never overwrites."""
    run_dir = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(json.dumps({
        **config,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }, indent=2, default=str))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if history is not None:
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    if model is not None:
        torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"saved -> {run_dir}")
    return run_dir


def results_table(rows, columns, float_format="{:.2f}"):
    """Markdown table you can paste straight into the report."""
    header = "| Model | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines = [header, sep]
    for name, metrics in rows.items():
        values = [float_format.format(metrics[c]) if isinstance(metrics.get(c), float)
                  else str(metrics.get(c, "")) for c in columns]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return "\n".join(lines)
