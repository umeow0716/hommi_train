from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact import save_portable_checkpoint_model
from .torch_export import export_portable_model_pt2


def default_portable_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / "model.pt"
    return checkpoint.with_name("model.pt")


def run_export(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pt2_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 1,
    strict_export: bool = False,
) -> dict[str, Path]:
    """Create the reliable portable artifact and optionally attempt PT2 export."""
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_portable_path(checkpoint_path)
    )
    save_portable_checkpoint_model(checkpoint_path, output)
    result = {"portable": output}

    if pt2_path is not None:
        pt2 = Path(pt2_path).expanduser().resolve()
        try:
            export_portable_model_pt2(
                output,
                pt2,
                batch_size=batch_size,
                device=device,
                strict=strict_export,
            )
        except Exception as exc:
            raise RuntimeError(
                f"portable model was saved to {output}, but torch.export PT2 "
                f"generation failed ({type(exc).__name__}: {exc}). The portable "
                "model remains valid; PT2 requires the complete inference path "
                "to export as one graph."
            ) from exc
        result["pt2"] = pt2
    return result
