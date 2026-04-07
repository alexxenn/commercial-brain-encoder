"""
monitor.py — Training monitor for CommercialBrainEncoder.

Handles:
  - WandB logging (lazy init, offline fallback when WANDB_API_KEY missing)
  - Rich 4-panel live display: Pearson r / Loss / PSNR / Context Acc
  - Optional Discord webhook notifications on epoch end
  - Best-checkpoint tracking via save_best()
  - Early-stop gate via should_stop()

No print() calls — all output via logging or Rich.
GELU convention followed (no activations here, but nn.GELU used in model).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Module-level logger — callers can configure the root logger as needed.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich imports — optional at import time so the module loads in minimal envs,
# but required at runtime (Rich is in requirements alongside WandB).
# ---------------------------------------------------------------------------
try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False
    logger.warning("rich not installed — live display disabled")

# ---------------------------------------------------------------------------
# WandB import — deferred to lazy init, but we check importability here.
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb_module

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _wandb_module = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed — metrics will not be logged remotely")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_metric(value: float, precision: int = 4) -> str:
    """Format a scalar metric for Rich display."""
    return f"{value:.{precision}f}"


def _build_display_table(
    step: int,
    pearson_r: float,
    losses: dict[str, float],
    psnr: float,
    context_acc: float,
) -> Table:
    """Build a single Rich Table with 4 metric columns."""
    table = Table(
        show_header=True,
        header_style="bold cyan",
        expand=True,
        title=f"[bold white]Step {step}[/bold white]",
    )
    table.add_column("Pearson r", justify="center", style="green")
    table.add_column("Loss", justify="center", style="red")
    table.add_column("PSNR", justify="center", style="blue")
    table.add_column("Context Acc", justify="center", style="magenta")

    loss_str = "  ".join(
        f"[dim]{k}[/dim] {_format_metric(v)}" for k, v in losses.items()
    )
    table.add_row(
        _format_metric(pearson_r),
        loss_str if loss_str else "-",
        _format_metric(psnr) if psnr >= 0.0 else "-",
        f"{context_acc * 100:.1f}%" if context_acc >= 0.0 else "-",
    )
    return table


def _build_four_panels(
    step: int,
    pearson_r: float,
    losses: dict[str, float],
    psnr: float,
    context_acc: float,
) -> Columns:
    """Build 4 separate Rich Panels, one per metric."""
    panels = [
        Panel(
            Text(_format_metric(pearson_r), justify="center", style="bold green"),
            title="[bold]Pearson r[/bold]",
            subtitle=f"step {step}",
            border_style="green",
        ),
        Panel(
            Text(
                "\n".join(f"{k}: {_format_metric(v)}" for k, v in losses.items()) or "-",
                justify="center",
                style="bold red",
            ),
            title="[bold]Loss[/bold]",
            border_style="red",
        ),
        Panel(
            Text(
                _format_metric(psnr) if psnr >= 0.0 else "-",
                justify="center",
                style="bold blue",
            ),
            title="[bold]PSNR[/bold]",
            border_style="blue",
        ),
        Panel(
            Text(
                f"{context_acc * 100:.1f}%" if context_acc >= 0.0 else "-",
                justify="center",
                style="bold magenta",
            ),
            title="[bold]Context Acc[/bold]",
            border_style="magenta",
        ),
    ]
    return Columns(panels, equal=True, expand=True)


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    """
    Central training monitor for CommercialBrainEncoder.

    Responsibilities:
      - WandB metric logging (step + epoch level)
      - Rich 4-panel live terminal display
      - Discord epoch-end notifications (optional)
      - Best-checkpoint tracking and saving
      - Early-stop gate based on Pearson r threshold

    Dry-run safe: initializes without error when WANDB_API_KEY is not set
    by falling back to ``wandb.init(mode="offline")``.

    Args:
        wandb_project: WandB project name.
        run_name: WandB run display name.
        discord_webhook: Full Discord webhook URL. If None, notifications are
            silently suppressed.
    """

    def __init__(
        self,
        wandb_project: str,
        run_name: str,
        discord_webhook: str | None = None,
    ) -> None:
        self._wandb_project = wandb_project
        self._run_name = run_name
        self._discord_webhook: str | None = discord_webhook

        # Internal state
        self._best_pearson_r: float = float("-inf")
        self._wandb_run: Any = None  # wandb.sdk.wandb_run.Run | None
        self._wandb_initialized: bool = False

        # Cached last values for live display (updated on log_step)
        self._last_step: int = 0
        self._last_pearson_r: float = 0.0
        self._last_losses: dict[str, float] = {}
        self._last_psnr: float = -1.0      # -1 sentinel = not yet available
        self._last_context_acc: float = -1.0

        # Rich live display
        self._console: Console | None = None
        self._live: Live | None = None
        if _RICH_AVAILABLE:
            self._console = Console()
            self._live = Live(
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()
            logger.debug("Rich live display started")

        # Lazy WandB init — deferred to first log call so __init__ never raises.
        # We kick it off here via _ensure_wandb() to surface config errors early
        # without crashing on missing credentials.
        self._ensure_wandb()

    # ------------------------------------------------------------------
    # WandB lifecycle
    # ------------------------------------------------------------------

    def _ensure_wandb(self) -> None:
        """Initialize WandB run if not already done. Falls back to offline mode."""
        if self._wandb_initialized:
            return
        if not _WANDB_AVAILABLE:
            logger.warning("wandb not available — skipping metric logging")
            self._wandb_initialized = True
            return

        api_key_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
        mode = "online" if api_key_present else "offline"

        if not api_key_present:
            logger.info(
                "WANDB_API_KEY not set — initializing wandb in offline mode. "
                "Metrics stored locally only."
            )

        try:
            self._wandb_run = _wandb_module.init(
                project=self._wandb_project,
                name=self._run_name,
                mode=mode,
                reinit=True,
            )
            logger.debug("wandb initialized: project=%s run=%s mode=%s",
                         self._wandb_project, self._run_name, mode)
        except Exception:  # noqa: BLE001
            logger.exception(
                "wandb.init() failed — metrics will not be logged. "
                "Training continues."
            )
            self._wandb_run = None

        self._wandb_initialized = True

    def _wandb_log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Safe wandb.log() wrapper — never raises."""
        if not _WANDB_AVAILABLE or self._wandb_run is None:
            return
        try:
            kwargs: dict[str, Any] = {"commit": True}
            if step is not None:
                kwargs["step"] = step
            self._wandb_run.log(payload, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("wandb.log() failed (suppressed)", exc_info=True)

    # ------------------------------------------------------------------
    # Rich display
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        """Re-render the live 4-panel display with latest cached values."""
        if not _RICH_AVAILABLE or self._live is None:
            return
        renderable = _build_four_panels(
            step=self._last_step,
            pearson_r=self._last_pearson_r,
            losses=self._last_losses,
            psnr=self._last_psnr,
            context_acc=self._last_context_acc,
        )
        self._live.update(renderable)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        losses: dict[str, float],
        pearson_r: float,
    ) -> None:
        """
        Log per-step training metrics.

        Writes to WandB and updates the Rich live display.

        Args:
            step: Global training step index.
            losses: Dict of named loss components, e.g.
                ``{"total": 0.5, "voxel": 0.3, "recon": 0.15, "context": 0.05}``.
            pearson_r: Batch Pearson r (scalar float from
                ``BrainEncoderLoss.pearson_metric()``, shape: ``()``).
        """
        # Extract optional derived metrics from losses dict if present.
        # (Callers may pass psnr / context_acc via the losses dict as a
        #  convenience — we pop them so they render in the correct panels.)
        psnr: float = float(losses.pop("psnr", self._last_psnr))
        context_acc: float = float(losses.pop("context_acc", self._last_context_acc))

        # Update cached state
        self._last_step = step
        self._last_pearson_r = pearson_r
        self._last_losses = dict(losses)
        self._last_psnr = psnr
        self._last_context_acc = context_acc

        # WandB
        payload: dict[str, Any] = {
            "train/pearson_r": pearson_r,
            **{f"train/{k}": v for k, v in losses.items()},
        }
        if psnr >= 0.0:
            payload["train/psnr"] = psnr
        if context_acc >= 0.0:
            payload["train/context_acc"] = context_acc
        self._wandb_log(payload, step=step)

        # Rich
        self._refresh_display()

    def log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Log per-epoch validation metrics.

        Writes to WandB and sends a Discord notification if webhook is set.

        Args:
            epoch: Zero-based epoch index.
            metrics: Epoch-level metrics, e.g.
                ``{"val_pearson": 0.18, "val_loss": 0.42}``.
        """
        # WandB
        payload: dict[str, Any] = {f"epoch/{k}": v for k, v in metrics.items()}
        payload["epoch/index"] = epoch
        self._wandb_log(payload)

        logger.info("Epoch %d — %s", epoch, metrics)

        # Discord
        if self._discord_webhook:
            self._send_discord(epoch, metrics)

    def save_best(
        self,
        model: torch.nn.Module,
        path: str | Path,
        pearson_r: float,
    ) -> None:
        """
        Save a model checkpoint only if ``pearson_r`` exceeds the current best.

        Tracks best internally across calls. Does NOT save if ``pearson_r``
        is equal to or below the previous best.

        Args:
            model: The PyTorch module to checkpoint
                (``state_dict()`` only — full model not pickled).
            path: Destination path for the ``.pt`` checkpoint file.
            pearson_r: Validation Pearson r achieved at this checkpoint
                (scalar float, shape: ``()``).
        """
        if pearson_r <= self._best_pearson_r:
            logger.debug(
                "save_best: %.4f <= best %.4f — skipping",
                pearson_r,
                self._best_pearson_r,
            )
            return

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "pearson_r": pearson_r,
                "model_state_dict": model.state_dict(),
            },
            save_path,
        )

        logger.info(
            "New best checkpoint: pearson_r=%.4f (prev=%.4f) → %s",
            pearson_r,
            self._best_pearson_r,
            save_path,
        )
        self._best_pearson_r = pearson_r

        self._wandb_log({"best/pearson_r": pearson_r})

        if _RICH_AVAILABLE and self._console is not None:
            self._console.print(
                f"[bold green]New best:[/bold green] pearson_r={pearson_r:.4f} "
                f"→ [dim]{save_path}[/dim]"
            )

    def should_stop(
        self,
        pearson_r: float,
        threshold: float = 0.23,
    ) -> bool:
        """
        Early-stop gate: returns True when target Pearson r is reached.

        This is a one-way gate — once True, training can be halted.
        The threshold of 0.23 matches the TRIBE v2 beat target from the brief.

        Args:
            pearson_r: Current validation Pearson r (scalar float).
            threshold: Stop when ``pearson_r >= threshold``.
                Default 0.23 = TRIBE v2 beat target.

        Returns:
            True if ``pearson_r >= threshold``, else False.
        """
        if pearson_r >= threshold:
            logger.info(
                "should_stop → True: pearson_r=%.4f >= threshold=%.4f",
                pearson_r,
                threshold,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Discord
    # ------------------------------------------------------------------

    def _send_discord(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Send a compact epoch-end message to the configured Discord webhook.

        Uses only ``urllib.request`` — no ``requests`` dependency.
        Silently logs on failure; never raises.

        Args:
            epoch: Zero-based epoch index.
            metrics: Epoch-level metrics dict.
        """
        assert self._discord_webhook is not None  # guarded by caller

        lines = [f"**Epoch {epoch}** — CommercialBrainEncoder"]
        for key, value in metrics.items():
            lines.append(f"  `{key}`: {value:.4f}")

        payload_bytes = json.dumps({"content": "\n".join(lines)}).encode("utf-8")

        try:
            req = urllib.request.Request(
                self._discord_webhook,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
                if status not in (200, 204):
                    logger.warning(
                        "Discord webhook returned unexpected status %d", status
                    )
        except Exception:  # noqa: BLE001
            logger.debug("Discord webhook delivery failed (suppressed)", exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Shut down live display and finish the WandB run.

        Call at end of training. Safe to call multiple times.
        """
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
            self._live = None

        if _WANDB_AVAILABLE and self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:  # noqa: BLE001
                pass
            self._wandb_run = None

    def __enter__(self) -> "TrainingMonitor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if user forgot to call close()
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
