"""Comet ML logging, with a no-op fallback when no API key is present.

Training must never depend on having credentials, so every method degrades to a
local-disk write (or nothing) when Comet is unavailable.  Visualisations are
also always written to ``<output_dir>/viz`` regardless, which is what you
actually end up looking at during a debugging session.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import configparser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class CometLogger:
    """Thin wrapper over ``comet_ml.Experiment``."""

    def __init__(self, api_key: str | None = None,
                 project_name: str = "echodiffusion",
                 workspace: str | None = None,
                 config: dict | None = None,
                 experiment_name: str | None = None,
                 tags: list[str] | None = None,
                 viz_dir: str | Path = "viz"):
        key = self._resolve_api_key(api_key)
        self.enabled = bool(key)
        self.experiment = None

        if self.enabled:
            try:
                from comet_ml import Experiment
                self.experiment = Experiment(
                    api_key=key, project_name=project_name, workspace=workspace)
                if experiment_name:
                    self.experiment.set_name(str(experiment_name))
                if tags:
                    self.experiment.add_tags([str(t) for t in tags])
                if config is not None:
                    self.experiment.log_parameters(_flatten(config))
            except Exception as exc:                     # noqa: BLE001
                # A broken Comet install or a network hiccup must not take the
                # run down after the data has already been loaded.
                print(f"[CometLogger] disabled ({type(exc).__name__}: {exc})")
                self.enabled, self.experiment = False, None
        else:
            print("[CometLogger] no API key found -- logging to disk only. "
                  "Set COMET_API_KEY, or logging.comet_api_key in the config.")

        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_api_key(api_key: str | None) -> str:
        if api_key:
            return api_key
        if os.environ.get("COMET_API_KEY"):
            return os.environ["COMET_API_KEY"]
        cfg_path = os.environ.get("COMET_CONFIG", str(Path.home() / ".comet.config"))
        parser = configparser.ConfigParser()
        try:
            if parser.read(cfg_path) and parser.has_option("comet", "api_key"):
                return parser.get("comet", "api_key", fallback="")
        except Exception:                                # noqa: BLE001
            pass
        return ""

    # ── scalars ───────────────────────────────────────────────────────────

    def log_metrics(self, metrics: dict, step: int | None = None,
                    epoch: int | None = None, prefix: str = "") -> None:
        if not self.enabled:
            return
        payload = {f"{prefix}{k}": float(v) for k, v in metrics.items()
                   if v is not None and np.isfinite(float(v))}
        self.experiment.log_metrics(payload, step=step, epoch=epoch)

    def log_parameters(self, params: dict) -> None:
        if self.enabled:
            self.experiment.log_parameters(_flatten(params))

    # ── figures ───────────────────────────────────────────────────────────

    def log_figure(self, name: str, fig, step: int | None = None) -> None:
        """Log to Comet and always mirror to ``viz_dir``."""
        path = self.viz_dir / f"{name}_{step if step is not None else 0:06d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        if self.enabled:
            self.experiment.log_figure(figure_name=name, figure=fig, step=step)
        plt.close(fig)

    def log_trajectory_samples(self, bev: np.ndarray, pred: np.ndarray,
                               target: np.ndarray, traj_scale: float,
                               field_range: float, step: int | None = None,
                               name: str = "val_samples",
                               max_items: int = 4,
                               source_xy: np.ndarray | None = None,
                               goal_scale: float = 8.0,
                               zoom: bool = True,
                               doa_bearing: np.ndarray | None = None,
                               doa_strength: np.ndarray | None = None) -> None:
        """Overlay trajectories and the heard sound direction on the BEV posterior.

        This is the main diagnostic figure: it puts the predicted path, the
        expert path, the *measured* DoA and the true source position in one
        frame, so "is the policy driving where it hears the sound?" is
        answerable at a glance.

        Args:
            bev: (B, C, H, W) field stack; channel 0 (the fused posterior) is drawn.
            pred / target: (B, horizon, 2) in normalised units.
            traj_scale: metres per normalised unit.
            field_range: half-extent of the BEV grid, for the axis extent.
            source_xy: (B, 2) normalised GT source position, if available.
            zoom: crop the view to the trajectories.  A 2 s path covers well
                under a metre while the field spans +/-6 m, so at full extent
                the trajectories collapse into the marker at the origin and the
                plot says nothing about prediction quality.  When the source
                falls outside the crop, an edge arrow points at it.
            doa_bearing: (B,) measured sound bearing in radians, robot frame.
                Drawn as a dashed ray -- this is what the model actually heard,
                as opposed to the GT source it was never shown.
            doa_strength: (B,) agreement in [0, 1]; sets the ray's opacity so a
                bearing averaged from contradictory detections looks weak.
        """
        n = min(max_items, bev.shape[0])
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9), squeeze=False)
        extent = [field_range, -field_range, -field_range, field_range]

        for i in range(n):
            ax = axes[0][i]
            # imshow rows run forward->back and cols left->right, matching the
            # field's own layout; ``extent`` relabels them in metres.
            ax.imshow(bev[i, 0], cmap="magma", vmin=0, vmax=1,
                      extent=extent, origin="upper")

            paths = []
            for traj, color, label in ((target[i], "#4ade80", "GT"),
                                       (pred[i], "#38bdf8", "pred")):
                xy = traj * traj_scale
                paths.append(xy)
                # Field axes are (y, x): y is the horizontal image axis.
                ax.plot(xy[:, 1], xy[:, 0], color=color, lw=2.0, label=label)
                ax.scatter(xy[-1, 1], xy[-1, 0], color=color, s=45,
                           marker="*", zorder=5)
            ax.scatter(0, 0, c="white", s=60, marker="^", zorder=6,
                       edgecolors="black", linewidths=0.6, label="robot")

            view = field_range
            if zoom:
                # Wide enough to keep some field texture visible for context,
                # tight enough that a sub-metre trajectory is still readable.
                reach = max(float(np.abs(np.concatenate(paths)).max()), 1e-3)
                view = min(max(2.0 * reach, 1.5), field_range)
                ax.set_xlim(view, -view)
                ax.set_ylim(-view, view)

            # Measured sound direction: a dashed ray from the robot, drawn
            # before the GT marker so the GT sits on top where they coincide.
            if doa_bearing is not None:
                theta = float(doa_bearing[i])
                strength = 1.0 if doa_strength is None else float(doa_strength[i])
                reach_r = view * 0.95
                ax.plot([0, np.sin(theta) * reach_r], [0, np.cos(theta) * reach_r],
                        color="#fbbf24", lw=2.2, ls="--",
                        alpha=0.35 + 0.65 * strength, zorder=4,
                        label=f"DoA heard ({np.degrees(theta):+.0f}°)")

            if source_xy is not None:
                sx, sy = np.asarray(source_xy[i], dtype=float) * goal_scale
                if abs(sx) <= view and abs(sy) <= view:
                    ax.scatter(sy, sx, c="#f87171", s=90, marker="X",
                               zorder=6, label="source (GT)")
                else:
                    # Outside the crop: draw an arrow from the origin toward it,
                    # so the direction is still readable.
                    norm = max(np.hypot(sx, sy), 1e-6)
                    ux, uy = sx / norm, sy / norm
                    ax.annotate(
                        "", xy=(uy * view * 0.88, ux * view * 0.88), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#f87171", lw=2.0),
                        zorder=6)
                    ax.plot([], [], color="#f87171", lw=2.0,
                            label=f"source GT ({norm:.1f} m)")

            ax.set_xlabel("y left (m)")
            ax.set_ylabel("x forward (m)")
            ax.set_title(f"sample {i}", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.75)
        fig.tight_layout()
        self.log_figure(name, fig, step=step)

    # ── assets ────────────────────────────────────────────────────────────

    def log_model(self, path: str | Path, name: str | None = None) -> None:
        if self.enabled:
            self.experiment.log_model(name or Path(path).stem, str(path))

    def end(self) -> None:
        if self.enabled:
            self.experiment.end()


def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    """Flatten nested config dicts -- Comet only accepts scalar parameters."""
    out: dict = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        elif isinstance(v, (list, tuple)):
            out[key] = str(list(v))
        else:
            out[key] = v
    return out
