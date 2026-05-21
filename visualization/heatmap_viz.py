import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

def visualize_surprise_heatmap(
    heatmap: np.ndarray,
    jump_targets: Optional[List[Tuple[int, int]]] = None,
    grid_meta: Optional[Tuple[int, int, int]] = None,
    gt_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "",
):

    n_plots = 1 + (1 if gt_mask is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    im = axes[0].imshow(heatmap, cmap="hot", interpolation="nearest", aspect='auto')
    axes[0].set_title(f"Surprise heatmap {title}", fontsize=11)
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    if jump_targets and grid_meta:
        x_min, y_min, step = grid_meta
        for i, (ox, oy) in enumerate(jump_targets):

            hx = (ox - x_min) / step
            hy = (oy - y_min) / step
            color = 'cyan' if i < 3 else 'lime'
            axes[0].plot(hx, hy, '*', color=color, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
            axes[0].annotate(f'{i+1}', (hx, hy), fontsize=7, color='white',
                           ha='center', va='bottom', fontweight='bold')

    if gt_mask is not None:
        axes[1].imshow(gt_mask, cmap="Greens", alpha=0.5, interpolation="nearest", aspect='auto')
        axes[1].imshow(heatmap, cmap="hot", alpha=0.5, interpolation="nearest", aspect='auto')
        axes[1].set_title("Surprise vs ground truth", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap saved to: {save_path}")
    plt.close()

def log_surprise_stats(yaad_memory, wsi_id: str = ""):

    stats = yaad_memory.get_stats()
    print(f"  [Yaad Stats] {wsi_id}: "
          f"patches={stats['count']}, "
          f"threshold={stats.get('threshold', 'N/A')}, "
          f"mean={stats.get('mean', 0):.4f}, "
          f"high_ratio={stats.get('high_surprise_ratio', 0):.1%}")
