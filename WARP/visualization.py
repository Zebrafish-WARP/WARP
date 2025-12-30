import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from typing import Dict, Mapping, Sequence, Optional, List, Callable, Literal

from distinctipy import get_colors

def set_mplstyle():
    import matplotlib as mpl
    import os

    # Directory where THIS file (the module) lives
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Full path to the mplstyle file
    style_path = os.path.join(module_dir, "rc_params.mplstyle")

    mpl.style.use(style_path)


def save_figure(fig, figure_dir, figure_string, dpi=1000, pad_inches=0.0):
    
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
        
    fig.savefig(figure_dir + f'/{figure_string}.svg', dpi=dpi, transparent=True, bbox_inches="tight")
    fig.savefig(figure_dir + f'/{figure_string}.jpg', dpi=dpi, bbox_inches='tight', pad_inches=pad_inches, transparent=False)
    fig.savefig(figure_dir + f'/{figure_string}.png', dpi=dpi, bbox_inches='tight', pad_inches=pad_inches, transparent=False)
    
    print(f'Succesfully saved figure to {figure_dir}')


def assign_subtype_colors(subtype_binary, subtype_names, distinct_colors=None):

    if distinct_colors is None:
        WHITE = (1, 1, 1)
        BLACK = (0, 0, 0)
        distinct_colors = [WHITE, BLACK]

    gene_ordering = np.argsort(np.sum(subtype_binary, axis=0))[::-1]
    n_subtypes = subtype_binary.shape[1]

    subtype_colors = {subtype_name: c for subtype_name, c in 
    zip(subtype_names[gene_ordering], get_colors(n_subtypes, distinct_colors, pastel_factor=0, rng=5))}
    return subtype_colors, gene_ordering


def get_brain_region_colors(brain_region_names=None):
    if brain_region_names is None:
        brain_region_names = ['InfMO', 'IntMO', 'SupMO', 'SupRaphe', 'Cb', 'Tg', 'NI', 'OTpv', 'OTnp', 'Pt', 
                            'preTh', 'Th', 'Hab', 'HypTh', 'SubP', 'Pal']
    n=len(brain_region_names)
    brain_region_colors = {region: c for region, c in zip(brain_region_names, plt.cm.nipy_spectral(np.linspace(1/n,1,n+1)))}
    return brain_region_colors
    

def fancy_chord_diagram_split_arcs(
    mat,
    labels=None,
    cmap='coolwarm',
    threshold=0.0,
    curvature=0.5,
    offset=0.08,
    arrow=True,
    alpha=0.9,
    alpha_power=1.0,
    node_cmap='tab10',
    gap=0.02,
    min_arc_fraction=0.01,
    linewidth=2.0,
    figsize=(8,8),
    label_fontsize=11,
    label_fontweight='medium',
    label_color=None,
    chord_pad=0.05,
    arc_radius=1.0,
    label_pad=0.1,
    ax=None,
    cbar_fraction=0.046,
    cbar_pad=0.04,
    cbar_label='Value',
    cbar_labelsize=11,
    self_arc_fraction=0.2,
    self_curve=0.3
):
    """
    Directed chord diagram with split arcs, curved self-connections,
    radial labels, colormap normalized around 0, and optional ax input.
    """
    n = mat.shape[0]
    assert mat.shape[1] == n, "Matrix must be square."

    # --- Node totals ---
    mat_abs = np.abs(mat)
    outgoing_totals = np.nansum(mat_abs, axis=1)
    incoming_totals = np.nansum(mat_abs, axis=0)
    totals = outgoing_totals + incoming_totals
    totals = np.maximum(totals, 1e-8)

    # --- Arc lengths ---
    min_arc = min_arc_fraction * 2*np.pi
    total_min = n*min_arc
    if total_min >= 2*np.pi - n*gap:
        raise ValueError("min_arc_fraction too large for number of nodes and gap size.")
    available_angle = 2*np.pi - n*gap - total_min
    totals_scaled = totals / np.sum(totals)
    arc_lengths = min_arc + totals_scaled * available_angle

    # --- Start/end angles ---
    start_angles = np.zeros(n)
    end_angles = np.zeros(n)
    cumulative_angle = 0
    for i in range(n):
        start_angles[i] = cumulative_angle
        end_angles[i] = cumulative_angle + arc_lengths[i]
        cumulative_angle = end_angles[i] + gap
    mid_angles = (start_angles + end_angles)/2

    # --- Node colors ---
    node_colors = get_cmap(node_cmap)(np.linspace(0,1,n))

    # --- Chord color normalization around 0 ---
    vmax_color = np.nanpercentile(np.abs(mat), 99)
    if vmax_color == 0: vmax_color = np.nanmax(abs(mat))
    vmin_color = -vmax_color
    norm = TwoSlopeNorm(vmin=vmin_color, vcenter=0, vmax=vmax_color)
    edge_cmap = get_cmap(cmap)
    abs_vmax = np.nanmax(mat_abs)
    abs_vmax = 1.0 if abs_vmax==0 or np.isnan(abs_vmax) else abs_vmax

    # --- Figure / ax setup ---
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect':'equal'})
        own_fig = True
    ax.axis('off')
    ax.set_aspect('equal')

    # --- Draw arcs and labels ---
    for i in range(n):
        start, end = start_angles[i], end_angles[i]
        mid = (start + end)/2

        # Outgoing half (left)
        arc_out = Arc((0,0), 2*arc_radius, 2*arc_radius,
                      theta1=np.degrees(start), theta2=np.degrees(mid),
                      lw=10, color=node_colors[i], zorder=5, alpha=0.9)
        ax.add_patch(arc_out)

        # Incoming half (right)
        arc_in = Arc((0,0), 2*arc_radius, 2*arc_radius,
                     theta1=np.degrees(mid), theta2=np.degrees(end),
                     lw=10, color=node_colors[i], zorder=5, alpha=0.9)
        ax.add_patch(arc_in)

        # --- Radial labels ---
        if labels is not None:
            angle = mid
            x = (arc_radius + label_pad) * np.cos(angle)
            y = (arc_radius + label_pad) * np.sin(angle)
            rotation = np.degrees(angle)
            if np.cos(angle)<0:
                rotation += 180
            ha = 'left' if np.cos(angle)>=0 else 'right'
            va = 'bottom' if np.sin(angle)>=0 else 'top'
            color = node_colors[i] if label_color is None else label_color
            ax.text(x, y, labels[i],
                    ha=ha, va=va,
                    fontsize=label_fontsize,
                    fontweight=label_fontweight,
                    color=color,
                    rotation=rotation,
                    rotation_mode='anchor')

    # --- Prepare chords ---
    chords = []
    for i in range(n):
        for j in range(n):
            val = mat[i,j]
            if np.isnan(val) or abs(val)<threshold:
                continue

            if i == j:
                # Self-connection: curved outward loop
                start_angle = (start_angles[i] + end_angles[i])/2 - self_arc_fraction*np.pi/2
                end_angle   = (start_angles[i] + end_angles[i])/2 + self_arc_fraction*np.pi/2
                mid_angle   = (start_angle + end_angle)/2

                # Start and end points on arc
                x_start = (arc_radius - chord_pad) * np.cos(start_angle)
                y_start = (arc_radius - chord_pad) * np.sin(start_angle)
                x_end   = (arc_radius - chord_pad) * np.cos(end_angle)
                y_end   = (arc_radius - chord_pad) * np.sin(end_angle)

                # Control point outward along radial perpendicular
                cx = (x_start + x_end)/2 + self_curve*np.cos(mid_angle)
                cy = (y_start + y_end)/2 + self_curve*np.sin(mid_angle)

                chords.append((x_start, y_start, cx, cy, x_end, y_end, val, True))
            else:
                t_i = np.random.uniform(start_angles[i], (start_angles[i]+end_angles[i])/2)
                t_j = np.random.uniform((start_angles[j]+end_angles[j])/2, end_angles[j])
                x1, y1 = (arc_radius - chord_pad) * np.cos(t_i), (arc_radius - chord_pad) * np.sin(t_i)
                x2, y2 = (arc_radius - chord_pad) * np.cos(t_j), (arc_radius - chord_pad) * np.sin(t_j)
                chords.append((x1, y1, x2, y2, val, False))

    # --- Sort by alpha ---
    def chord_alpha(c):
        if c[-1]:
            return abs(c[6])/abs_vmax
        else:
            return abs(c[4])/abs_vmax
    chords.sort(key=chord_alpha, reverse=True)

    # --- Draw chords ---
    for chord in chords:
        if chord[-1]:  # self-connection
            x_start, y_start, cx, cy, x_end, y_end, val, _ = chord
            color = edge_cmap(norm(val))
            mag_norm = abs(val)/abs_vmax
            alpha_scaled = alpha * (mag_norm**alpha_power)

            verts = [(x_start, y_start), (cx, cy), (x_end, y_end)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)

            patch = FancyArrowPatch(
                path=path,
                arrowstyle='-|>' if arrow else '-',
                mutation_scale=10 + 6*mag_norm,
                lw=linewidth,
                color=color,
                alpha=alpha_scaled,
                zorder=2
            )
            ax.add_patch(patch)
        else:
            x1, y1, x2, y2, val, _ = chord
            color = edge_cmap(norm(val))
            mag_norm = abs(val)/abs_vmax
            alpha_scaled = alpha * (mag_norm**alpha_power)

            mx, my = (x1+x2)/2, (y1+y2)/2
            vx, vy = -(y2 - y1), (x2 - x1)
            vnorm = np.hypot(vx, vy)
            vx, vy = vx/vnorm, vy/vnorm
            direction = np.sign(val)
            cx = curvature*mx + offset*direction*vx
            cy = curvature*my + offset*direction*vy

            verts = [(x1,y1),(cx,cy),(x2,y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)

            patch = FancyArrowPatch(
                path=path,
                arrowstyle='-|>' if arrow else '-',
                mutation_scale=10 + 6*mag_norm,
                lw=linewidth,
                color=color,
                alpha=alpha_scaled,
                zorder=2
            )
            ax.add_patch(patch)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(norm=norm, cmap=edge_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize)

    ax.set_xlim(-1.6,1.6)
    ax.set_ylim(-1.6,1.6)
    ax.set_aspect('equal')
    if own_fig:
        plt.tight_layout()
        plt.show()




import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def brain_chord_map_with_neurons(
    mat,
    node_coords,
    all_neuron_coords=None,
    labels=None,
    label_fontsize=8,
    cmap='coolwarm',
    threshold=0.0,
    curvature=0.2,
    offset=0.05,
    arrow=True,
    alpha=0.9,
    alpha_power=1.0,
    node_cmap='tab10',
    linewidth=2.0,
    figsize=(6,6),
    self_curve=0.2,
    projection=(0,1),
    ax=None,
    cbar_fraction=0.046,
    cbar_pad=0.04,
    cbar_label='Value',
    cbar_labelsize=11,
    neuron_color='gray',
    neuron_alpha=0.01,
    neuron_size=1,
    node_size=10
):
    """
    Draw brain map with neurons in the background and chords between region nodes.
    
    Parameters
    ----------
    all_neuron_coords : np.ndarray
        (N x 3) coordinates of all neurons. Plotted as background points.
    projection : tuple
        Indices of coordinates to plot, e.g., (0,1)=x-y.
    """
    n = mat.shape[0]
    assert mat.shape[1] == n, "Matrix must be square."
    
    # Node colors
    node_colors = get_cmap(node_cmap)(np.linspace(0,1,n))
    
    # Chord color normalization
    vmax_color = np.nanpercentile(np.abs(mat), 99)
    if vmax_color == 0: vmax_color = np.nanmax(abs(mat))
    vmin_color = -vmax_color
    norm = TwoSlopeNorm(vmin=vmin_color, vcenter=0, vmax=vmax_color)
    edge_cmap = get_cmap(cmap)
    abs_vmax = np.nanmax(np.abs(mat))
    abs_vmax = 1.0 if abs_vmax==0 or np.isnan(abs_vmax) else abs_vmax
    
    # Figure / ax
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Projected coordinates
    node_xy = node_coords[:, projection]
    
    # --- Background neurons ---
    if all_neuron_coords is not None:
        neuron_xy = all_neuron_coords[:, projection]
        ax.scatter(neuron_xy[:,0], neuron_xy[:,1],
                   s=neuron_size, color=neuron_color, alpha=neuron_alpha, zorder=0)
    
    # --- Draw nodes ---
    for i in range(n):
        x, y = node_xy[i]
        ax.scatter(x, y, s=node_size, color=node_colors[i], zorder=5)
        if labels is not None:
            ax.text(x, y, labels[i],
                    ha='center', va='center',
                    fontsize=label_fontsize, fontweight='bold', color='black')
    
    # --- Prepare chords ---
    chords = []
    for i in range(n):
        for j in range(n):
            val = mat[i,j]
            if np.isnan(val) or abs(val)<threshold:
                continue
            x1, y1 = node_xy[i]
            x2, y2 = node_xy[j]
            if i == j:
                # self-loop
                mx, my = x1, y1
                cx, cy = mx + self_curve, my + self_curve
                chords.append((x1, y1, cx, cy, x2, y2, val, True))
            else:
                # curved connection
                mx, my = (x1 + x2)/2, (y1 + y2)/2
                vx, vy = -(y2 - y1), (x2 - x1)
                vnorm = np.hypot(vx, vy)
                vx, vy = vx/vnorm, vy/vnorm
                direction = np.sign(val)
                cx, cy = mx + curvature*mx + offset*direction*vx, my + curvature*my + offset*direction*vy
                chords.append((x1, y1, cx, cy, x2, y2, val, False))
    
    # --- Draw chords sorted by magnitude ---
    chords.sort(key=lambda c: abs(c[6]), reverse=True)
    for chord in chords:
        x1, y1, cx, cy, x2, y2, val, self_loop = chord
        color = edge_cmap(norm(val))
        mag_norm = abs(val)/abs_vmax
        alpha_scaled = alpha * (mag_norm**alpha_power)
        verts = [(x1, y1), (cx, cy), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = FancyArrowPatch(
            path=path,
            arrowstyle='-|>' if arrow else '-',
            mutation_scale=10 + 6*mag_norm,
            lw=linewidth,
            color=color,
            alpha=alpha_scaled,
            zorder=2
        )
        ax.add_patch(patch)
    
    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(norm=norm, cmap=edge_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize)
    
    if own_fig:
        plt.show()


def make_cmap_dict(mapping_strings, sort_inds=None, distinct_colors=None):
    from distinctipy import get_colors
    import numpy as np

    if distinct_colors is None:
        WHITE = (1, 1, 1)
        BLACK = (0, 0, 0)
        distinct_colors = [WHITE, BLACK]

    if sort_inds is None:
        sort_inds = np.arange(len(mapping_strings))
    
    colors = get_colors(len(mapping_strings), distinct_colors, pastel_factor=0, rng=5)
    cmap_dict = {mapping_string: colors[i] for i, mapping_string in enumerate(mapping_strings)}
    return cmap_dict


def _draw_cut_mark(ax, where: str = "top", dx: float = 0.02, dy: float = 0.03):
    """
    Draw a small 'cut' marker at the top or bottom of an axis, in axes coordinates.

    Parameters
    ----------
    ax
        Matplotlib Axes to draw onto.
    where
        'top' or 'bottom'.
    dx, dy
        Size of the diagonal segments in axes coordinates.
    """
    if where not in ("top", "bottom"):
        return

    # y coordinate in axes fraction
    if where == "top":
        y = 1.0
        sgn = 1.0
    else:
        y = 0.0
        sgn = -1.0

    # A few x positions across the width
    xs = [0.15, 0.35, 0.55, 0.75]

    for x in xs:
        ax.plot(
            [x - dx, x, x + dx],
            [y - sgn * dy, y, y - sgn * dy],
            transform=ax.transAxes,
            clip_on=False,
            color="k",
            linewidth=0.7,
        )


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Mapping, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Mapping, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from typing import Mapping, Optional, Sequence


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Mapping, Sequence, Optional, Dict, Any, Tuple


def compute_lcd_color_dict(
    cluster_dict: Dict[str, Dict[str, Any]],
    highN: Optional[int] = 12,
    lowN: Optional[int] = 3,
    color_mode: str = "dict",          # "dict", "median_highlow", "median_all", "discrete"
    base_color_dict: Optional[Mapping[str, Sequence[float]]] = None,
    median_cmap: str = "coolwarm",     # e.g. "coolwarm", "bwr"
    grey_color: Sequence[float] = (0.8, 0.8, 0.8, 1.0),
    pooled_color: Sequence[float] = (0.2, 0.2, 0.2, 1.0),
    include_pooled_row: bool = True,
    stim_key: str = "visrap"
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a color dictionary for LCD clusters.

    Parameters
    ----------
    cluster_dict
        Mapping cluster_id -> cluster summary dict, where each summary contains
        at least 'vals' with shape (N, 3) or (N,) for that cluster.
        Column 0 of 'vals' is assumed to be the LCD difference used here.
    highN, lowN
        Number of highest / lowest median entries used for:
        - defining "high" and "low" sets (for some color modes),
        - rank-based discrete coloring.
    color_mode
        "dict":
            Use `base_color_dict` directly (if provided), otherwise fallback to tab20.
        "median_highlow":
            Only highN and lowN entries colored by `median_cmap` using their median LCD
            and a symmetric TwoSlopeNorm around 0. Others get `grey_color`.
        "median_all":
            All entries colored via `median_cmap` based on median LCD.
        "discrete":
            HighN and lowN entries colored by rank along the colormap:
            - lowN on the negative side (blue-ish for coolwarm),
            - highN on the positive side (red-ish).
            Others are `grey_color`.

    Returns
    -------
    output_dict
        Mapping cluster_id (and optionally "all clusters") -> dict with:
          - 'color' : RGBA color
          - 'name'  : label (string, e.g. "3 (gad1b)")
          - 'order' : integer ordering index (0 if unused)
    """
    # -------------------------------------------------------------------------
    # Collect cluster names + medians
    # -------------------------------------------------------------------------
    cluster_names = np.array(list(cluster_dict))
    n_clusters = len(cluster_names)

    # Per-cluster LCD values: assume cluster_dict[c]['vals'][:, 0] is LCD diff
    per_cluster_vals = []
    for c in cluster_names:
        v = np.asarray(cluster_dict[c]["lcd_vals"][stim_key])
        if v.ndim == 1:
            per_cluster_vals.append(v)
        else:
            per_cluster_vals.append(v[:, 0])

    vals_diff_medians = np.array([np.nanmedian(v) for v in per_cluster_vals])

    # Sort by median
    sort_inds = np.argsort(vals_diff_medians)      # indices into cluster_names

    # Identify high and low sets (indices into cluster_names)
    high_inds = np.array([], dtype=int)
    low_inds = np.array([], dtype=int)

    if highN is not None and highN > 0:
        high_inds = sort_inds[-min(highN, n_clusters):]
    if lowN is not None and lowN > 0:
        low_inds = sort_inds[:min(lowN, n_clusters)]

    # -------------------------------------------------------------------------
    # Initialize unified output dict
    # -------------------------------------------------------------------------
    output_dict: Dict[str, Dict[str, Any]] = {
        name: {"color": None, "name": None, "order": 0} for name in cluster_names
    }

    # -------------------------------------------------------------------------
    # Color assignment
    # -------------------------------------------------------------------------
    if color_mode == "dict":
        # Use base_color_dict if given, otherwise fallback to tab20
        cmap_fallback = plt.get_cmap("tab20")
        for i, name in enumerate(cluster_names):
            if base_color_dict is not None and name in base_color_dict:
                output_dict[name]["color"] = base_color_dict[name]
            else:
                output_dict[name]["color"] = cmap_fallback(i % cmap_fallback.N)
            # Label is just the cluster_id, order unused
            output_dict[name]["name"] = name
            output_dict[name]["order"] = 0

    else:
        if color_mode not in ("median_all", "median_highlow", "discrete"):
            raise ValueError(
                f"Unknown color_mode='{color_mode}'. "
                "Use 'dict', 'median_highlow', 'median_all', or 'discrete'."
            )

        cmap = plt.get_cmap(median_cmap)
        neutral_band = 0.16
        low_end = 0.5 - neutral_band / 2.0
        high_end = 0.5 + neutral_band / 2.0

        # ---------------------------------------------------------------------
        # median_all / median_highlow modes
        # ---------------------------------------------------------------------
        if color_mode in ("median_all", "median_highlow"):
            # Which subset determines the scale?
            if color_mode == "median_all":
                scale_inds = np.arange(n_clusters, dtype=int)
            else:
                # Only use high+low subset for scaling
                if high_inds.size == 0 and low_inds.size == 0:
                    scale_inds = np.array([], dtype=int)
                else:
                    scale_inds = np.unique(np.concatenate([high_inds, low_inds]))

            if scale_inds.size == 0:
                # No usable medians
                for name in cluster_names:
                    output_dict[name]["color"] = grey_color
                    output_dict[name]["name"] = "none"
                    output_dict[name]["order"] = 0
            else:
                med_subset = vals_diff_medians[scale_inds]
                max_abs = np.nanmax(np.abs(med_subset))
                if not np.isfinite(max_abs) or max_abs <= 0:
                    for name in cluster_names:
                        output_dict[name]["color"] = grey_color
                        output_dict[name]["name"] = "none"
                        output_dict[name]["order"] = 0
                else:
                    vmin, vmax = -max_abs, max_abs
                    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

                    def median_to_color(med: float):
                        if not np.isfinite(med):
                            return grey_color
                        v = norm(med)  # 0..1
                        if not np.isfinite(v):
                            return grey_color

                        # Push away from the neutral center (0.5)
                        if v < 0.5:
                            if 0.5 > 0:
                                v = v * (low_end / 0.5)
                        elif v > 0.5:
                            if 1.0 > 0.5:
                                v = 1.0 - (1.0 - v) * ((1.0 - high_end) / 0.5)

                        return cmap(np.clip(v, 0.0, 1.0))

                    if color_mode == "median_all":
                        # Color *all* clusters by their median
                        for i, name in enumerate(cluster_names):
                            output_dict[name]["color"] = median_to_color(
                                vals_diff_medians[i]
                            )
                            output_dict[name]["name"] = name
                            output_dict[name]["order"] = 0
                    else:
                        # median_highlow: default grey, color only extremes
                        for name in cluster_names:
                            output_dict[name]["color"] = grey_color
                            output_dict[name]["name"] = "none"
                            output_dict[name]["order"] = 0

                        # Low side
                        if low_inds.size > 0:
                            for rank, idx in enumerate(low_inds, start=1):
                                name = cluster_names[idx]
                                output_dict[name]["color"] = median_to_color(
                                    vals_diff_medians[idx]
                                )
                                output_dict[name]["name"] = (
                                    f"{rank} ({name.split('_')[0]})"
                                )
                                output_dict[name]["order"] = rank

                        # High side
                        if high_inds.size > 0:
                            offset = low_inds.size
                            for rank, idx in enumerate(high_inds, start=1):
                                name = cluster_names[idx]
                                out_rank = rank + offset
                                output_dict[name]["color"] = median_to_color(
                                    vals_diff_medians[idx]
                                )
                                output_dict[name]["name"] = (
                                    f"{out_rank} ({name.split('_')[0]})"
                                )
                                output_dict[name]["order"] = out_rank

        # ---------------------------------------------------------------------
        # discrete mode (your previous logic, generalized into output_dict)
        # ---------------------------------------------------------------------
        else:
            # color_mode == "discrete"
            # default grey
            for name in cluster_names:
                output_dict[name]["color"] = grey_color
                output_dict[name]["name"] = "none"
                output_dict[name]["order"] = 0

            n_low = low_inds.size
            n_high = high_inds.size

            if not (n_low == 0 and n_high == 0):
                # Low side: positions from 0 -> low_end
                if n_low > 0:
                    low_positions = np.linspace(0.0, low_end, n_low, endpoint=True)
                    for i, (idx, pos) in enumerate(zip(low_inds, low_positions)):
                        name = cluster_names[idx]
                        output_dict[name]["color"] = cmap(np.clip(pos, 0.0, 1.0))
                        output_dict[name]["name"] = (
                            f"{i+1} ({name.split('_')[0]})"
                        )
                        output_dict[name]["order"] = i + 1

                # High side: positions from high_end -> 1
                if n_high > 0:
                    high_positions = np.linspace(high_end, 1.0, n_high, endpoint=True)
                    for i, (idx, pos) in enumerate(zip(high_inds, high_positions)):
                        name = cluster_names[idx]
                        output_dict[name]["color"] = cmap(np.clip(pos, 0.0, 1.0))
                        output_dict[name]["name"] = (
                            f"{i+1+n_clusters-n_high} ({name.split('_')[0]})"
                        )
                        output_dict[name]["order"] = i + 1 + n_low

    # -------------------------------------------------------------------------
    # Optional pooled "all clusters"
    # -------------------------------------------------------------------------
    if include_pooled_row:
        pooled_name = "all clusters"
        output_dict[pooled_name] = {
            "color": pooled_color,
            "name": "all clusters",
            "order": 0,
        }

    return output_dict



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Mapping, Optional, Sequence
from scipy.stats import gaussian_kde


def plot_lcd_corr_scatter(
    lcd_vals: Mapping[str, np.ndarray],
    save_dir: str = None,
    save_name: str = None,
    include_hist: bool = True,
    contour: bool = True,
    kde_step: float = 0.05,
    bins_2d: int = 75,
    cmap: str = "inferno",
    contour_levels: Sequence[float] = (0.1, 0.5, 5.0),
    xlim: Sequence[float] = (-0.5, 1.0),
    ylim: Sequence[float] = (-0.5, 1.0),
    diff_range: Sequence[float] = (-0.2, 0.2),
    diff_bins: int = 15,
    hist_extent_frac: float = 0.10,
    show: bool = True,
    xlabel: str = '',
    ylabel: str = ''
) -> plt.Figure:
    """
    Plot a 2D histogram of:
        x = correlations with remaining neighbors (corr_other)
        y = correlations with gene-matching neighbors (corr_same)

    Optionally:
      - overlay KDE contours (contour=True)
      - overlay a rotated histogram of the difference (remaining - matching)
        along an axis perpendicular to y=x, anchored near the top-right
        of the plot (include_hist=True).

    Geometry is done in main-axis *data* coordinates:
    - Difference axis: (x, y) = (c + d/2, c - d/2) so x - y = d.
      Here c is chosen close to the top-right corner in data units.
    - Bars extend *inward* from the corner along (-1, -1)/sqrt(2).

    Parameters
    ----------
    lcd_vals : mapping
        Dict: gene -> array of shape (N_neurons, 3),
        with columns [LCD_diff, corr_same, corr_other].
    save_figure : callable or None
        Optional function with signature save_figure(save_dir, save_name).
    save_dir : str
        Directory passed to save_figure if provided.
    save_name : str
        Filename stem passed to save_figure.
    include_hist : bool
        If True, overlay the rotated histogram of (corr_other - corr_same).
    contour : bool
        If True, overlay KDE contours on top of the 2D histogram.
    kde_step : float
        Grid step for KDE grid when contour=True.
    bins_2d : int
        Number of bins per axis for the 2D histogram.
    cmap : str
        Colormap for the 2D histogram (default: 'inferno').
    contour_levels : sequence of float
        Levels for KDE contour.
    xlim, ylim : [min, max]
        Axis limits for the main plot.
    diff_range : (min, max)
        Range of the difference (remaining - matching) shown along the
        rotated axis.
    diff_bins : int
        Number of bins for the difference histogram.
    hist_extent_frac : float
        Fraction of the main axis range used to scale the histogram "height"
        inward from the corner.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # ------------------------------------------------------------------
    # 1. Collect x/y from lcd_vals
    # ------------------------------------------------------------------
    x_list = []
    y_list = []

    for vals in lcd_vals.values():
        vals = np.asarray(vals)
        if vals.ndim != 2 or vals.shape[1] < 3:
            continue

        corr_same = vals[:, 1]
        corr_other = vals[:, 2]

        mask = ~(np.isnan(corr_same) | np.isnan(corr_other))
        if np.any(mask):
            x_list.append(corr_other[mask])  # remaining
            y_list.append(corr_same[mask])   # matching

    if not x_list:
        raise ValueError("No valid correlation values found in lcd_vals.")

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    # Difference = remaining - matching
    diff = x - y
    diff = diff[~np.isnan(diff)]

    # ------------------------------------------------------------------
    # 2. KDE for contour (optional)
    # ------------------------------------------------------------------
    if contour:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)

        Xg, Yg = np.mgrid[xmin:xmax:kde_step, ymin:ymax:kde_step]
        positions = np.vstack([Xg.ravel(), Yg.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        Zg = np.reshape(kernel(positions).T, Xg.shape)
    else:
        Xg = Yg = Zg = None

    # ------------------------------------------------------------------
    # 3. Main figure + 2D histogram
    #    Reserve some space on the right for a separate colorbar.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(1.6, 1.6))
    # Main axis on the left
    ax = fig.add_axes([0.10, 0.10, 0.68, 0.80])  # [left, bottom, width, height]

    x_bins = np.linspace(np.nanmin(x), np.nanmax(x), bins_2d)
    y_bins = np.linspace(np.nanmin(y), np.nanmax(y), bins_2d)

    h = ax.hist2d(
        x,
        y,
        bins=(x_bins, y_bins),
        norm=mcolors.LogNorm(),
        cmap=cmap,
        rasterized=True
    )

    # Colorbar on the right
    cax = fig.add_axes([0.82, 0.20, 0.03, 0.40])
    cbar = fig.colorbar(h[3], cax=cax)
    cbar.set_label("Neurons", fontsize=7)

    if contour and Xg is not None:
        ax.contour(
            Xg,
            Yg,
            Zg,
            levels=np.array(contour_levels),
            colors=["w"] * len(contour_levels),
        )

    # y = x line
    ax.plot([-10, 10], [-10, 10], linestyle=(0, (1, 3)), color="r", lw=1)

    # Axes lines at 0
    ax.axhline(0, linestyle=(0, (1, 3)), color="k", lw=1)
    ax.axvline(0, linestyle=(0, (1, 3)), color="k", lw=1)

    # Limits & aspect
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)

    # ------------------------------------------------------------------
    # 4. Rotated histogram of (remaining - matching), anchored at top-right
    # ------------------------------------------------------------------
    if include_hist and diff.size > 0:
        diff_min, diff_max = diff_range
        bins = np.linspace(diff_min, diff_max, diff_bins)

        counts, bin_edges = np.histogram(diff, bins=bins, density=True)
        if np.all(counts == 0):
            counts = np.ones_like(counts)

        # Scale density so histogram doesn't cover the whole plot
        axis_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        density_scale = hist_extent_frac * axis_range / np.nanmax(counts)

        # Anchor c close to the top-right corner in data coordinates.
        diag_max = min(1, 1)
        corner_margin = 0.03 * axis_range  # tiny offset from [1,1]
        c = diag_max - corner_margin

        # Bars extend *inward* from the corner along (-1, -1)/sqrt(2)
        bar_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)

        # Base difference axis
        d_line = np.linspace(diff_min, diff_max, 100)
        x_line = c + d_line / 2.0
        y_line = c - d_line / 2.0
        ax.plot(
            x_line,
            y_line,
            color="k",
            lw=0.7,
            alpha=0.7,
            linestyle="-",
        )

        # Bars
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for d_val, count in zip(bin_centers, counts):
            if count <= 0:
                continue

            # Base point on difference axis
            base_x = c + d_val / 2.0
            base_y = c - d_val / 2.0

            # Bar endpoint inward from the corner
            length = density_scale * count
            end_x = base_x + length * bar_dir[0]
            end_y = base_y + length * bar_dir[1]

            ax.plot(
                [base_x, end_x],
                [base_y, end_y],
                color=(.6, .6, .6),
                linewidth=1.0,
            )

        # # Median marker
        # median_diff = float(np.nanmean(diff))
        # base_x_med = c + median_diff / 2.0
        # base_y_med = c - median_diff / 2.0
        # med_x = base_x_med + 1.1 * density_scale * 0.3 * bar_dir[0]
        # med_y = base_y_med + 1.1 * density_scale * 0.3 * bar_dir[1]
        # ax.scatter(
        #     [med_x],
        #     [med_y],
        #     marker=(3, 0, -90),
        #     color="r",
        #     s=12,
        #     zorder=5,
        # )

        # Tick labels along the difference axis (e.g. -0.2, 0, 0.2)
        tick_vals = np.linspace(diff_min, diff_max, 3)
        for tv in tick_vals:
            tx = c + tv / 2.0
            ty = c - tv / 2.0
            ax.text(
                tx,
                ty,
                f"{tv:.1f}",
                fontsize=6,
                ha="center",
                va="center",
                rotation=-45,
                rotation_mode="anchor",
                color="k",
            )

    # ------------------------------------------------------------------
    # 5. Save / show
    # ------------------------------------------------------------------
    if save_dir is not None:
        save_figure(fig, save_dir, save_name)

    if show:
        plt.show()

    return fig


import matplotlib.gridspec as gridspec


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Mapping, Optional, Sequence


def plot_lcd_distributions(
    lcd_vals: Mapping[str, np.ndarray],
    lcd_pvals: Mapping[str, float],
    color_dict: Optional[Mapping[str, Sequence[float]]] = None,
    save_figure: Optional[callable] = None,
    save_dir: str = "./Figure_Panels/",
    save_name: str = "LCD_gene_distributions",
    highN: Optional[int] = 12,
    lowN: Optional[int] = 3,
    filter_percentile: float = 2.5,
    bw_method: float = .2,
    include_pooled_row: bool = True,
    show_all_genes: bool = False,
    pooled_color: Sequence[float] = (0.2, 0.2, 0.2, 1.0),
    show: bool = True,
    ylabel: str = "Gene",
    name_dict=None,
    # ---------------------- NEW (optional) ----------------------
    lcd_vals2: Optional[Mapping[str, np.ndarray]] = None,
    lcd_pvals2: Optional[Mapping[str, float]] = None,
) -> plt.Figure:
    """
    Plot LCD distributions per gene/cluster, optionally highlighting extremes
    and/or all entries. Automatically sets x-axis ranges based on the data.

    NEW: If `lcd_vals2` is provided, we draw a split representation:
      - top half of each violin row = lcd_vals (set 1)
      - bottom half = lcd_vals2 (set 2)
    The neighbor-avg heatmap and median±SEM panel also get top/bottom rows.
    """

    def _nice_limits_and_ticks(lo: float, hi: float, round_to: float = 0.05):
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -0.1, 0.1
        data_range = hi - lo
        pad = 0.05 * data_range if data_range > 0 else 0.01
        x_min = lo - pad
        x_max = hi + pad
        x_min_round = np.floor(x_min / round_to) * round_to
        x_max_round = np.ceil(x_max / round_to) * round_to
        x_min_round = min(x_min_round, 0.0)
        x_max_round = max(x_max_round, 0.0)
        xticks = [x_min_round, 0.0, x_max_round]
        return x_min_round, x_max_round, xticks

    def _nansem(v: np.ndarray) -> float:
        v = np.asarray(v)
        n = np.sum(np.isfinite(v))
        if n <= 1:
            return np.nan
        return np.nanstd(v) / np.sqrt(n)

    def _clip_violin_half(polycoll, y_center: float, which: str):
        """
        Matplotlib violin bodies are polygons. We "half-clip" by flattening
        vertices on the forbidden side onto the center line.
        """
        for path in polycoll.get_paths():
            verts = path.vertices
            if which == "top":
                verts[:, 1] = np.maximum(verts[:, 1], y_center)
            elif which == "bottom":
                verts[:, 1] = np.minimum(verts[:, 1], y_center)
            else:
                raise ValueError("which must be 'top' or 'bottom'")

    dual = lcd_vals2 is not None

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------
    gene_names = np.array(sorted(lcd_vals.keys()))
    if name_dict is None:
        name_dict = {g: g for g in gene_names}

    n_genes = len(gene_names)

    # Set 1
    per_gene_vals1 = [np.asarray(lcd_vals[g][:, 0]) for g in gene_names]
    vals_diff1 = np.array(per_gene_vals1, dtype=object)
    all_vals1 = np.concatenate(per_gene_vals1) if len(per_gene_vals1) > 0 else np.array([])
    vals_diff1 = np.array(list(vals_diff1) + [all_vals1], dtype=object)

    p_vals1 = np.array([lcd_pvals.get(g, np.nan) for g in gene_names], dtype=float)
    # --- CHANGED: allow lcd_pvals['pooled'] to define pooled row p-value ---
    pooled_pval1 = lcd_pvals.get("pooled", np.nan)
    p_vals_full1 = np.append(p_vals1, pooled_pval1)

    # Set 2 (optional; aligned to set 1 gene list)
    if dual:
        per_gene_vals2 = []
        for g in gene_names:
            if g in lcd_vals2:
                per_gene_vals2.append(np.asarray(lcd_vals2[g][:, 0]))
            else:
                per_gene_vals2.append(np.array([], dtype=float))
        vals_diff2 = np.array(per_gene_vals2, dtype=object)
        all_vals2 = np.concatenate(per_gene_vals2) if len(per_gene_vals2) > 0 else np.array([])
        vals_diff2 = np.array(list(vals_diff2) + [all_vals2], dtype=object)

        if lcd_pvals2 is None:
            p_vals_full2 = np.full(n_genes + 1, np.nan, dtype=float)
        else:
            p_vals2 = np.array([lcd_pvals2.get(g, np.nan) for g in gene_names], dtype=float)
            # --- CHANGED: allow lcd_pvals2['pooled'] to define pooled row p-value ---
            pooled_pval2 = lcd_pvals2.get("pooled", np.nan)
            p_vals_full2 = np.append(p_vals2, pooled_pval2)
    else:
        vals_diff2 = None
        all_vals2 = None
        p_vals_full2 = None

    pooled_name = "all " + ylabel.lower() + "s"
    gene_names_tot = np.concatenate((gene_names, [pooled_name]))
    name_dict[pooled_name] = pooled_name
    pooled_index = n_genes

    # Neighbor averages (heatmap)
    # Each row is [neighbors, gene-matched] after flip, as in your original code.
    same_other_neighbor_avgs1 = np.stack(
        [np.flip(np.mean(lcd_vals[g][:, 1:], axis=0)) for g in gene_names]
    )
    all_gene_same_other_neighbor_avgs1 = np.mean(
        np.vstack([np.flip(lcd_vals[g][:, 1:]) for g in gene_names]), axis=0
    )
    same_other_neighbor_avgs_full1 = np.vstack(
        [same_other_neighbor_avgs1, all_gene_same_other_neighbor_avgs1]
    )

    if dual:
        same_other_neighbor_avgs2_rows = []
        for g in gene_names:
            if g in lcd_vals2 and lcd_vals2[g].shape[1] > 1:
                same_other_neighbor_avgs2_rows.append(
                    np.flip(np.mean(lcd_vals2[g][:, 1:], axis=0))
                )
            else:
                same_other_neighbor_avgs2_rows.append(np.full_like(same_other_neighbor_avgs1[0], np.nan))
        same_other_neighbor_avgs2 = np.stack(same_other_neighbor_avgs2_rows)

        # pooled for set2: mean over genes (ignoring nans)
        all_gene_same_other_neighbor_avgs2 = np.nanmean(
            np.vstack([r[None, :] for r in same_other_neighbor_avgs2_rows]), axis=0
        )
        same_other_neighbor_avgs_full2 = np.vstack(
            [same_other_neighbor_avgs2, all_gene_same_other_neighbor_avgs2]
        )
    else:
        same_other_neighbor_avgs_full2 = None

    # Sorting (keep original behavior: sort by set1 medians)
    vals_diff_medians = np.array([np.nanmedian(v) for v in vals_diff1[:n_genes]])
    sort_inds = np.argsort(vals_diff_medians)

    # -------------------------------------------------------------------------
    # Global ranges for x-limits (use both sets if dual)
    # -------------------------------------------------------------------------
    pooled_for_limits = all_vals1
    if dual and all_vals2 is not None and all_vals2.size > 0:
        pooled_for_limits = np.concatenate([all_vals1, all_vals2]) if all_vals1.size > 0 else all_vals2

    if pooled_for_limits.size > 0 and np.any(~np.isnan(pooled_for_limits)):
        lo_all = np.nanpercentile(pooled_for_limits, filter_percentile)
        hi_all = np.nanpercentile(pooled_for_limits, 100 - filter_percentile)
    else:
        lo_all, hi_all = -0.1, 0.1
    x_min_other, x_max_other, xticks_other = _nice_limits_and_ticks(lo_all, hi_all)

    # Violin range: use trimmed per-gene from both sets (if dual)
    lo_vio_global = np.inf
    hi_vio_global = -np.inf
    all_gene_lists = [per_gene_vals1]
    if dual:
        all_gene_lists.append([np.asarray(v) for v in vals_diff2[:n_genes]])

    for gene_list in all_gene_lists:
        for v in gene_list:
            v = np.asarray(v)
            if v.size == 0 or np.all(np.isnan(v)):
                continue
            lo_g = np.nanpercentile(v, filter_percentile)
            hi_g = np.nanpercentile(v, 100 - filter_percentile)
            if np.isfinite(lo_g) and np.isfinite(hi_g):
                lo_vio_global = min(lo_vio_global, lo_g)
                hi_vio_global = max(hi_vio_global, hi_g)

    if not np.isfinite(lo_vio_global) or not np.isfinite(hi_vio_global):
        lo_vio_global, hi_vio_global = lo_all, hi_all
    x_min_vio, x_max_vio, xticks_vio = _nice_limits_and_ticks(lo_vio_global, hi_vio_global)

    # -------------------------------------------------------------------------
    # Decide which blocks (rows) to show
    # -------------------------------------------------------------------------
    high_inds = np.array([], dtype=int)
    low_inds = np.array([], dtype=int)

    if highN is not None and highN > 0:
        high_inds = sort_inds[-min(highN, n_genes):]
    if lowN is not None and lowN > 0:
        low_inds = sort_inds[:min(lowN, n_genes)]

    extremes_cover_all = (
        highN is not None and lowN is not None and highN > 0 and lowN > 0 and highN + lowN >= n_genes
    )

    if show_all_genes or extremes_cover_all:
        gene_blocks = [("all_genes", sort_inds)]
    else:
        gene_blocks = []
        if high_inds.size > 0:
            gene_blocks.append(("high", high_inds))
        if low_inds.size > 0:
            gene_blocks.append(("low", low_inds))

    blocks = []
    if include_pooled_row:
        blocks.append({"kind": "pooled", "indices": np.array([pooled_index], dtype=int)})
    for kind, inds in gene_blocks:
        if inds.size > 0:
            blocks.append({"kind": kind, "indices": np.asarray(inds, dtype=int)})

    if len(blocks) == 0:
        raise ValueError("No blocks to plot: check highN, lowN, include_pooled_row, show_all_genes.")

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------
    colors = np.empty(n_genes + 1, dtype=object)
    cmap_fallback = plt.get_cmap("tab20")
    for i, g in enumerate(gene_names_tot):
        if color_dict is not None and g in color_dict:
            colors[i] = color_dict[g]
        elif g == pooled_name:
            colors[i] = pooled_color
        else:
            colors[i] = cmap_fallback(i % cmap_fallback.N)

    # -------------------------------------------------------------------------
    # Figure & gridspec
    # -------------------------------------------------------------------------
    height_ratios = []
    for b in blocks:
        if b["kind"] == "pooled":
            height_ratios.append(1.0)
        else:
            height_ratios.append(float(len(b["indices"])))

    totN = sum(r for r in height_ratios if r > 1) or 1
    fig = plt.figure(figsize=(4.5, 0.2 * totN))

    n_rows = len(blocks)
    gs = gridspec.GridSpec(
        ncols=6,
        nrows=n_rows,
        figure=fig,
        width_ratios=[0.5, 0.8, 0.1, 1.0, 0.3, 1.0],
        height_ratios=height_ratios,
        wspace=0.1,
    )

    gene_ylabel_set = False
    last_row_index = n_rows - 1

    # heatmap vmax across both sets if dual
    vmax_heat = np.nanmax(same_other_neighbor_avgs_full1)
    if dual and same_other_neighbor_avgs_full2 is not None:
        vmax_heat = np.nanmax([vmax_heat, np.nanmax(same_other_neighbor_avgs_full2)])
    if not np.isfinite(vmax_heat):
        vmax_heat = 1.0

    im_for_cbar = None
    heat_axes_bottom = None
    vio_axes_bottom = None

    # Geometry constants (single vs dual)
    y_step = 6 if dual else 3
    y_center_offset = 3 if dual else 0  # center per gene row in dual mode
    y_half_offset = 1.5 if dual else 0.0

    # -------------------------------------------------------------------------
    # Loop over blocks
    # -------------------------------------------------------------------------
    for row_idx, block in enumerate(blocks):
        inds = block["indices"]
        block_kind = block["kind"]
        n_block = len(inds)

        # ----------------- (1) Neighbor avg heatmap -----------------
        ax = fig.add_subplot(gs[row_idx, 0])

        if dual:
            # For each entry (gene/pooled): stack [set2_row, set1_row] in that order
            block_rows = []
            for ind in inds:
                block_rows.append(same_other_neighbor_avgs_full2[ind] if same_other_neighbor_avgs_full2 is not None else np.full(2, np.nan))
                block_rows.append(same_other_neighbor_avgs_full1[ind])
            block_neighbor_avgs = np.vstack(block_rows)  # shape (2*n_block, 2)

            im = ax.imshow(
                block_neighbor_avgs,
                cmap="binary",
                vmin=0,
                vmax=vmax_heat,
                extent=[0, 10, 0, n_block * y_step],
                origin="lower",
            )

            if im_for_cbar is None:
                im_for_cbar = im
            if row_idx == last_row_index:
                heat_axes_bottom = ax

            # One label per gene centered between the two sub-rows
            ax.set_yticks(np.arange(n_block) * y_step + (y_step / 2))
            ylabels = ax.set_yticklabels(
                [name_dict[gene_names_tot[ind]] for ind in inds], rotation=0, style="italic"
            )
        else:
            block_neighbor_avgs = same_other_neighbor_avgs_full1[inds]
            im = ax.imshow(
                block_neighbor_avgs,
                cmap="binary",
                vmin=0,
                vmax=vmax_heat,
                extent=[0, 10, 0, n_block * y_step],
                origin="lower",
            )

            if im_for_cbar is None:
                im_for_cbar = im
            if row_idx == last_row_index:
                heat_axes_bottom = ax

            ax.set_yticks(np.arange(n_block) * y_step + 1.5)
            ylabels = ax.set_yticklabels(
                [name_dict[gene_names_tot[ind]] for ind in inds], rotation=0, style="italic"
            )

        if block_kind == "pooled":
            for lab in ylabels:
                lab.set_style("normal")

        if block_kind != "pooled" and not gene_ylabel_set:
            ax.set_ylabel(ylabel)
            gene_ylabel_set = True

        if row_idx == last_row_index:
            ax.set_xticks([2.5, 7.5])
            ax.set_xticklabels(
                ["Neighbors", "Gene-matched \nneurons"],
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
        else:
            ax.set_xticks([])

        above = blocks[row_idx - 1] if row_idx > 0 else None
        below = blocks[row_idx + 1] if row_idx < n_rows - 1 else None

        hide_top = False
        hide_bottom = False
        if block_kind == "high" and below is not None and below["kind"] == "low":
            hide_bottom = True
        if block_kind == "low" and above is not None and above["kind"] == "high":
            hide_top = True
        if hide_top:
            ax.spines["top"].set_visible(False)
        if hide_bottom:
            ax.spines["bottom"].set_visible(False)

        # ----------------- (2) LCD violin plots -----------------
        ax = fig.add_subplot(gs[row_idx, 1])
        if row_idx == last_row_index:
            vio_axes_bottom = ax

        # positions: one center per gene/pooled entry
        positions = np.arange(n_block) * y_step + (y_step / 2 if dual else 0)

        if dual:
            # Prepare trimmed blocks for both sets
            vals_block1 = vals_diff1[inds].copy()
            vals_block2 = vals_diff2[inds].copy()

            for v_i, v in enumerate(vals_block1):
                v = np.asarray(v)
                if v.size == 0:
                    continue
                lo_g = np.nanpercentile(v, filter_percentile)
                hi_g = np.nanpercentile(v, 100 - filter_percentile)
                keep = (v >= lo_g) & (v <= hi_g)
                vals_block1[v_i] = v[keep]

            for v_i, v in enumerate(vals_block2):
                v = np.asarray(v)
                if v.size == 0:
                    continue
                lo_g = np.nanpercentile(v, filter_percentile)
                hi_g = np.nanpercentile(v, 100 - filter_percentile)
                keep = (v >= lo_g) & (v <= hi_g)
                vals_block2[v_i] = v[keep]

            # Use a larger width so each half occupies its “half-row”
            vio_width = 3.0

            p1 = ax.violinplot(
                vals_block1,
                positions=positions,
                widths=vio_width,
                vert=False,
                showextrema=False,
                points=1000,
                showmedians=False,
                bw_method=bw_method,
            )
            p2 = ax.violinplot(
                vals_block2,
                positions=positions,
                widths=vio_width,
                vert=False,
                showextrema=False,
                points=1000,
                showmedians=False,
                bw_method=bw_method,
            )

            # Style + clip: set1 top, set2 bottom
            half = vio_width / 2
            for i_body, (pc, c) in enumerate(zip(p1["bodies"], colors[inds])):
                pc.set_facecolor(c)
                pc.set_alpha(1.0)
                _clip_violin_half(pc, positions[i_body], "top")
                # median tick (top half)
                v = np.asarray(vals_diff1[inds][i_body])
                if v.size > 0:
                    med = np.nanmedian(v)
                    ax.vlines(med, positions[i_body], positions[i_body] + half, color="k", lw=1)

            for i_body, (pc, c) in enumerate(zip(p2["bodies"], colors[inds])):
                pc.set_facecolor(c)
                pc.set_alpha(0.35)
                _clip_violin_half(pc, positions[i_body], "bottom")
                # median tick (bottom half)
                v = np.asarray(vals_diff2[inds][i_body])
                if v.size > 0:
                    med = np.nanmedian(v)
                    ax.vlines(med, positions[i_body] - half, positions[i_body], color="grey", lw=1)

        else:
            vals_diff_block = vals_diff1[inds].copy()
            for v_i, v in enumerate(vals_diff_block):
                v = np.asarray(v)
                if v.size == 0:
                    continue
                lo_g = np.nanpercentile(v, filter_percentile)
                hi_g = np.nanpercentile(v, 100 - filter_percentile)
                keep = (v >= lo_g) & (v <= hi_g)
                vals_diff_block[v_i] = v[keep]

            p = ax.violinplot(
                vals_diff_block,
                positions=np.arange(n_block) * y_step,
                widths=1.5,
                vert=False,
                showextrema=False,
                points=1000,
                showmedians=True,
                bw_method=bw_method,
            )
            for pc, c in zip(p["bodies"], colors[inds]):
                pc.set_facecolor(c)
                pc.set_alpha(1.0)
            p["cmedians"].set_colors("k")

        ax.axvline(0, linestyle="-", color="k", lw=0.5)
        ax.set_ylim(-2, n_block * y_step - 2)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(x_min_vio, x_max_vio)

        if row_idx == last_row_index:
            ax.set_xticks(xticks_vio)
        else:
            ax.spines[["bottom"]].set_visible(False)
            ax.set_xticks([])

        # ----------------- (3) Median +/- SEM per gene/cluster -----------------
        ax = fig.add_subplot(gs[row_idx, 3])

        if dual:
            for v_i in range(n_block):
                ind = inds[v_i]
                y_center = v_i * y_step + (y_step / 2)

                v1 = np.asarray(vals_diff1[ind])
                if v1.size > 0:
                    med1 = np.nanmedian(v1)
                    sem1 = _nansem(v1)
                    ax.errorbar(
                        x=[med1],
                        y=y_center + y_half_offset,
                        xerr=[sem1],
                        marker=".",
                        markersize=3,
                        capsize=1,
                        linestyle="",
                        color=colors[ind],
                        ecolor="k",
                    )

                v2 = np.asarray(vals_diff2[ind])
                if v2.size > 0:
                    med2 = np.nanmedian(v2)
                    sem2 = _nansem(v2)
                    ax.errorbar(
                        x=[med2],
                        y=y_center - y_half_offset,
                        xerr=[sem2],
                        marker=".",
                        markersize=3,
                        capsize=1,
                        linestyle="",
                        color=colors[ind],
                        ecolor="k", alpha=.8
                    )
        else:
            for v_i, v in enumerate(vals_diff1[inds]):
                v = np.asarray(v)
                if v.size == 0:
                    continue
                med = np.nanmedian(v)
                sem = _nansem(v)
                ax.errorbar(
                    x=[med],
                    y=v_i * y_step,
                    xerr=[sem],
                    marker=".",
                    markersize=3,
                    capsize=1,
                    linestyle="",
                    color=colors[inds][v_i],
                    ecolor="k",
                )

        ax.axvline(0, linestyle="-", color="k", lw=0.5)
        ax.set_ylim(-2, n_block * y_step - 2)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(x_min_other, x_max_other)

        if row_idx == last_row_index:
            ax.set_xticks(xticks_other)
            ax.set_xlabel("Local Correlation Difference")
        else:
            ax.set_xticks([])
            ax.spines[["bottom"]].set_visible(False)

        # ----------------- (4) Significance asterisks -----------------
        # Put stars next to the corresponding half-row (top=set1, bottom=set2)
        levels = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
        x_star = ax.get_xlim()[1] - 0.02 * (x_max_other - x_min_other)

        def _stars(pval: float) -> str:
            if np.isnan(pval):
                return ""
            for thr, s in levels:
                if pval < thr:
                    return s
            return "ns"

        if dual:
            for v_i, ind in enumerate(inds):
                y_center = v_i * y_step + (y_step / 2)
                s1 = _stars(p_vals_full1[ind])
                s2 = _stars(p_vals_full2[ind] if p_vals_full2 is not None else np.nan)
                if s1:
                    ax.text(x_star, y_center + y_half_offset - 0.3, s1, fontsize=6)
                if s2:
                    ax.text(x_star, y_center - y_half_offset - 0.3, s2, fontsize=6, color='grey')
        else:
            block_pvals = p_vals_full1[inds]
            for l, pval in enumerate(block_pvals):
                s = _stars(pval)
                if not s:
                    continue
                if s != 'ns': ax.text(x_star, l * y_step - 1.2, s, fontsize=6)
                else: ax.text(x_star, l * y_step - .5, s, fontsize=6)

    # -------------------------------------------------------------------------
    # Rightmost panel: all entries sorted by median + pooled
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[:, 5])

    order_all = np.concatenate((sort_inds, [pooled_index]))
    colors_full_order = colors[order_all]

    if dual:
        for i_ord, (ind, c) in enumerate(zip(order_all, colors_full_order)):
            # keep the original "pooled gap" behavior
            if i_ord == len(sort_inds):
                y_base = i_ord * 2 + 6
            else:
                y_base = i_ord * 2

            v1 = np.asarray(vals_diff1[ind])
            if v1.size > 0:
                med1 = np.nanmedian(v1)
                sem1 = _nansem(v1)
                ax.errorbar(
                    x=[med1],
                    y=y_base + 0.4,
                    xerr=[sem1],
                    marker=".",
                    markersize=2,
                    capsize=1,
                    linestyle="",
                    color=c,
                    ecolor="k",
                )

            v2 = np.asarray(vals_diff2[ind])
            if v2.size > 0:
                med2 = np.nanmedian(v2)
                sem2 = _nansem(v2)
                ax.errorbar(
                    x=[med2],
                    y=y_base - 0.4,
                    xerr=[sem2],
                    marker=".",
                    markersize=2,
                    capsize=1,
                    linestyle="",
                    color=c,
                    ecolor="k",
                )
    else:
        vals_diff_full_order = vals_diff1[order_all]
        for v_i, (v, c) in enumerate(zip(vals_diff_full_order, colors_full_order)):
            v = np.asarray(v)
            if v.size == 0:
                continue
            med = np.nanmedian(v)
            sem = _nansem(v)
            if v_i == len(sort_inds):
                y = v_i * 2 + 6
            else:
                y = v_i * 2
            ax.errorbar(
                x=[med],
                y=y,
                xerr=[sem],
                marker=".",
                markersize=2,
                capsize=1,
                linestyle="",
                color=c,
                ecolor="k",
            )

    ax.axvline(0, linestyle="-", color="k", lw=0.5)
    ax.set_ylim(-2, None)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(x_min_other, x_max_other)
    ax.set_xticks(xticks_other)

    # -------------------------------------------------------------------------
    # Horizontal colorbar for neighbor-avg heatmaps
    # -------------------------------------------------------------------------
    if im_for_cbar is not None and heat_axes_bottom is not None and vio_axes_bottom is not None:
        bbox0 = heat_axes_bottom.get_position()
        bbox1 = vio_axes_bottom.get_position()
        x0 = bbox1.x0
        x1 = bbox1.x1
        width = (x1 - x0) / 2
        x0 += width / 2
        y0 = min(bbox0.y0, bbox1.y0) - 0.15
        height = 0.02

        cax = fig.add_axes([x0, y0, width, height])
        cbar = fig.colorbar(
            im_for_cbar,
            cax=cax,
            orientation="horizontal",
            ticks=[0, np.nanmax(same_other_neighbor_avgs_full1) if np.isfinite(np.nanmax(same_other_neighbor_avgs_full1)) else vmax_heat],
        )
        max_tick = np.nanmax(same_other_neighbor_avgs_full1)
        if dual and same_other_neighbor_avgs_full2 is not None:
            max_tick = np.nanmax([max_tick, np.nanmax(same_other_neighbor_avgs_full2)])
        if not np.isfinite(max_tick):
            max_tick = vmax_heat

        cbar.ax.set_xticklabels(["0", f"{max_tick:.1f}"])
        cbar.set_label("Average LCD", fontsize=6)
        cbar.ax.tick_params(labelsize=6)

    # -------------------------------------------------------------------------
    # Save / show
    # -------------------------------------------------------------------------
    if save_figure:
        save_figure(save_dir, save_name)

    if show:
        plt.show()

    return fig



import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Dict, Any, Mapping, Tuple, Optional


def _plot_brain_clusters(
    cell_centers_all: np.ndarray,
    clusters: Sequence[Dict[str, Any]],
    cluster_ids: Sequence[str],
    color_dict: Mapping[str, Dict[str, Any]],  # cluster_id -> {"color", "name", "order"}
    fill_mode: str = "value",                  # "value" or "solid"
    val_cmap: str = "coolwarm",
    val_vmin: Optional[float] = None,
    val_vmax: Optional[float] = None,
    proj_axes: Sequence[Tuple[int, int]] = ((1, 2), (0, 2)),
    figsize: Tuple[float, float] = (2.35, 2.25),
    # marker controls
    s: float = 6.0,                  # outer circle size
    inner_scale: float = 0.5,        # inner size = s * inner_scale
    alpha_outer: float = 1.0,
    alpha_inner: float = 0.9,
    alpha_bg: float = 0.025,
    # sorting controls
    sort_mode: str = "none",         # "none", "by_value", "by_size", "by_cluster_value", "by_cluster_size"
    sort_values: Optional[Mapping[str, float]] = None,
    sort_ascending: bool = False,
    show_legend: bool = True,
    stim_key: str = "visrap"
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Low-level helper to plot clusters on brain projections.

    Parameters
    ----------
    cell_centers_all
        (N_all, 3) array of all neuron coordinates.
    clusters
        List of cluster dicts, each containing at least:
          - 'coords': (n_cluster, 3)
          - 'vals'  : LCD values for those neurons (N, D or N,)
          - optionally 'size' : number of neurons (for by_size/by_cluster_size sorting)
          - optionally 'mean_lcd' : scalar mean used for sorting if desired
    cluster_ids
        List of string IDs (same order as clusters), used to look up colors.
    color_dict
        Mapping cluster_id -> {"color": RGBA, "name": label, "order": int}.
        Typically the output of `compute_lcd_color_dict(cluster_dict, ...)`.
    fill_mode
        "value": inner marker colored by LCD value (val_cmap, using vals[:,0]).
        "solid": inner marker uses same color as cluster color.
    val_cmap
        Colormap name for LCD values when fill_mode="value".
    val_vmin, val_vmax
        Value range for LCD colormap. If None, inferred from all vals[:,0].
        They are always symmetrized around 0 afterwards.
    proj_axes
        Iterable of (ax1, ax2) coordinate indices for projections.
    figsize
        Figure size.
    s
        Marker area for the outer ring.
    inner_scale
        Relative size of the inner filled marker: inner_s = s * inner_scale.
    alpha_outer
        Alpha for the outer ring.
    alpha_inner
        Alpha for the inner marker.
    alpha_bg
        Alpha for the background neurons.
    sort_mode
        "none"
            Keep the clusters in the given order.
        "by_value"
            NEW meaning: sort **all neurons globally** by LCD value (vals[:,0]),
            ignoring cluster identity for the *draw order* of the inner markers
            (outer rings remain per-cluster).
        "by_cluster_value"
            OLD "by_value": sort clusters by one scalar per cluster. If
            `sort_values` is given, it's interpreted as {cluster_id -> value}.
            Otherwise, the median of c['vals'][:,0] is used.
        "by_size"
            Backwards-compatible alias for "by_cluster_size".
        "by_cluster_size"
            OLD "by_size": sort clusters by size (c['size'] if present,
            otherwise len(c['vals'])).

        NOTE: sorting controls the *draw order*; later items are drawn on top.
    sort_values
        Optional mapping {cluster_id -> value} used when sort_mode="by_cluster_value".
        If None and sort_mode="by_cluster_value", medians of c['vals'][:,0] are used.
    sort_ascending
        If True, smallest first (drawn underneath if you later reverse);
        if False (default), largest values first and smaller ones drawn last
        (on top).
    show_legend
        Whether to draw the cluster legend on the rightmost axis.

    Returns
    -------
    fig, axes
    """
    cell_centers_all = np.asarray(cell_centers_all)
    clusters = list(clusters)
    cluster_ids = list(cluster_ids)

    # Normalize sort_mode aliases
    if sort_mode == "by_size":
        warnings.warn(
            "sort_mode='by_size' is deprecated; use 'by_cluster_size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sort_mode = "by_cluster_size"

    valid_modes = ("none", "by_value", "by_cluster_value", "by_cluster_size")
    if sort_mode not in valid_modes:
        raise ValueError(
            f"sort_mode must be one of {valid_modes}, got '{sort_mode}'."
        )

    # ----------------------------- cluster-level sorting -----------------------------
    # "none" or "by_value" (global per-neuron) -> keep cluster order
    if sort_mode in ("none", "by_value"):
        order = np.arange(len(clusters), dtype=int)
    else:
        # "by_cluster_value" or "by_cluster_size"
        sort_keys = []
        for cid, c in zip(cluster_ids, clusters):
            if sort_mode == "by_cluster_size":
                if "size" in c:
                    key = float(c["size"])
                else:
                    key = float(len(c.get("lcd_vals", [])[stim_key]))
            else:  # "by_cluster_value"
                if sort_values is not None and cid in sort_values:
                    key = float(sort_values[cid])
                else:
                    v = np.asarray(c.get("vals", []), dtype=float)
                    if v.ndim == 2:
                        v = v[:, 0]
                    key = float(np.nanmedian(v)) if v.size > 0 else np.nan

            sort_keys.append(key)

        sort_keys = np.asarray(sort_keys, dtype=float)
        if np.any(np.isnan(sort_keys)):
            if sort_ascending:
                sort_keys[np.isnan(sort_keys)] = np.inf
            else:
                sort_keys[np.isnan(sort_keys)] = -np.inf

        order = np.argsort(sort_keys)
        if not sort_ascending:
            order = order[::-1]

    clusters = [clusters[i] for i in order]
    cluster_ids = [cluster_ids[i] for i in order]

    # ----------------------------- value range -----------------------------
    cmap_vals = None
    norm_vals = None

    if fill_mode == "value":
        # Compute vmin/vmax from data if needed
        if val_vmin is None or val_vmax is None:
            all_vals_list = []
            for c in clusters:
                v = np.asarray(c["lcd_vals"][stim_key][:, 0], dtype=float)
                if v.ndim == 2:
                    v = v[:, 0]
                all_vals_list.append(v)

            if len(all_vals_list) == 0:
                val_vmin, val_vmax = -0.1, 0.1
            else:
                all_vals = np.concatenate(all_vals_list)
                mask = np.isfinite(all_vals)
                if not np.any(mask):
                    val_vmin, val_vmax = -0.1, 0.1
                else:
                    lo = np.nanpercentile(all_vals[mask], 1)
                    hi = np.nanpercentile(all_vals[mask], 99)
                    if val_vmin is None:
                        val_vmin = lo
                    if val_vmax is None:
                        val_vmax = hi

        # Enforce symmetry around zero (original behavior)
        max_abs = max(abs(val_vmin), abs(val_vmax))
        val_vmin, val_vmax = -max_abs, max_abs

        cmap_vals = plt.get_cmap(val_cmap)
        norm_vals = plt.Normalize(vmin=val_vmin, vmax=val_vmax)

    # ----------------------------- precompute per-cluster arrays -----------------------------
    xs_clusters = []
    ys_clusters = []
    vals_scalar_clusters = []
    colors_clusters = []
    labels_clusters = []

    for cid, c in zip(cluster_ids, clusters):
        coords = np.asarray(c["coords"], dtype=float)
        vals = np.asarray(c["lcd_vals"][stim_key], dtype=float)

        if coords.shape[0] != vals.shape[0]:
            raise ValueError(
                f"Cluster '{cid}' has mismatched coords/vals shapes: "
                f"{coords.shape} vs {vals.shape}"
            )

        if vals.ndim == 2:
            vals_scalar = vals[:, 0]
        else:
            vals_scalar = vals

        info = color_dict.get(cid, {"color": (0.0, 0.0, 0.0, 1.0), "name": cid})
        cluster_color = np.array(info.get("color", (0.0, 0.0, 0.0, 1.0)))
        cluster_label = info.get("name", cid)

        xs_clusters.append(coords[:, 0])  # we'll slice by proj_axes later
        ys_clusters.append(coords[:, 1])  # (temporary; we’ll re-slice)
        vals_scalar_clusters.append(vals_scalar)
        colors_clusters.append(cluster_color)
        labels_clusters.append(cluster_label)

    # ----------------------------- plotting -----------------------------
    n_proj = len(proj_axes)
    fig, axes = plt.subplots(
        1,
        n_proj,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1] * n_proj, "wspace": -0.15},
    )
    if n_proj == 1:
        axes = np.array([axes])

    inner_s = s * inner_scale

    for ax_i, (ax1, ax2) in enumerate(proj_axes):
        ax = axes[ax_i]

        # Background: all neurons in gray
        ax.scatter(
            cell_centers_all[:, ax1],
            cell_centers_all[:, ax2],
            s=1,
            color=(0.5, 0.5, 0.5),
            alpha=alpha_bg,
            linewidths=0,
            rasterized=True,
        )

        # Recompute per-projection coordinates
        proj_xs = []
        proj_ys = []
        for c in clusters:
            coords = np.asarray(c["coords"], dtype=float)
            proj_xs.append(coords[:, ax1])
            proj_ys.append(coords[:, ax2])

        # --- outer rings: cluster identity ---
        for x, y, cluster_color, cluster_label in zip(
            proj_xs, proj_ys, colors_clusters, labels_clusters
        ):
            label = cluster_label if (show_legend and ax_i == n_proj - 1) else None

            ax.scatter(
                x,
                y,
                s=s,                      # outer size
                facecolors="none",
                edgecolors=[cluster_color],
                linewidths=1.0,
                alpha=alpha_outer,
                zorder=4,
                label=label,
                rasterized=True,
            )

        # --- inner fill ---
        if fill_mode == "value" and cmap_vals is not None and norm_vals is not None:
            if sort_mode == "by_value":
                # Global per-neuron sort by LCD value, ignoring cluster identity
                x_all = np.concatenate(proj_xs)
                y_all = np.concatenate(proj_ys)
                v_all = np.concatenate(vals_scalar_clusters)

                mask = np.isfinite(v_all)
                x_all = x_all[mask]
                y_all = y_all[mask]
                v_all = v_all[mask]

                if v_all.size > 0:
                    idx = np.argsort(np.abs(v_all))
                    if not sort_ascending:
                        idx = idx[::-1]

                    ax.scatter(
                        x_all[idx],
                        y_all[idx],
                        s=inner_s,
                        c=v_all[idx],
                        cmap=cmap_vals,
                        norm=norm_vals,
                        edgecolors="none",
                        alpha=alpha_inner,
                        zorder=5,
                        rasterized=True,
                    )
            else:
                # Cluster-wise draw order (possibly sorted at cluster level)
                for x, y, v in zip(proj_xs, proj_ys, vals_scalar_clusters):
                    mask = np.isfinite(v)
                    if not np.any(mask):
                        continue

                    ax.scatter(
                        x[mask],
                        y[mask],
                        s=inner_s,
                        c=v[mask],
                        cmap=cmap_vals,
                        norm=norm_vals,
                        edgecolors="none",
                        alpha=alpha_inner,
                        zorder=5,
                        rasterized=True,
                    )
        elif fill_mode == "solid":
            for x, y, cluster_color in zip(proj_xs, proj_ys, colors_clusters):
                ax.scatter(
                    x,
                    y,
                    s=inner_s,
                    c=[cluster_color],
                    edgecolors="none",
                    alpha=alpha_inner,
                    zorder=5,
                    rasterized=True,
                )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

        # legend
        if show_legend and ax_i == n_proj - 1:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                legend = ax.legend(
                    handles[::-1],
                    labels[::-1],
                    bbox_to_anchor=(0.9, 1.1),
                    markerscale=2.0,
                    frameon=False,
                    fontsize=6,
                    labelspacing=0.25,
                    title="Cluster number \n(leading gene)",
                )
                legend.get_title().set_fontsize(6)

    # ----------------------------- colorbar -----------------------------
    if fill_mode == "value" and cmap_vals is not None and norm_vals is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_vals, norm=norm_vals)
        sm.set_array([])

        bbox = axes[-1].get_position()
        x0 = bbox.x0
        x1 = bbox.x1
        width = (x1 - x0) / 10.0
        height = (x1 - x0) / 1.5

        cax = fig.add_axes([x1 + width, bbox.y0 + height, width, height])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=6, length=2, pad=1)
        cbar.set_label("LCD", fontsize=6, labelpad=2)

    return fig, axes


# ---------------------------------------------------------------------
# 2. Wrapper: plot clusters for a single gene
# ---------------------------------------------------------------------

def plot_gene_cluster_brain_map(
    cell_centers_all: np.ndarray,
    clusters_by_gene: Dict[str, Dict[str, Any]],
    gene: str,
    color_dict: Mapping[str, Dict[str, Any]],
    fill_mode: str = "value",        # "value" or "solid"
    val_cmap: str = "coolwarm",
    val_vmin: Optional[float] = None,
    val_vmax: Optional[float] = None,
    proj_axes: Sequence[Tuple[int, int]] = ((1, 2), (0, 2)),
    figsize: Tuple[float, float] = (2.35, 2.25),
    s: float = 6.0,
    inner_scale: float = 0.5,
    alpha_outer: float = 1.0,
    alpha_inner: float = 0.9,
    alpha_bg: float = 0.025,
    sort_mode: str = "none",
    sort_values: Optional[Mapping[str, float]] = None,
    sort_ascending: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot all clusters for a single gene on brain projections.

    Parameters
    ----------
    cell_centers_all
        (N_all, 3) array of all neuron coordinates.
    clusters_by_gene
        Dict: gene -> dict of {cluster_id_str -> cluster dict}.
    gene
        The gene whose clusters should be plotted.
    color_dict
        Mapping cluster_id_str -> {"color", "name", "order"}.
    Other parameters
        Passed to _plot_brain_clusters.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes for the projections.
    """
    if gene not in clusters_by_gene:
        raise ValueError(f"Gene '{gene}' not found in clusters_by_gene.")

    gene_clusters_dict = clusters_by_gene[gene]
    if len(gene_clusters_dict) == 0:
        raise ValueError(f"No clusters found for gene '{gene}'.")

    cluster_ids = list(gene_clusters_dict.keys())
    clusters = [gene_clusters_dict[cid] for cid in cluster_ids]

    fig, axes = _plot_brain_clusters(
        cell_centers_all=cell_centers_all,
        clusters=clusters,
        cluster_ids=cluster_ids,
        color_dict=color_dict,
        fill_mode=fill_mode,
        val_cmap=val_cmap,
        val_vmin=val_vmin,
        val_vmax=val_vmax,
        proj_axes=proj_axes,
        figsize=figsize,
        s=s,
        inner_scale=inner_scale,
        alpha_outer=alpha_outer,
        alpha_inner=alpha_inner,
        alpha_bg=alpha_bg,
        sort_mode=sort_mode,
        sort_values=sort_values,
        sort_ascending=sort_ascending,
        show_legend=True,
    )

    return fig, axes


# ---------------------------------------------------------------------
# 3. Wrapper: plot extreme clusters across all genes
# ---------------------------------------------------------------------

def plot_extreme_clusters_brain_map(
    cell_centers_all: np.ndarray,
    cluster_dict: Dict[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    which: str = "both",             # "both", "high", "low"
    fill_mode: str = "value",        # "value" or "solid"
    val_cmap: str = "coolwarm",
    val_vmin: Optional[float] = None,
    val_vmax: Optional[float] = None,
    proj_axes: Sequence[Tuple[int, int]] = ((1, 2), (0, 2)),
    figsize: Tuple[float, float] = (2.35, 2.25),
    s: float = 6.0,
    inner_scale: float = 0.5,
    alpha_outer: float = 1.0,
    alpha_inner: float = 0.9,
    alpha_bg: float = 0.025,
    sort_mode: str = "none",         # "none", "by_value", "by_size"
    sort_values: Optional[Mapping[str, float]] = None,
    sort_ascending: bool = False,
    show_legend: bool = True,
    stim_key: str = "visrap",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot only the extreme clusters (as defined by `order` in `color_dict`)
    across all genes on brain projections.

    Parameters
    ----------
    which
        "both": plot all extreme clusters (order > 0).
        "high": plot only extreme clusters with positive mean LCD.
        "low" : plot only extreme clusters with negative mean LCD.
        (NaN or zero means are excluded for "high"/"low".)
    Other parameters
        Passed to _plot_brain_clusters.
    """
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")

    # Select clusters with order > 0 (extremes) and then filter by sign if needed
    selected_ids: List[str] = []
    for cid, info in color_dict.items():
        if cid not in cluster_dict:
            continue
        order = info.get("order", 0)
        if order <= 0:
            continue

        c = cluster_dict[cid]
        # Prefer precomputed mean_lcd if present
        if "mean_lcd" in c:
            mean_lcd = float(c["mean_lcd"])
        else:
            v = np.asarray(c["lcd_vals"][stim_key], dtype=float)
            if v.ndim == 2:
                v = v[:, 0]
            mean_lcd = float(np.nanmean(v)) if v.size > 0 else np.nan

        if which == "both":
            selected_ids.append(cid)
        elif which == "high":
            if np.isfinite(mean_lcd) and mean_lcd > 0:
                selected_ids.append(cid)
        elif which == "low":
            if np.isfinite(mean_lcd) and mean_lcd < 0:
                selected_ids.append(cid)

    if len(selected_ids) == 0:
        raise ValueError("No extreme clusters selected (check 'which' and color_dict).")

    clusters_selected = [cluster_dict[cid] for cid in selected_ids]

    fig, axes = _plot_brain_clusters(
        cell_centers_all=cell_centers_all,
        clusters=clusters_selected,
        cluster_ids=selected_ids,
        color_dict=color_dict,
        fill_mode=fill_mode,
        val_cmap=val_cmap,
        val_vmin=val_vmin,
        val_vmax=val_vmax,
        proj_axes=proj_axes,
        figsize=figsize,
        s=s,
        inner_scale=inner_scale,
        alpha_outer=alpha_outer,
        alpha_inner=alpha_inner,
        alpha_bg=alpha_bg,
        sort_mode=sort_mode,
        sort_values=sort_values,
        sort_ascending=sort_ascending,
        show_legend=show_legend,
        stim_key=stim_key
    )

    return fig, axes



def heatmap_scatter(
    data: np.ndarray,
    ax: plt.Axes,
    size_norm_obj: Normalize,
    size_min: float,
    size_max: float,
    color_array: Optional[np.ndarray] = None,
    use_abs_for_size: bool = True,
) -> None:
    """
    Draw a 'bubble heatmap' for a 2D data array using scatter.

    Parameters
    ----------
    data
        2D array, shape (n_rows, n_cols). Each entry is mapped to a marker.
    ax
        Matplotlib Axes to draw on.
    size_norm_obj
        A matplotlib.colors.Normalize instance used to map data -> [0, 1].
        Typically shared across all panels for consistent size scaling.
    size_min, size_max
        Minimum and maximum marker size used for plotting.
    color_array
        Optional array of shape (n_rows, n_cols, 4) or (n_rows, n_cols, 3)
        giving RGBA/RGB colors per entry. If None, all markers are black.
    use_abs_for_size
        If True, map abs(data) through size_norm_obj; otherwise, use data directly.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {data.shape}")

    n_rows, n_cols = data.shape

    xs, ys, sizes, colors = [], [], [], []
    default_color = np.array([0.0, 0.0, 0.0, 1.0])

    if color_array is not None:
        color_array = np.asarray(color_array)
        if color_array.shape[:2] != (n_rows, n_cols):
            raise ValueError(
                f"color_array shape {color_array.shape} does not match data "
                f"shape {(n_rows, n_cols)}."
            )

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue

            xs.append(j)
            ys.append(i)

            v_for_size = abs(val) if use_abs_for_size else val
            v01 = size_norm_obj(v_for_size)
            v01 = float(np.clip(v01, 0.0, 1.0))
            s = size_min + (size_max - size_min) * v01
            sizes.append(s)

            if color_array is not None:
                colors.append(color_array[i, j])
            else:
                colors.append(default_color)

    if len(xs) > 0:
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=np.asarray(colors),
            marker="o",
            edgecolors="none",
            rasterized=False,
        )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()  # row 0 at the top


from typing import Dict, Any, Mapping, Sequence, Tuple, Optional, Callable, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize


def plot_cluster_gene_counts_scatter(
    cluster_dict: Dict[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    gene_names: Sequence[str],
    which: Literal["both", "high", "low"] = "both",
    lowN: int = 3,
    highN: int = 12,
    # selection / filtering
    min_cells_per_cluster: int = 5,
    order_genes: bool = True,
    # size mapping
    use_abs_for_size: bool = True,
    size_min: float = 0.5,
    size_max: float = 50.0,
    legend_display_values: Sequence[float] = (10, 50, 100, 250),
    # saving
    save_figure: Optional[Callable[[str, str], None]] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[float, float] = (3.4, 1.4),
    use_order_label: bool = True,
    stim_key: str = "visrap"
) -> plt.Figure:
    """
    Bubble heatmap of per-cluster median gene counts (clusters x genes),
    using the new cluster_dict + color_dict (no DataFrame).

    Cluster selection logic mirrors `plot_cluster_stim_responses`:

      - First build a list of valid clusters (present in color_dict, passing
        size filter).
      - Then:
          * if any cluster has non-zero 'order' in color_dict:
                use 'order' to rank clusters (ascending);
                extremes = those with order>0, sorted by order.
          * else:
                rank all clusters by mean_lcd (ascending).
      - From this ranked list 'extremes_sorted', take:
          * low_clusters  = first lowN
          * high_clusters = last highN
        (with bounds checked against available number of clusters).
      - `which` controls which of these are actually plotted.

    Inputs
    ------
    cluster_dict
        Mapping cluster_id_str -> cluster dict, as produced by
        summarize_significant_clusters_for_gene / run_multiscale_lcd_clustering_across_genes.
        Each cluster dict must contain at least:
          - 'gene_counts': (n_cells_in_cluster, n_genes)
          - 'vals'       : LCD values per neuron (N or N×D)
          - 'size'       : cluster size (optional; else inferred from gene_counts)
    color_dict
        Mapping cluster_id_str -> {"color", "name", "order"} as returned by
        compute_lcd_color_dict(cluster_dict, ...). Clusters with order > 0 are
        considered “extreme” when using the 'order' ranking.
    gene_names
        1D array-like of gene names, length = n_genes. Must match the second
        dimension of each cluster['gene_counts'].

    which
        "both": show lowN lowest-ranked and highN highest-ranked clusters
                in two panels (top=high, bottom=low).
        "high": show only highN highest-ranked clusters (single panel).
        "low" : show only lowN lowest-ranked clusters (single panel).

    Returns
    -------
    fig
        Matplotlib Figure.
    """
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")

    gene_names = np.asarray(gene_names)
    n_genes = gene_names.size

    # ------------------------------------------------------------------
    # 1) Collect cluster-level info, similar to plot_cluster_stim_responses
    # ------------------------------------------------------------------
    cluster_infos = []

    for cid, c in cluster_dict.items():
        if cid not in color_dict:
            continue
        if cid == "all clusters":
            continue  # ignore pooled entry if present

        # Size filter
        size = int(c.get("size", c.get("coords", np.empty((0, 3))).shape[0]))
        if size < min_cells_per_cluster:
            continue

        # mean_lcd (for fallback ranking)
        mean_lcd = c.get("mean_lcd", None)
        if mean_lcd is None:
            vals = np.asarray(c.get("lcd_vals", [])[stim_key], dtype=float)
            if vals.ndim == 2:
                v = vals[:, 0]
            else:
                v = vals
            mean_lcd = float(np.nanmean(v)) if v.size > 0 else np.nan

        # gene_counts
        gc = np.asarray(c.get("gene_counts", []), dtype=float)
        if gc.ndim != 2:
            raise ValueError(
                f"cluster[{cid!r}]['gene_counts'] must be 2D (n_cells, n_genes); "
                f"got shape {gc.shape}"
            )
        if gc.shape[1] != n_genes:
            raise ValueError(
                f"cluster[{cid!r}]['gene_counts'] second dimension {gc.shape[1]} "
                f"does not match len(gene_names)={n_genes}."
            )

        # median gene counts per cluster
        gene_counts_agg = np.nanmedian(gc, axis=0)  # (n_genes,)

        # Color / label / order from color_dict
        cd_entry = color_dict.get(cid, {})
        color = np.asarray(cd_entry.get("color", (0.0, 0.0, 0.0, 1.0)))
        if use_order_label:
            label = cd_entry.get("order", cid)
        else:
            label = cd_entry.get("name", cid)
        order = cd_entry.get("order", 0)
        if order is None:
            order = 0
        name = cd_entry.get("name", 'none')

        cluster_infos.append(
            {
                "cluster_id": cid,
                "size": size,
                "mean_lcd": float(mean_lcd),
                "gene_counts_agg": gene_counts_agg,
                "color": color,
                "label": label,
                "name": name,
                "order": int(order),
            }
        )

    if len(cluster_infos) == 0:
        raise ValueError("No clusters to plot after filtering and key intersection.")

    # ------------------------------------------------------------------
    # 2) Use 'order' if present; otherwise rank by mean_lcd (same as stim)
    # ------------------------------------------------------------------
    orders = np.array([ci["order"] for ci in cluster_infos], dtype=int)
    has_ranking = np.any(orders != 0)

    if has_ranking:
        # Use 'order' from color_dict; keep only those with order > 0
        extremes = [ci for ci in cluster_infos if ci["order"] > 0]
        if len(extremes) == 0:
            raise ValueError(
                "Color dict indicates extremes via 'order', but none are non-zero."
            )
        # Sort by order (1,2,...,lowN+highN)
        extremes_sorted = sorted(extremes, key=lambda ci: ci["order"])
    else:
        # Fallback: rank all clusters by mean_lcd (ascending)
        extremes_sorted = sorted(cluster_infos, key=lambda ci: ci["mean_lcd"])

    # determine how many we can actually take
    n_ext = len(extremes_sorted)
    eff_lowN = min(lowN, n_ext)
    eff_highN = min(highN, max(0, n_ext - eff_lowN))

    # Low extremes: first eff_lowN
    low_clusters = extremes_sorted[:eff_lowN][::-1]
    # High extremes: last eff_highN
    high_clusters = extremes_sorted[-eff_highN:][::-1] if eff_highN > 0 else []

    # Apply 'which' filter
    if which == "high":
        low_clusters = []
    elif which == "low":
        high_clusters = []

    if len(high_clusters) == 0 and len(low_clusters) == 0:
        raise ValueError("No clusters to plot after applying 'which' + cutoffs.")

    N_high = len(high_clusters)
    N_low = len(low_clusters)

    # ------------------------------------------------------------------
    # 3) Build per-cluster median gene counts in final row order
    #     (low block, then high block)
    # ------------------------------------------------------------------
    clusters_ordered = low_clusters + high_clusters
    n_clusters_total = len(clusters_ordered)

    gene_counts_stack_all = []
    cluster_labels_all = []
    cluster_colors_all = []

    for ci in clusters_ordered:
        gene_counts_stack_all.append(ci["gene_counts_agg"])
        cluster_labels_all.append(ci["name"].split(' ')[0])
        cluster_colors_all.append(ci["color"])

    gene_counts_stack_all = np.stack(gene_counts_stack_all, axis=0)  # (n_clusters_total, n_genes)
    cluster_colors_all = np.stack(cluster_colors_all, axis=0)        # (n_clusters_total, 4)

    # ------------------------------------------------------------------
    # 4) Column (gene) ordering (optional)
    # ------------------------------------------------------------------
    if order_genes:
        # Leading gene for each cluster from cluster_id prefix
        leading_gene_for_ci = {
            ci["cluster_id"]: ci["cluster_id"].split("_")[0]
            for ci in clusters_ordered
        }

        # Use ranking (extremes_sorted) to prioritize leading genes:
        lead_genes: list[str] = []
        for ci in extremes_sorted[::-1]:  # highest rank last in extremes_sorted, so iterate reversed
            cid = ci["cluster_id"]
            g = leading_gene_for_ci.get(cid, None)
            if g is None:
                continue
            if g in gene_names and g not in lead_genes:
                lead_genes.append(g)

        order_lead = [
            int(np.where(gene_names == g)[0][0])
            for g in lead_genes
            if g in gene_names
        ]

        if gene_counts_stack_all.size > 0:
            max_per_gene = np.nanmax(gene_counts_stack_all, axis=0)
            order_rest = np.flip(np.argsort(max_per_gene))
            order_lead_set = set(order_lead)
            order_rest = [int(idx) for idx in order_rest if idx not in order_lead_set]
            column_order = np.array(order_lead + order_rest, dtype=int)
        else:
            column_order = np.arange(n_genes, dtype=int)
    else:
        column_order = np.arange(n_genes, dtype=int)

    cluster_gene_expressions_full = gene_counts_stack_all[:, column_order]
    num_genes_after_order = cluster_gene_expressions_full.shape[1]

    # Color array: one row color per cluster, repeated across genes
    cluster_gene_expressions_colors = np.repeat(
        cluster_colors_all[:, None, :], num_genes_after_order, axis=1
    )

    # ------------------------------------------------------------------
    # 5) Size normalization across all selected clusters
    # ------------------------------------------------------------------
    size_data = cluster_gene_expressions_full.flatten()
    if use_abs_for_size:
        size_data = np.abs(size_data)
    valid_size_data = size_data[~np.isnan(size_data)]

    if valid_size_data.size > 0:
        global_size_norm_obj = Normalize(
            vmin=float(np.nanmin(valid_size_data)),
            vmax=float(np.nanmax(valid_size_data)),
        )
    else:
        global_size_norm_obj = Normalize(vmin=0.0, vmax=1.0)

    # ------------------------------------------------------------------
    # 6) Split into low/high blocks in row space
    # ------------------------------------------------------------------
    low_block_data = cluster_gene_expressions_full[:N_low]
    high_block_data = cluster_gene_expressions_full[N_low:]

    low_block_colors = cluster_gene_expressions_colors[:N_low]
    high_block_colors = cluster_gene_expressions_colors[N_low:]

    low_block_labels = cluster_labels_all[:N_low]
    high_block_labels = cluster_labels_all[N_low:]

    # ------------------------------------------------------------------
    # 7) Plotting: two-panel or one-panel layout
    # ------------------------------------------------------------------
    if which == "both":
        nrows = 2
        height_ratios = [
            N_high if N_high > 0 else 0.01,
            N_low if N_low > 0 else 0.01,
        ]
    elif which == "high":
        nrows = 1
        height_ratios = [N_high if N_high > 0 else 0.01]
    else:  # which == "low"
        nrows = 1
        height_ratios = [N_low if N_low > 0 else 0.01]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=nrows,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[1, 0.2],
        hspace=0.1,
        wspace=-0.25,
    )

    # Helper to add one panel
    def _plot_block(
        row_idx: int,
        block_data: np.ndarray,
        block_colors: np.ndarray,
        block_labels: Sequence[str],
        show_xticks: bool,
    ) -> plt.Axes:
        ax = fig.add_subplot(gs[row_idx, 0])
        if block_data.shape[0] > 0:
            heatmap_scatter(
                data=block_data,
                ax=ax,
                size_norm_obj=global_size_norm_obj,
                size_min=size_min,
                size_max=size_max,
                color_array=block_colors,
                use_abs_for_size=use_abs_for_size,
            )
            if show_xticks:
                ax.set_xticks(np.arange(num_genes_after_order))
                ax.set_xticklabels(
                    [f"{gene_names[g]}" for g in column_order],
                    fontsize=6,
                    rotation=85,
                    ha="center",
                    style="italic",
                )
            else:
                ax.set_xticks([])

            ax.set_yticks(np.arange(block_data.shape[0]))
            ax.set_yticklabels(block_labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_aspect("auto")
        return ax

    # TOP (high) panel
    if which in ("both", "high"):
        ax_high = _plot_block(
            row_idx=0,
            block_data=high_block_data,
            block_colors=high_block_colors,
            block_labels=high_block_labels,
            show_xticks=False if which == "both" else True,
        )
        ax_high.spines[["bottom"]].set_visible(False if which == "both" else True)
    else:
        ax_high = None

    # BOTTOM (low) panel
    if which in ("both", "low"):
        row_low = 1 if which == "both" else 0
        ax_low = _plot_block(
            row_idx=row_low,
            block_data=low_block_data,
            block_colors=low_block_colors,
            block_labels=low_block_labels,
            show_xticks=True,
        )
        ax_low.spines[["top"]].set_visible(False if which == "both" else True)
        ax_low.set_xlabel("Gene", fontsize=8, labelpad=0.1)
    else:
        ax_low = None

    # Size legend on right of an existing panel
    ax_for_legend = ax_high if ax_high is not None else ax_low

    if ax_for_legend is not None:
        for val in legend_display_values:
            norm_v = global_size_norm_obj(val)
            s_val = size_min + (size_max - size_min) * norm_v
            ax_for_legend.scatter(
                [], [], s=s_val, color="k", label=str(val), marker="o", edgecolors="none"
            )

        legend = ax_for_legend.legend(
            loc="right",
            bbox_to_anchor=(1.20, 0.5),
            frameon=False,
            title="Median \n gene count",
            alignment="center",
            fontsize=6,
        )
        legend.get_title().set_fontsize(6)

    # Y-label axis (Cluster)
    ax_y_label = fig.add_subplot(gs[:, 0])
    ax_y_label.axis("off")
    ax_y_label.text(
        x=-0.1,
        y=0.4,
        s="Cluster",
        rotation=90,
        fontsize=8,
        transform=ax_y_label.transAxes,
        ha="center",
        va="center",
    )

    fig.tight_layout()
    if save_figure is not None and save_dir is not None and save_name is not None:
        save_figure(save_dir, save_name)

    return fig


from typing import Mapping, Dict, Any, Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_cluster_stim_responses(
    cluster_dict: Mapping[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    which: str = "both",            # "both", "high", "low"
    lowN: int = 3,
    highN: int = 12,
    min_cells_per_cluster: int = 5,
    stim_key: str = "visrap",

    # stimulus structure
    n_stim_reps: int = 8,
    stim_block_len: int = 15 + 7 + 16,   # baseline + stim + ITI
    stim_onset: int = 15,
    stim_dur: int = 7,

    # time axis
    sample_rate_hz: float = 2.0,    # 2 Hz -> 0.5 s per sample
    xtick_step: int = 100,

    # layout
    figsize: Tuple[float, float] = (1.35, 1.4),
    offset_step: float = 2.0,
    use_order_label: bool = True,

    # neighbor visualization
    plot_cluster: bool = True,
    plot_neighbors: bool = False,
    plot_difference: bool = False,
    fish_data: Optional[Mapping[str, Dict[str, Any]]] = None,
    neighbor_radius: Optional[float] = None,
    neighbor_traces: Optional[Mapping[str, Dict[str, np.ndarray]]] = None,

    # saving
    save_figure: Optional[Callable] = None,
    save_dir: str = "./Figure_Panels/",
    save_name: str = "LCD_cluster_stimulus_responses",
) -> plt.Figure:
    """
    Plot average stimulus response traces for extreme LCD clusters
    (lowN / highN) using the new cluster_dict + color_dict pipeline.

    Selection of extremes is based on:
      - primary: the 'order' field in color_dict (if any non-zero),
      - fallback: ranking by cluster mean_lcd.

    plot_neighbors
        If True, overlay the average response of local non-expressing neighbors
        (radius-based in 3D) for each cluster.
    plot_difference
        If True, also plot the time-wise difference (cluster - neighbors) in
        the same color as the cluster.
    neighbor_traces
        Optional precomputed neighbor traces from `compute_cluster_neighbor_traces`.
        If provided, they are used directly and `fish_data` / `neighbor_radius`
        are ignored for neighbor computation.
    """
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")

    # ------------------------------------------------------------
    # 1) Collect cluster-level information
    # ------------------------------------------------------------
    cluster_infos = []
    time_length = None

    for cid, c in cluster_dict.items():
        if cid not in color_dict:
            continue
        if cid == "all clusters":
            continue

        size = int(c.get("size", c.get("coords", np.empty((0, 3))).shape[0]))
        if size < min_cells_per_cluster:
            continue

        # Mean LCD
        mean_lcd = c.get("mean_lcd", None)
        if mean_lcd is None:
            vals = np.asarray(c.get("lcd_vals", [])[stim_key], dtype=float)
            if vals.ndim == 2:
                vals_scalar = vals[:, 0]
            else:
                vals_scalar = vals
            mean_lcd = float(np.nanmean(vals_scalar)) if vals_scalar.size > 0 else np.nan

        # Stim responses avg
        stim_avg = c.get("stim_responses_avg", None)
        if stim_avg is None:
            raise ValueError(
                f"Cluster '{cid}' has no 'stim_responses_avg' in cluster_dict."
            )

        stim_avg = np.asarray(stim_avg, dtype=float)
        if stim_avg.ndim == 1:
            stim_avg = stim_avg[None, :]

        # Average over neurons, keep temporal dims, then flatten
        mean_trace = np.nanmean(stim_avg, axis=0)
        n_non_nan = np.sum(~np.isnan(stim_avg), axis=0)
        n_non_nan = np.maximum(1, n_non_nan)
        std_trace = np.nanstd(stim_avg, axis=0)
        sem_trace = std_trace / np.sqrt(n_non_nan)

        mean_trace = np.asarray(mean_trace, dtype=float).reshape(-1)
        sem_trace = np.asarray(sem_trace, dtype=float).reshape(-1)

        if time_length is None:
            time_length = mean_trace.shape[0]
        else:
            if mean_trace.shape[0] != time_length:
                raise ValueError(
                    f"Stimulus trace length mismatch in cluster {cid}: "
                    f"{mean_trace.shape[0]} vs expected {time_length}"
                )

        cd_entry = color_dict.get(cid, {})
        color = np.asarray(cd_entry.get("color", (0, 0, 0, 1.0)))

        if use_order_label:
            label = cd_entry.get("order", cid)
        else:
            label = cd_entry.get("name", cid).split(' ')[0]

        order = cd_entry.get("order", 0)
        if order is None:
            order = 0

        cluster_infos.append(
            dict(
                cluster_id=cid,
                size=size,
                mean_lcd=float(mean_lcd),
                mean_trace=mean_trace,
                sem_trace=sem_trace,
                color=color,
                label=label,
                order=int(order),
            )
        )

    if len(cluster_infos) == 0:
        raise ValueError("No clusters to plot after filtering.")

    # ------------------------------------------------------------
    # 2) Determine extremes
    # ------------------------------------------------------------
    orders = np.array([ci["order"] for ci in cluster_infos], dtype=int)
    has_ranking = np.any(orders != 0)

    if has_ranking:
        extremes = [ci for ci in cluster_infos if ci["order"] > 0]
        if len(extremes) == 0:
            raise ValueError("Ranking present but none have order > 0.")
        extremes_sorted = sorted(extremes, key=lambda ci: ci["order"])
    else:
        extremes_sorted = sorted(cluster_infos, key=lambda ci: ci["mean_lcd"])

    n_ext = len(extremes_sorted)
    eff_lowN = min(lowN, n_ext)
    eff_highN = min(highN, max(0, n_ext - eff_lowN))

    low_clusters = extremes_sorted[:eff_lowN]
    high_clusters = extremes_sorted[-eff_highN:] if eff_highN > 0 else []

    if which == "high":
        low_clusters = []
    elif which == "low":
        high_clusters = []

    if len(high_clusters) == 0 and len(low_clusters) == 0:
        raise ValueError("No clusters to plot.")

    N_high = len(high_clusters)
    N_low = len(low_clusters)

    # ------------------------------------------------------------
    # 2a) Prepare neighbor traces (precomputed or on-the-fly)
    # ------------------------------------------------------------
    need_neighbors = plot_neighbors or plot_difference

    # If precomputed neighbor traces are not provided, compute them now
    if need_neighbors and neighbor_traces is None:
        if fish_data is None:
            raise ValueError(
                "plot_neighbors / plot_difference=True requires either "
                "`neighbor_traces` or (`fish_data` and `neighbor_radius`)."
            )

        # Only compute for the extremes we will actually plot
        extreme_ids = [ci["cluster_id"] for ci in (low_clusters + high_clusters)]
        neighbor_traces = compute_cluster_neighbor_traces(
            cluster_dict=cluster_dict,
            fish_data=fish_data,
            neighbor_radius=neighbor_radius,
            stim_key_default="visrap",
            cluster_ids=extreme_ids,
        )

    # ------------------------------------------------------------
    # 3) Layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=[
            N_high if N_high > 0 else 0.01,
            N_low if N_low > 0 else 0.01,
        ],
        width_ratios=[0.3, 1.0],
        hspace=0.1,
        wspace=-0.25,
    )

    ax_high = fig.add_subplot(gs[0, 1]) if N_high > 0 else None
    ax_low = fig.add_subplot(gs[1, 1]) if N_low > 0 else None

    x = np.arange(time_length)

    # ---------------- LOW block ----------------
    if N_low > 0:
        for i, ci in enumerate(low_clusters):
            mean = ci["mean_trace"]
            sem = ci["sem_trace"]
            color = ci["color"]
            offset = i * offset_step

            if plot_cluster:
                # main cluster trace
                ax_low.plot(x, mean + offset, color=color, lw=0.5, zorder=3)
                ax_low.fill_between(
                    x,
                    mean + offset - sem,
                    mean + offset + sem,
                    color=color,
                    alpha=0.5,
                    linewidth=0,
                    zorder=2,
                )

            # neighbor & difference traces
            neighbor_mean = None
            neighbor_sem = None
            if need_neighbors and neighbor_traces is not None:
                tr = neighbor_traces.get(ci["cluster_id"])
                if tr is not None:
                    neighbor_mean = tr.get("neighbor_mean_trace")
                    neighbor_sem = tr.get("neighbor_sem_trace")
                    if (
                        neighbor_mean is not None
                        and neighbor_mean.shape[0] != time_length
                    ):
                        raise ValueError(
                            f"Neighbor trace length mismatch for cluster {ci['cluster_id']}: "
                            f"{neighbor_mean.shape[0]} vs expected {time_length}"
                        )

            if plot_neighbors and neighbor_mean is not None and not np.all(
                np.isnan(neighbor_mean)
            ):
                ax_low.plot(
                    x,
                    neighbor_mean + offset,
                    color='k',
                    lw=0.4,
                    alpha=0.6,
                    linestyle="-",
                    zorder=2,
                )
                if neighbor_sem is not None:
                    ax_low.fill_between(
                        x,
                        neighbor_mean + offset - neighbor_sem,
                        neighbor_mean + offset + neighbor_sem,
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                        zorder=1,
                    )

            if plot_difference and neighbor_mean is not None and not np.all(
                np.isnan(neighbor_mean)
            ):
                diff = mean - neighbor_mean
                if neighbor_sem is not None:
                    diff_sem = np.sqrt(sem**2 + neighbor_sem**2)
                else:
                    diff_sem = sem

                ax_low.plot(
                    x,
                    diff + offset,
                    color=color,
                    lw=0.7,
                    alpha=0.9,
                    zorder=4,
                )
                ax_low.fill_between(
                    x,
                    diff + offset - diff_sem,
                    diff + offset + diff_sem,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                    zorder=3,
                )

    # ---------------- HIGH block ----------------
    if N_high > 0:
        for i, ci in enumerate(high_clusters):
            mean = ci["mean_trace"]
            sem = ci["sem_trace"]
            color = ci["color"]
            offset = i * offset_step

            if plot_cluster:
                # main cluster trace
                ax_high.plot(x, mean + offset, color=color, lw=0.5, zorder=3)
                ax_high.fill_between(
                    x,
                    mean + offset - sem,
                    mean + offset + sem,
                    color=color,
                    alpha=0.5,
                    linewidth=0,
                    zorder=2,
                )

            # neighbor & difference traces
            neighbor_mean = None
            neighbor_sem = None
            if need_neighbors and neighbor_traces is not None:
                tr = neighbor_traces.get(ci["cluster_id"])
                if tr is not None:
                    neighbor_mean = tr.get("neighbor_mean_trace")
                    neighbor_sem = tr.get("neighbor_sem_trace")
                    if (
                        neighbor_mean is not None
                        and neighbor_mean.shape[0] != time_length
                    ):
                        raise ValueError(
                            f"Neighbor trace length mismatch for cluster {ci['cluster_id']}: "
                            f"{neighbor_mean.shape[0]} vs expected {time_length}"
                        )

            if plot_neighbors and neighbor_mean is not None and not np.all(
                np.isnan(neighbor_mean)
            ):
                ax_high.plot(
                    x,
                    neighbor_mean + offset,
                    color='k',
                    lw=0.4,
                    alpha=0.6,
                    linestyle="-",
                    zorder=2,
                )
                if neighbor_sem is not None:
                    ax_high.fill_between(
                        x,
                        neighbor_mean + offset - neighbor_sem,
                        neighbor_mean + offset + neighbor_sem,
                        color='k',
                        alpha=0.2,
                        linewidth=0,
                        zorder=1,
                    )

            if plot_difference and neighbor_mean is not None and not np.all(
                np.isnan(neighbor_mean)
            ):
                diff = mean - neighbor_mean
                if neighbor_sem is not None:
                    diff_sem = np.sqrt(sem**2 + neighbor_sem**2)
                else:
                    diff_sem = sem

                ax_high.plot(
                    x,
                    diff + offset,
                    color=color,
                    lw=0.7,
                    alpha=0.9,
                    zorder=4,
                )
                ax_high.fill_between(
                    x,
                    diff + offset - diff_sem,
                    diff + offset + diff_sem,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                    zorder=3,
                )

    # ------------------------------------------------------------
    # 4) Stim shading + cosmetics
    # ------------------------------------------------------------
    stim_offset = stim_dur + stim_onset
    axes_to_format = []
    if ax_high is not None:
        axes_to_format.append(ax_high)
    if ax_low is not None:
        axes_to_format.append(ax_low)

    for ax in axes_to_format:
        for stim_i in range(n_stim_reps):
            start = stim_i * stim_block_len + stim_onset
            end = stim_i * stim_block_len + stim_offset
            ax.axvspan(start, end, color="k", alpha=0.1, lw=0)

        if ax is ax_low:
            xticks = np.arange(0, time_length, xtick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks / sample_rate_hz).astype(int))
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlim(0, time_length)
        else:
            ax.spines[["right", "top", "bottom"]].set_visible(False)
            ax.set_xticks([])
            ax.set_xlim(0, time_length)

    # yticks
    if ax_low is not None and N_low > 0:
        ax_low.set_yticks(np.arange(N_low) * offset_step)
        ax_low.set_yticklabels([ci["label"] for ci in low_clusters], fontsize=6)

    if ax_high is not None and N_high > 0:
        ax_high.set_yticks(np.arange(N_high) * offset_step)
        ax_high.set_yticklabels([ci["label"] for ci in high_clusters], fontsize=6)

    # Left label column
    ax_label = fig.add_subplot(gs[:, 0])
    ax_label.axis("off")
    ax_label.text(
        x=-1.0,
        y=0.4,
        s="Cluster",
        rotation=90,
        fontsize=8,
        transform=ax_label.transAxes,
        ha="center",
        va="center",
    )

    fig.tight_layout()

    if save_figure is not None:
        save_figure(save_dir, save_name)

    return fig






def plot_lcd_spontaneous_vs_visstim(
    stats: Dict[str, Any],
    N_text: int = 5,
    cmap_dict: Optional[Mapping[str, Sequence[float]]] = None,
    ylim: Tuple[float, float] = (0.0, 0.07),
    save_figure: Optional[Callable] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot LCD medians (spont vs visstim) from precomputed stats.

    Parameters
    ----------
    stats
        Dictionary returned by `compute_lcd_spontaneous_vs_visstim_stats`.
    N_text
        Number of top-difference genes to label (plus always "all genes").
    cmap_dict
        Optional mapping gene_name -> RGBA. If None, a cmap is built from
        stats['gene_names_tot'] using make_cmap_dict.
    ylim
        Y-axis limits.
    save_figure, save_dir, save_name
        Optional saving hook (same semantics as your other plotting functions).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    gene_names_tot = stats["gene_names_tot"]
    gene_names_all = stats["gene_names_all"]
    engaged_meds = np.asarray(stats["engaged_meds"], dtype=float)
    spont_meds = np.asarray(stats["spont_meds"], dtype=float)
    engaged_SEMs = np.asarray(stats["engaged_SEMs"], dtype=float)
    spont_SEMs = np.asarray(stats["spont_SEMs"], dtype=float)
    diff_meds = np.asarray(stats["diff_meds"], dtype=float)
    sig_asterisks = stats.get("sig_asterisks", [""] * len(gene_names_all))

    n_genes = len(gene_names_tot)
    gene_names_all_plot = np.array(gene_names_all, dtype=object)

    # (spont, engaged) pairs
    meds = [(sp, en) for sp, en in zip(spont_meds, engaged_meds)]
    SEMs = [(sp, en) for sp, en in zip(spont_SEMs, engaged_SEMs)]

    # Threshold for annotated genes (top N_text by diff_meds)
    valid_diff = diff_meds[~np.isnan(diff_meds)]
    if valid_diff.size == 0:
        diff_thr = np.nan
    else:
        k = min(N_text, valid_diff.size)
        diff_thr = np.sort(valid_diff)[-k]

    # Colors
    if cmap_dict is None:
        cmap_dict = make_cmap_dict(gene_names_tot)

    colors = np.array(
        [cmap_dict[gene_name] for gene_name in gene_names_tot] + [(0.0, 0.0, 0.0, 1.0)],
        dtype=object,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(0.65, 1.4))
    ax = plt.subplot()

    for i in range(len(gene_names_all_plot)):
        # skip if missing medians
        if np.isnan(meds[i][0]):
            continue

        # alpha scaling based on diff_meds (for genes only, not pooled)
        if i < n_genes and np.nanmax(diff_meds) > 0:
            alpha = (np.maximum(diff_meds[i], 0.0) / np.nanmax(diff_meds)) ** 1.2
        else:
            alpha = 1.0

        ax.errorbar(
            x=[0, 1],
            y=meds[i],
            yerr=SEMs[i],
            color=colors[i],
            marker=".",
            markersize=1,
            alpha=alpha,
        )

        # Label genes whose diff_meds exceed threshold, plus always label pooled row
        if (not np.isnan(diff_thr) and diff_meds[i] > diff_thr) or (i == n_genes):
            if i < n_genes:
                # avoid overlapping labels: count previous "large" labels
                prev_mask = (np.arange(i) < n_genes) & (diff_meds[:i] >= diff_thr)
                close_count = np.isclose(
                    engaged_meds[i],
                    engaged_meds[:i][prev_mask],
                    atol=0.003,
                    rtol=1e-10,
                ).sum()
            else:
                close_count = 0

            if close_count > 0:
                x_text = 1.1 + (close_count - 1) * 0.4
                y_text = engaged_meds[i] * (1.1 ** close_count)
            else:
                x_text = 1.1
                y_text = engaged_meds[i]

            if i < n_genes:
                ax.text(
                    x=x_text,
                    y=y_text,
                    s=f"$\\it{{{gene_names_all_plot[i]}}}$",
                    c=colors[i],
                    fontsize=6,
                )
            else:
                ax.text(
                    x=x_text,
                    y=y_text,
                    s="all genes",
                    c=colors[i],
                    fontsize=6,
                )

            # significance stars
            ax.text(
                x=2.2,
                y=y_text,
                s=sig_asterisks[i],
                fontsize=6,
            )

    ax.set_xticks([0, 1])
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticklabels(
        ["Spontaneous", "Visually \n stimulated"],
        rotation=20,
        fontsize=6,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Average LCD", fontsize=6)
    ax.set_ylim(*ylim)
    ax.spines[["right", "top"]].set_visible(False)

    if save_dir is not None and save_figure is not None:
        save_figure(save_dir, save_name)

    return fig


def plot_neighbor_correlation_vs_radius(
    average_correlations_radii: Dict[str, Any],
    neighbors_radii: Dict[str, Any],
    stim_type: str = "visrap",
    source_key: str = "fish_concat",   # or a specific fish id in "per_fish"
    chosen_radius: float = 20.0,
    figsize: Tuple[float, float] = (2.4, 1.2),
    color_corr: str = "b",
    color_neigh: str = "r",
    save_figure: Optional[Callable] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Recreate the "average correlation vs radius" + "neighbors vs radius" plot.

    Parameters
    ----------
    average_correlations_radii, neighbors_radii
        Output from `compute_neighbor_correlation_vs_radius`.
    stim_type
        e.g. "visrap".
    source_key
        "fish_concat" to use pooled stats, or a fish id present in
        average_correlations_radii["per_fish"] and neighbors_radii["per_fish"].
    chosen_radius
        Vertical line radius (µm) to highlight (e.g. 20).
    figsize
        Figure size.
    color_corr, color_neigh
        Colors for correlation and neighbor curves.
    """
    # ------------------------------------------------------------------
    # Select data source
    # ------------------------------------------------------------------
    if source_key == "fish_concat":
        corr_entry = average_correlations_radii["fish_concat"][stim_type]
        neigh_entry = neighbors_radii["fish_concat"]
    else:
        if source_key not in average_correlations_radii["per_fish"]:
            raise KeyError(f"Fish '{source_key}' not found in average_correlations_radii['per_fish'].")
        if source_key not in neighbors_radii["per_fish"]:
            raise KeyError(f"Fish '{source_key}' not found in neighbors_radii['per_fish'].")

        corr_entry = average_correlations_radii["per_fish"][source_key][stim_type]
        neigh_entry = neighbors_radii["per_fish"][source_key]

    radii = np.asarray(corr_entry["radii"], dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # ------------------------------------------------------------------
    # Left y-axis: average correlation
    # ------------------------------------------------------------------
    avg_corr = np.asarray(corr_entry["avg"], dtype=float)
    sem_corr = np.asarray(corr_entry["SEM"], dtype=float)

    ax.plot(radii, avg_corr, c=color_corr)
    ax.fill_between(
        radii,
        avg_corr - sem_corr,
        avg_corr + sem_corr,
        color=color_corr,
        alpha=0.25,
    )
    ax.set_ylim(0, None)
    ax.set_xlabel("Radius (µm)", fontsize=6)
    ax.axvline(chosen_radius, linestyle=":", color="k")
    ax.spines[["top"]].set_visible(False)
    ax.set_ylabel("Average correlation \nto neighbors", color=color_corr, fontsize=6)

    # xticks: you previously used radii[::2] + [0]; replicate that logic
    # Include 0 explicitly if not in radii
    xticks = list(radii[::2])
    if 0.0 not in xticks:
        xticks = [0.0] + xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(r) for r in xticks], fontsize=6)
    ax.set_xlim(0, float(np.max(radii)))
    # ax.set_yticks([0, 0.05, 0.10, 0.15])

    # ------------------------------------------------------------------
    # Right y-axis: neighbor counts (log scale)
    # ------------------------------------------------------------------
    twin_ax = ax.twinx()

    avg_neigh = np.asarray(neigh_entry["avg"], dtype=float)
    std_neigh = np.asarray(neigh_entry["std"], dtype=float)

    twin_ax.plot(radii, avg_neigh, c=color_neigh)
    twin_ax.fill_between(
        radii,
        avg_neigh - std_neigh,
        avg_neigh + std_neigh,
        color=color_neigh,
        alpha=0.25,
    )
    twin_ax.set_yscale("log")
    twin_ax.spines[["top"]].set_visible(False)
    twin_ax.set_ylabel("Neighbors", color=color_neigh, fontsize=6)
    twin_ax.set_yticks([1, 10, 100, 1000])

    fig.tight_layout()

    if save_figure is not None and save_dir is not None and save_name is not None:
        save_figure(save_dir, save_name)

    return fig



import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Dict, Any, Optional, Tuple, Callable, Sequence


def plot_cluster_lcd_time_heatmaps(
    cluster_dict: Mapping[str, Dict[str, Any]],
    cluster_id: str,
    # time axis
    sample_rate_hz: float = 2.0,      # 2 Hz -> 0.5 s per sample
    xtick_step: int = 100,            # in samples
    # colormap / scaling
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # layout
    figsize: Tuple[float, float] = (1.4, 2.4),
    # stimulus / averaging
    fish_data: Optional[Mapping[Any, Dict[str, Any]]] = None,
    fish_for_stim: Optional[Any] = None,
    stim_key: str = "visrap",
    stim_cmap: Optional[Mapping[int, Any]] = None,
    show_stim_axis: bool = True,
    use_stim_averages: bool = False,
    pre_duration: int = 15,
    stim_duration: int = 7,
    post_duration: int = 16,
    # saving
    save_figure: Optional[Callable] = None,
    save_dir: str = "./Figure_Panels/",
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot time-resolved LCD contributions for a single cluster as heatmaps,
    with an optional stimulus axis on top.

    Panels (from top to bottom if stimuli are shown):
        0) stimulus identity over time (axvspan blocks, 1..8; 0 = no stimulus)
        1) neighbors (other-type correlations)
        2) within-gene (same-type correlations)
        3) LCD = same - other

    The function expects the cluster entry to contain time-resolved LCD arrays,
    ideally of shape (n_cells, 3, T), where:

        [0] LCD         (same - other)
        [1] corr_same   (same-gene correlations)
        [2] corr_other  (other-type neighbor correlations)

    To be robust to different naming conventions / storage modes, it tries the
    following keys in order:

        - 'lcd_time_full'
        - 'lcd_time_vals'
        - 'vals_time_full'

    and expects them to be either
        (n_cells, 3, T)    or
        None

    If no suitable time-resolved data are found (or all are None), the function
    returns None and prints a short message.

    If `use_stim_averages=False` (default), heatmaps are plotted over the full
    continuous time trace and, if `fish_data` is provided, a top axis will show
    the stimulus train using:

        fish_data[fish_for_stim]['ephys_data'][stim_key]['stimulus']

    If `use_stim_averages=True`, the function:
        - Segments each time series into pre/stim/post epochs using the above
          stimulus train (visrap-style).
        - Computes per-stimulus average responses using
          `calc_stim_responses` and `calc_avg_stim_responses`.
        - Flattens the per-stimulus averages into a long axis (n_stim * block_len)
          and plots heatmaps over that axis.
        - Draws a top axis with one block per stimulus, shading only the stim
          window inside each block (pre/post unshaded).

    This keeps the external API compatible while adding the stimulus-locked view.
    """
    if cluster_id not in cluster_dict:
        raise KeyError(f"cluster_id '{cluster_id}' not found in cluster_dict.")

    c = cluster_dict[cluster_id]

    # ------------------------------------------------------------------
    # 1) Extract time-resolved LCD data from the cluster
    # ------------------------------------------------------------------
    key = 'lcd_vals_time'

    if key in c and c[key] is not None:
        lcd_time_full = c[key][stim_key]

    if lcd_time_full is None:
        print(
            "[plot_cluster_lcd_time_heatmaps] "
            f"No time-resolved LCD data found for cluster '{cluster_id}'. "
            f"Expected one of {time_keys}."
        )
        return None

    lcd_time_full = np.asarray(lcd_time_full)
    if lcd_time_full.ndim != 3 or lcd_time_full.shape[1] != 3:
        raise ValueError(
            f"Time-resolved LCD for cluster '{cluster_id}' must have shape "
            f"(n_cells, 3, T). Got shape {lcd_time_full.shape}."
        )

    # Features: 0 = LCD, 1 = corr_same, 2 = corr_other
    corr_other = lcd_time_full[:, 2, :]   # neighbors
    corr_same  = lcd_time_full[:, 1, :]   # within-gene
    lcd_diff   = lcd_time_full[:, 0, :]   # LCD

    n_cells, T = corr_other.shape

    # ------------------------------------------------------------------
    # 2) Prepare stimulus vector (for axis and/or averaging)
    # ------------------------------------------------------------------
    have_stim_vec = False
    stim_vec = None
    T_stim = None
    fish_for_stim_eff = fish_for_stim

    need_stim_vec = use_stim_averages or (fish_data is not None and show_stim_axis)

    if fish_data is not None and need_stim_vec:
        fish_ids_in_cluster = np.asarray(c.get("fish_ids", []))

        if fish_for_stim_eff is None:
            unique_fish = np.unique(fish_ids_in_cluster)
            if unique_fish.size == 0:
                print(
                    "[plot_cluster_lcd_time_heatmaps] No fish_ids stored in "
                    f"cluster '{cluster_id}', cannot infer stimulus train."
                )
            else:
                fish_for_stim_eff = unique_fish[0]
                if unique_fish.size > 1:
                    print(
                        "[plot_cluster_lcd_time_heatmaps] Cluster contains "
                        f"multiple fish {unique_fish}, using '{fish_for_stim_eff}' "
                        "for stimulus axis / averaging."
                    )

        if fish_for_stim_eff is not None:
            try:
                stim_vec = np.asarray(
                    fish_data[fish_for_stim_eff]["ephys_data"][stim_key]["stimulus"],
                    dtype=int,
                )
                T_stim = stim_vec.shape[0]
                have_stim_vec = True
            except Exception as e:
                print(
                    f"[plot_cluster_lcd_time_heatmaps] Could not retrieve "
                    f"stimulus train for fish '{fish_for_stim_eff}' "
                    f"and stim_key '{stim_key}': {e}"
                )

    if use_stim_averages and not have_stim_vec:
        raise ValueError(
            "use_stim_averages=True requires a valid stimulus train "
            "(fish_data + ephys_data[stim_key]['stimulus'])."
        )

    # ------------------------------------------------------------------
    # 3) Optional: compute stimulus-locked averages for LCD contributions
    # ------------------------------------------------------------------
    stim_block_info = None  # (n_stim, block_len) if averaging, else None

    if use_stim_averages:
        from WARP.stimulus_response_utils import (
            calc_stim_responses,
            calc_avg_stim_responses,
        )

        if T_stim is None or T_stim != T:
            print(
                "[plot_cluster_lcd_time_heatmaps] Warning: LCD time length "
                f"(T={T}) and stimulus length (T_stim={T_stim}) differ. "
                "Stimulus-locked averages will use the common time window."
            )
            T_eff = min(T, T_stim if T_stim is not None else T)
            # truncate to common window
            corr_other = corr_other[:, :T_eff]
            corr_same  = corr_same[:, :T_eff]
            lcd_diff   = lcd_diff[:, :T_eff]
            stim_vec   = stim_vec[:T_eff]
            T = T_eff

        def _avg_over_stim(data_2d: np.ndarray) -> np.ndarray:
            """
            data_2d: (n_cells, T)
            returns: avg_responses: (n_cells, n_stim, block_len)
            """
            stim_responses = calc_stim_responses(
                data_2d,
                stim_vec,
                pre_duration=pre_duration,
                stim_duration=stim_duration,
                post_duration=post_duration,
            )
            avg_responses, _ = calc_avg_stim_responses(stim_responses)
            return avg_responses

        # Compute per-stimulus averages for each component
        avg_other = _avg_over_stim(corr_other)  # (n_cells, n_stim, block_len)
        avg_same  = _avg_over_stim(corr_same)
        avg_lcd   = _avg_over_stim(lcd_diff)

        if avg_other.shape != avg_same.shape or avg_other.shape != avg_lcd.shape:
            raise ValueError(
                "Inconsistent shapes for averaged LCD components: "
                f"other={avg_other.shape}, same={avg_same.shape}, lcd={avg_lcd.shape}"
            )

        n_cells, n_stim_cond, block_len = avg_other.shape
        stim_block_info = (n_stim_cond, block_len)

        # Flatten (n_cells, n_stim, block_len) -> (n_cells, n_stim * block_len)
        corr_other = avg_other.reshape(n_cells, n_stim_cond * block_len)
        corr_same  = avg_same.reshape(n_cells,  n_stim_cond * block_len)
        lcd_diff   = avg_lcd.reshape(n_cells,   n_stim_cond * block_len)

        T = n_stim_cond * block_len  # new "time" axis is concatenated blocks

    # ------------------------------------------------------------------
    # 4) Determine color range
    # ------------------------------------------------------------------
    if vmin is None or vmax is None:
        all_vals = np.concatenate(
            [
                corr_other.ravel(),
                corr_same.ravel(),
                lcd_diff.ravel(),
            ]
        )
        mask = np.isfinite(all_vals)
        if not np.any(mask):
            vmin_eff, vmax_eff = -0.1, 0.1
        else:
            lo = np.nanpercentile(all_vals[mask], 2.5)
            hi = np.nanpercentile(all_vals[mask], 97.5)
            max_abs = max(abs(lo), abs(hi))
            vmin_eff, vmax_eff = -max_abs, max_abs

        if vmin is None:
            vmin = vmin_eff
        if vmax is None:
            vmax = vmax_eff

    # ------------------------------------------------------------------
    # 5) Sort neurons by mean LCD over time (for nicer visualization)
    # ------------------------------------------------------------------
    lcd_mean = np.nanmean(lcd_diff, axis=1)
    sort_order = np.argsort(lcd_mean)  # low → high

    corr_other_s = corr_other[sort_order]
    corr_same_s  = corr_same[sort_order]
    lcd_diff_s   = lcd_diff[sort_order]

    # ------------------------------------------------------------------
    # 6) Build a proper stim_cmap (dict) for shading (if we will draw axis)
    # ------------------------------------------------------------------
    have_stim_axis = have_stim_vec and show_stim_axis

    if have_stim_axis:
        # If stim_cmap is None or a string, convert it to a dict {1..8 -> RGBA}
        if stim_cmap is None or isinstance(stim_cmap, str):
            if isinstance(stim_cmap, str):
                base_cmap = plt.cm.get_cmap(stim_cmap)
            else:
                base_cmap = plt.cm.tab10
            stim_cmap = {
                i: base_cmap((i - 1) / 8.0) for i in range(1, 9)
            }
            # stim_id 0 (no stimulus) intentionally left unmapped → no shading
        else:
            stim_cmap = dict(stim_cmap)  # ensure dict-like

    # ------------------------------------------------------------------
    # 7) Create figure and axes (with or without stimulus axis)
    # ------------------------------------------------------------------
    if have_stim_axis:
        fig, axes = plt.subplots(
            4,
            1,
            figsize=figsize,
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [0.1, 1, 1, 1]},
        )
        ax_stim, ax_other, ax_same, ax_lcd = axes
    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=figsize,
            sharex=True,
            constrained_layout=True,
        )
        ax_other, ax_same, ax_lcd = axes
        ax_stim = None  # not used

    # ------------------------------------------------------------------
    # 8) Plot stimulus axis using axvspan (if available)
    # ------------------------------------------------------------------
    if have_stim_axis and ax_stim is not None:
        if not use_stim_averages:
            # ----- Original mode: use raw stimulus vector over full trace -----
            T_eff = min(T, T_stim)

            current_val = stim_vec[0]
            seg_start = 0

            for t in range(1, T_eff):
                if stim_vec[t] != current_val:
                    color = stim_cmap.get(int(current_val)) if current_val != 0 else None
                    if color is not None:
                        ax_stim.axvspan(
                            seg_start,
                            t,
                            color='grey',
                            alpha=0.9,
                            lw=0,
                        )
                    seg_start = t
                    current_val = stim_vec[t]

            # Close last segment
            color = stim_cmap.get(int(current_val)) if current_val != 0 else None
            if color is not None:
                ax_stim.axvspan(
                    seg_start,
                    T_eff,
                    color='grey',
                    alpha=0.9,
                    lw=0,
                )

            if T_stim != T:
                print(
                    f"[plot_cluster_lcd_time_heatmaps] Warning: LCD time length (T={T}) "
                    f"and stimulus length (T_stim={T_stim}) differ. Using T_eff={T_eff} "
                    "for stimulus spans."
                )

        else:
            # ----- Stimulus-averaged mode: one block per stimulus -----
            n_stim_cond, block_len = stim_block_info

            # For each condition, shade only the stim window inside that block
            for s in range(n_stim_cond):
                stim_id = s + 1  # assume stimuli are labeled 1..n_stim
                color = stim_cmap.get(int(stim_id))
                if color is None:
                    continue

                block_start = s * block_len
                stim_start = block_start + pre_duration
                stim_end   = block_start + pre_duration + stim_duration

                # Clip to plotted range just in case
                stim_start = max(0, min(stim_start, T))
                stim_end   = max(0, min(stim_end, T))

                if stim_end > stim_start:
                    ax_stim.axvspan(
                        stim_start,
                        stim_end,
                        color='grey',
                        alpha=0.9,
                        lw=0,
                    )

        # Cosmetics
        ax_stim.set_xlim(0, T)
        ax_stim.set_yticks([])
        ax_stim.set_ylabel("Stim", fontsize=6)
        ax_stim.tick_params(axis="x", labelbottom=False)
        ax_stim.spines[["right", "top"]].set_visible(False)

    # ------------------------------------------------------------------
    # 9) Heatmap helper
    # ------------------------------------------------------------------
    def _plot_heat(ax, data, title: str):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Neurons", fontsize=6)
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(axis="y", labelsize=5)
        return im

    im_other = _plot_heat(ax_other, corr_other_s, "Neighbors (corr other)")
    im_same  = _plot_heat(ax_same,  corr_same_s,  "Within-gene (corr same)")
    im_lcd   = _plot_heat(ax_lcd,   lcd_diff_s,   "LCD (same - other)")

    # X-axis: time in seconds
    xticks = np.arange(0, T, xtick_step)
    ax_lcd.set_xticks(xticks)
    ax_lcd.set_xticklabels((xticks / sample_rate_hz).astype(int), fontsize=6)
    ax_lcd.set_xlabel("Time (s)", fontsize=7)

    # Top axes: hide x tick labels
    ax_other.tick_params(axis="x", labelbottom=False)
    ax_same.tick_params(axis="x", labelbottom=False)
    if ax_stim is not None:
        ax_stim.tick_params(axis="x", labelbottom=False)

    # ------------------------------------------------------------------
    # 10) Shared colorbar on the right (for correlation / LCD panels)
    # ------------------------------------------------------------------
    if have_stim_axis:
        cbar_axes = [ax_other, ax_same, ax_lcd]
    else:
        cbar_axes = axes

    cbar = fig.colorbar(
        im_lcd,
        ax=cbar_axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar.set_label("Correlation / LCD", fontsize=6, labelpad=2)

    # ------------------------------------------------------------------
    # 11) Overall title with cluster info if available
    # ------------------------------------------------------------------
    title_pieces = [cluster_id]
    mean_lcd_global = cluster_dict[cluster_id].get("mean_lcd", None)
    if mean_lcd_global is not None and np.isfinite(mean_lcd_global):
        title_pieces.append(f"mean LCD = {mean_lcd_global:.3f}")
    gene = cluster_dict[cluster_id].get("gene", None)
    if gene is not None:
        title_pieces.insert(0, gene)

    fig.suptitle(" | ".join(title_pieces), fontsize=7, y=1.02)

    # ------------------------------------------------------------------
    # 12) Optional saving
    # ------------------------------------------------------------------
    if save_figure is not None:
        if save_name is None:
            save_name = f"{cluster_id}_lcd_time_heatmaps"
        save_figure(save_dir, save_name)

    return fig


from typing import Dict, Any, Mapping, Sequence, Tuple, Optional, Callable, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize


def plot_cluster_stim_lcd_scatter(
    cluster_dict: Dict[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    cluster_stim_stats: Mapping[str, Dict[str, Any]],
    which: Literal["both", "high", "low"] = "both",
    lowN: int = 3,
    highN: int = 12,
    # selection / filtering
    min_cells_per_cluster: int = 5,
    use_order_label: bool = True,
    epoch: Literal["pre", "stim", "post"] = "stim",
    # size mapping
    use_abs_for_size: bool = True,
    size_min: float = 2.0,
    size_max: float = 80.0,
    legend_display_values: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
    # saving
    save_figure: Optional[Callable[[str, str], None]] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[float, float] = (3.4, 1.4),
) -> plt.Figure:
    """
    Bubble heatmap of per-cluster stimulus LCD contributions
    (clusters x stimuli), using pre-computed cluster_stim_stats.

    Each cluster must have an entry in cluster_stim_stats[cluster_id] with:
        - 'stim_ids' : (n_stim,)
        - 'mean'     : (n_stim, 3)   # columns: [pre, stim, post]
        - 'epoch_names': ['pre', 'stim', 'post'] (optional but recommended)

    We first select clusters and rank them in the same way as other LCD plots:

      - Start from clusters present in both cluster_dict and color_dict, with
        size >= min_cells_per_cluster and stats available in cluster_stim_stats.
      - If any cluster has non-zero 'order' in color_dict:
            use those with order > 0, sorted ascending by 'order'.
        Else:
            rank all valid clusters by 'mean_lcd' (ascending).
      - From this ranked list, define:
            * low_clusters  = first lowN
            * high_clusters = last highN
        (with bounds checking).
      - `which` determines which blocks to actually plot ("both", "high", "low").

    The plotted values are the per-cluster mean LCD for the chosen epoch
    (epoch='pre'/'stim'/'post'), one column per stimulus condition.

    Marker size encodes |LCD| (or LCD if use_abs_for_size=False) via a global
    Normalize object, same as in the gene-count bubble plots. Marker color
    is taken from color_dict[cluster_id]['color'].

    Returns
    -------
    fig : Matplotlib Figure
    """
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")

    # Map epoch name to index 0..2
    epoch_idx_map = {"pre": 0, "stim": 1, "post": 2}
    if epoch not in epoch_idx_map:
        raise ValueError(f"epoch must be one of {list(epoch_idx_map.keys())}, "
                         f"got {epoch!r}.")
    epoch_idx = epoch_idx_map[epoch]

    # ------------------------------------------------------------------
    # 1) Collect cluster-level info (similar to plot_cluster_stim_responses)
    # ------------------------------------------------------------------
    cluster_infos = []
    stim_ids_ref = None  # reference stimulus ordering
    n_stim_ref = None

    for cid, c in cluster_dict.items():
        if cid not in color_dict:
            continue
        if cid == "all clusters":
            continue

        stats = cluster_stim_stats.get(cid, None)
        if stats is None:
            continue

        mean = np.asarray(stats.get("mean"))
        stim_ids = np.asarray(stats.get("stim_ids"))
        if mean.ndim != 2 or mean.shape[1] < 3:
            # Expect shape (n_stim, 3) for pre/stim/post
            continue

        n_stim = mean.shape[0]
        if stim_ids.shape[0] != n_stim:
            raise ValueError(
                f"cluster_stim_stats[{cid!r}]['stim_ids'] length "
                f"{stim_ids.shape[0]} does not match mean.shape[0]={n_stim}."
            )

        if stim_ids_ref is None:
            stim_ids_ref = stim_ids
            n_stim_ref = n_stim
        else:
            # ensure consistent stimulus ordering across clusters
            if n_stim != n_stim_ref or not np.array_equal(stim_ids, stim_ids_ref):
                raise ValueError(
                    f"Inconsistent stim_ids for cluster {cid!r}. "
                    "All clusters must have the same stimulus set/order."
                )

        # Size filter (using cluster size)
        size = int(c.get("size", c.get("coords", np.empty((0, 3))).shape[0]))
        if size < min_cells_per_cluster:
            continue

        # mean_lcd (for fallback ranking)
        mean_lcd = c.get("mean_lcd", None)
        if mean_lcd is None:
            vals = np.asarray(c.get("vals", []), dtype=float)
            if vals.ndim == 2:
                v = vals[:, 0]
            else:
                v = vals
            mean_lcd = float(np.nanmean(v)) if v.size > 0 else np.nan

        # per-stimulus LCD for chosen epoch: (n_stim,)
        stim_lcd_epoch = mean[:, epoch_idx]

        cd_entry = color_dict.get(cid, {})
        color = np.asarray(cd_entry.get("color", (0.0, 0.0, 0.0, 1.0)))
        if use_order_label:
            label = cd_entry.get("order", cid)
        else:
            label = cd_entry.get("name", cid)
        order = cd_entry.get("order", 0)
        if order is None:
            order = 0

        cluster_infos.append(
            {
                "cluster_id": cid,
                "size": size,
                "mean_lcd": float(mean_lcd),
                "stim_lcd_epoch": stim_lcd_epoch,
                "color": color,
                "label": label,
                "order": int(order),
            }
        )

    if len(cluster_infos) == 0:
        raise ValueError(
            "No clusters to plot after filtering / missing cluster_stim_stats."
        )

    n_stim = n_stim_ref
    stim_labels = [f"S{int(sid)}" for sid in stim_ids_ref]

    # ------------------------------------------------------------------
    # 2) Ranking: use 'order' if present, else mean_lcd
    # ------------------------------------------------------------------
    orders = np.array([ci["order"] for ci in cluster_infos], dtype=int)
    has_ranking = np.any(orders != 0)

    if has_ranking:
        extremes = [ci for ci in cluster_infos if ci["order"] > 0]
        if len(extremes) == 0:
            raise ValueError(
                "Color dict indicates extremes via 'order', but none are non-zero."
            )
        extremes_sorted = sorted(extremes, key=lambda ci: ci["order"])
    else:
        extremes_sorted = sorted(cluster_infos, key=lambda ci: ci["mean_lcd"])

    # ------------------------------------------------------------------
    # 3) Select low/high extremes
    # ------------------------------------------------------------------
    n_ext = len(extremes_sorted)
    eff_lowN = min(lowN, n_ext)
    eff_highN = min(highN, max(0, n_ext - eff_lowN))

    # low: first eff_lowN, reverse so "strongest low" is top
    low_clusters = extremes_sorted[:eff_lowN][::-1]
    # high: last eff_highN, reverse so "strongest high" is top
    high_clusters = extremes_sorted[-eff_highN:][::-1] if eff_highN > 0 else []

    if which == "high":
        low_clusters = []
    elif which == "low":
        high_clusters = []

    if len(high_clusters) == 0 and len(low_clusters) == 0:
        raise ValueError("No clusters to plot after applying 'which' + cutoffs.")

    N_high = len(high_clusters)
    N_low = len(low_clusters)

    clusters_ordered = low_clusters + high_clusters
    n_clusters_total = len(clusters_ordered)

    # ------------------------------------------------------------------
    # 4) Stack per-cluster stim LCD vectors, build color rows
    # ------------------------------------------------------------------
    stim_lcd_stack_all = []
    cluster_labels_all = []
    cluster_colors_all = []

    for ci in clusters_ordered:
        stim_lcd_stack_all.append(ci["stim_lcd_epoch"])
        cluster_labels_all.append(ci["label"])
        cluster_colors_all.append(ci["color"])

    stim_lcd_stack_all = np.stack(stim_lcd_stack_all, axis=0)   # (n_clusters_total, n_stim)
    cluster_colors_all = np.stack(cluster_colors_all, axis=0)   # (n_clusters_total, 4)

    # Colors repeated across stimuli
    color_array_all = np.repeat(
        cluster_colors_all[:, None, :], n_stim, axis=1
    )  # (n_clusters_total, n_stim, 4)

    # ------------------------------------------------------------------
    # 5) Global size normalization
    # ------------------------------------------------------------------
    size_data = stim_lcd_stack_all.flatten()
    if use_abs_for_size:
        size_data = np.abs(size_data)
    valid_size_data = size_data[~np.isnan(size_data)]

    if valid_size_data.size > 0:
        global_size_norm_obj = Normalize(
            vmin=float(np.nanmin(valid_size_data)),
            vmax=float(np.nanmax(valid_size_data)),
        )
    else:
        global_size_norm_obj = Normalize(vmin=0.0, vmax=1.0)

    # Split into low/high blocks in row space
    low_block_data = stim_lcd_stack_all[:N_low]
    high_block_data = stim_lcd_stack_all[N_low:]

    low_block_colors = color_array_all[:N_low]
    high_block_colors = color_array_all[N_low:]

    low_block_labels = cluster_labels_all[:N_low]
    high_block_labels = cluster_labels_all[N_low:]

    # ------------------------------------------------------------------
    # 6) Layout: one or two row blocks
    # ------------------------------------------------------------------
    if which == "both":
        nrows = 2
        height_ratios = [
            N_high if N_high > 0 else 0.01,
            N_low if N_low > 0 else 0.01,
        ]
    elif which == "high":
        nrows = 1
        height_ratios = [N_high if N_high > 0 else 0.01]
    else:  # which == "low"
        nrows = 1
        height_ratios = [N_low if N_low > 0 else 0.01]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=nrows,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[1, 0.2],
        hspace=0.1,
        wspace=-0.25,
    )

    # ------------------------------------------------------------------
    # 7) Helper to plot one block
    # ------------------------------------------------------------------
    def _plot_block(
        row_idx: int,
        block_data: np.ndarray,
        block_colors: np.ndarray,
        block_labels: Sequence[str],
        show_xticks: bool,
    ) -> plt.Axes:
        ax = fig.add_subplot(gs[row_idx, 0])
        if block_data.shape[0] > 0:
            heatmap_scatter(
                data=block_data,
                ax=ax,
                size_norm_obj=global_size_norm_obj,
                size_min=size_min,
                size_max=size_max,
                color_array=block_colors,
                use_abs_for_size=use_abs_for_size,
            )

            if show_xticks:
                ax.set_xticks(np.arange(n_stim))
                ax.set_xticklabels(
                    stim_labels,
                    fontsize=6,
                    rotation=0,
                    ha="center",
                )
            else:
                ax.set_xticks([])

            ax.set_yticks(np.arange(block_data.shape[0]))
            ax.set_yticklabels(block_labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_aspect("auto")
        return ax

    # TOP (high) panel
    if which in ("both", "high"):
        ax_high = _plot_block(
            row_idx=0,
            block_data=high_block_data,
            block_colors=high_block_colors,
            block_labels=high_block_labels,
            show_xticks=False if which == "both" else True,
        )
        ax_high.spines[["bottom"]].set_visible(False if which == "both" else True)
    else:
        ax_high = None

    # BOTTOM (low) panel
    if which in ("both", "low"):
        row_low = 1 if which == "both" else 0
        ax_low = _plot_block(
            row_idx=row_low,
            block_data=low_block_data,
            block_colors=low_block_colors,
            block_labels=low_block_labels,
            show_xticks=True,
        )
        ax_low.spines[["top"]].set_visible(False if which == "both" else True)
        ax_low.set_xlabel("Stimulus", fontsize=8, labelpad=0.1)
    else:
        ax_low = None

    # ------------------------------------------------------------------
    # 8) Size legend on the right
    # ------------------------------------------------------------------
    ax_for_legend = ax_high if ax_high is not None else ax_low

    if ax_for_legend is not None:
        for val in legend_display_values:
            norm_v = global_size_norm_obj(val if not use_abs_for_size else abs(val))
            s_val = size_min + (size_max - size_min) * norm_v
            ax_for_legend.scatter(
                [], [], s=s_val, color="k", label=f"{val:g}", marker="o", edgecolors="none"
            )

        legend_title = f"Mean LCD\n(epoch={epoch})"
        legend = ax_for_legend.legend(
            loc="right",
            bbox_to_anchor=(1.20, 0.5),
            frameon=False,
            title=legend_title,
            alignment="center",
            fontsize=6,
        )
        legend.get_title().set_fontsize(6)

    # ------------------------------------------------------------------
    # 9) Y-label axis (Cluster)
    # ------------------------------------------------------------------
    ax_y_label = fig.add_subplot(gs[:, 0])
    ax_y_label.axis("off")
    ax_y_label.text(
        x=-0.1,
        y=0.4,
        s="Cluster",
        rotation=90,
        fontsize=8,
        transform=ax_y_label.transAxes,
        ha="center",
        va="center",
    )

    fig.tight_layout()

    if save_figure is not None and save_dir is not None and save_name is not None:
        save_figure(save_dir, save_name)

    return fig



def plot_cluster_stim_lcd_traces(
    cluster_dict: Mapping[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    cluster_stim_stats: Mapping[str, Dict[str, Any]],
    which: str = "both",            # "both", "high", "low"
    lowN: int = 3,
    highN: int = 12,
    min_cells_per_cluster: int = 5,

    # time / stimulus structure
    sample_rate_hz: float = 2.0,      # 2 Hz -> 0.5 s per sample
    xtick_step: int = 50,             # in samples
    offset_step: float = 0.5,         # vertical spacing between clusters

    # labels
    use_order_label: bool = True,

    # saving
    figsize: Tuple[float, float] = (1.35, 1.4),
    save_figure: Optional[Callable[[str, str], None]] = None,
    save_dir: str = "./Figure_Panels/",
    save_name: str = "LCD_cluster_stimulus_lcd_traces",
) -> plt.Figure:
    """
    Plot average stimulus-locked LCD contribution traces for extreme clusters.

    Uses precomputed cluster_stim_stats from compute_cluster_stim_lcd_stats.

    For each cluster:
        - 'time_mean' : (n_stim, block_len) mean timecourse per stimulus
        - 'time_sem'  : (n_stim, block_len) SEM per stimulus

    We flatten across stimuli to obtain a 1D trace of length n_stim * block_len,
    with each block having a pre/stim/post structure (durations taken from
    cluster_stim_stats). Stimulus epochs are indicated by axvspan shading.

    Cluster selection and ranking:
        - If any cluster has non-zero 'order' in color_dict:
              use those with order > 0, sorted ascending by 'order'.
        - Else:
              use all clusters, sorted by mean_lcd (ascending).
        - low_clusters  = first lowN
        - high_clusters = last highN
        - 'which' selects high / low / both.

    Lines are stacked vertically with 'offset_step' between clusters.
    """
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")

    # ------------------------------------------------------------------
    # 1) Collect cluster-level LCD timecourse info
    # ------------------------------------------------------------------
    cluster_infos = []
    time_length = None
    n_stim_ref = None
    block_len_ref = None
    pre_ref = None
    stim_ref = None
    post_ref = None

    for cid, c in cluster_dict.items():
        if cid not in color_dict:
            continue
        if cid == "all clusters":
            continue

        stats = cluster_stim_stats.get(cid, None)
        if stats is None:
            continue

        time_mean = np.asarray(stats.get("time_mean"))
        time_sem = np.asarray(stats.get("time_sem"))

        if time_mean.ndim != 2 or time_sem.shape != time_mean.shape:
            continue

        n_stim, block_len = time_mean.shape

        if n_stim_ref is None:
            n_stim_ref = n_stim
            block_len_ref = block_len
            pre_ref = int(stats["pre_duration"])
            stim_ref = int(stats["stim_duration"])
            post_ref = int(stats["post_duration"])
        else:
            if n_stim != n_stim_ref or block_len != block_len_ref:
                raise ValueError(
                    f"Inconsistent (n_stim, block_len) for cluster {cid!r}: "
                    f"{(n_stim, block_len)} vs ref {(n_stim_ref, block_len_ref)}."
                )

        # Flatten across stimuli: (n_stim * block_len,)
        mean_flat = time_mean.reshape(-1)
        sem_flat = time_sem.reshape(-1)

        if time_length is None:
            time_length = mean_flat.shape[0]
        else:
            if mean_flat.shape[0] != time_length:
                raise ValueError(
                    f"Trace length mismatch for cluster {cid}: "
                    f"{mean_flat.shape[0]} vs expected {time_length}"
                )

        size = int(c.get("size", c.get("coords", np.empty((0, 3))).shape[0]))
        if size < min_cells_per_cluster:
            continue

        # mean_lcd for ranking (same logic as other plots)
        mean_lcd = c.get("mean_lcd", None)
        if mean_lcd is None:
            vals = np.asarray(c.get("vals", []), dtype=float)
            if vals.ndim == 2:
                v = vals[:, 0]
            else:
                v = vals
            mean_lcd = float(np.nanmean(v)) if v.size > 0 else np.nan

        cd_entry = color_dict.get(cid, {})
        color = np.asarray(cd_entry.get("color", (0, 0, 0, 1.0)))

        if use_order_label:
            label = cd_entry.get("order", cid)
        else:
            label = cd_entry.get("name", cid).split(' ')[0]

        order = cd_entry.get("order", 0)
        if order is None:
            order = 0

        cluster_infos.append(
            dict(
                cluster_id=cid,
                size=size,
                mean_lcd=float(mean_lcd),
                mean_trace=mean_flat,
                sem_trace=sem_flat,
                color=color,
                label=label,
                order=int(order),
            )
        )

    if len(cluster_infos) == 0:
        raise ValueError(
            "No clusters to plot after filtering / missing cluster_stim_stats."
        )

    # ------------------------------------------------------------------
    # 2) Determine extremes (same logic as plot_cluster_stim_responses)
    # ------------------------------------------------------------------
    orders = np.array([ci["order"] for ci in cluster_infos], dtype=int)
    has_ranking = np.any(orders != 0)

    if has_ranking:
        extremes = [ci for ci in cluster_infos if ci["order"] > 0]
        if len(extremes) == 0:
            raise ValueError("Ranking present but none have order > 0.")
        extremes_sorted = sorted(extremes, key=lambda ci: ci["order"])
    else:
        extremes_sorted = sorted(cluster_infos, key=lambda ci: ci["mean_lcd"])

    n_ext = len(extremes_sorted)
    eff_lowN = min(lowN, n_ext)
    eff_highN = min(highN, max(0, n_ext - eff_lowN))

    low_clusters = extremes_sorted[:eff_lowN]
    high_clusters = extremes_sorted[-eff_highN:] if eff_highN > 0 else []

    if which == "high":
        low_clusters = []
    elif which == "low":
        high_clusters = []

    if len(high_clusters) == 0 and len(low_clusters) == 0:
        raise ValueError("No clusters to plot.")

    N_high = len(high_clusters)
    N_low = len(low_clusters)

    # ------------------------------------------------------------------
    # 3) Layout: similar to plot_cluster_stim_responses
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=[
            N_high if N_high > 0 else 0.01,
            N_low if N_low > 0 else 0.01,
        ],
        width_ratios=[0.3, 1.0],
        hspace=0.1,
        wspace=-0.25,
    )

    ax_high = fig.add_subplot(gs[0, 1]) if N_high > 0 else None
    ax_low = fig.add_subplot(gs[1, 1]) if N_low > 0 else None

    x = np.arange(time_length)

    # ------------------------------------------------------------------
    # 4) LOW block traces
    # ------------------------------------------------------------------
    if N_low > 0:
        for i, ci in enumerate(low_clusters):
            mean = ci["mean_trace"]
            sem = ci["sem_trace"]
            color = ci["color"]
            offset = i * offset_step

            ax_low.plot(x, mean + offset, color=color, lw=0.6, zorder=3)
            ax_low.plot(x, np.zeros_like(x) + offset, color='k', lw=.5, linestyle=':', alpha=.5)
            ax_low.fill_between(
                x,
                mean + offset - sem,
                mean + offset + sem,
                color=color,
                alpha=0.4,
                linewidth=0,
                zorder=2,
            )

    # ------------------------------------------------------------------
    # 5) HIGH block traces
    # ------------------------------------------------------------------
    if N_high > 0:
        for i, ci in enumerate(high_clusters):
            mean = ci["mean_trace"]
            sem = ci["sem_trace"]
            color = ci["color"]
            offset = i * offset_step

            ax_high.plot(x, mean + offset, color=color, lw=0.6, zorder=3)
            ax_high.plot(x, np.zeros_like(x) + offset, color='k', lw=.5, linestyle=':', alpha=.5)
            ax_high.fill_between(
                x,
                mean + offset - sem,
                mean + offset + sem,
                color=color,
                alpha=0.4,
                linewidth=0,
                zorder=2,
            )

    # ------------------------------------------------------------------
    # 6) Stimulus shading & cosmetics
    # ------------------------------------------------------------------
    stim_block_len = block_len_ref
    pre = pre_ref
    stim = stim_ref

    axes_to_format = []
    if ax_high is not None:
        axes_to_format.append(ax_high)
    if ax_low is not None:
        axes_to_format.append(ax_low)

    for ax in axes_to_format:
        # One block per stimulus: shade stim epochs
        for s in range(n_stim_ref):
            start = s * stim_block_len + pre
            end = s * stim_block_len + pre + stim
            ax.axvspan(start, end, color="k", alpha=0.08, lw=0)

        if ax is ax_low:
            xticks = np.arange(0, time_length, xtick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels((xticks / sample_rate_hz).astype(int))
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlim(0, time_length)
        else:
            ax.spines[["right", "top", "bottom"]].set_visible(False)
            ax.set_xticks([])
            ax.set_xlim(0, time_length)

    # y-ticks / labels
    if ax_low is not None and N_low > 0:
        ax_low.set_yticks(np.arange(N_low) * offset_step)
        ax_low.set_yticklabels([ci["label"] for ci in low_clusters], fontsize=6)

    if ax_high is not None and N_high > 0:
        ax_high.set_yticks(np.arange(N_high) * offset_step)
        ax_high.set_yticklabels([ci["label"] for ci in high_clusters], fontsize=6)

    # Left label column
    ax_label = fig.add_subplot(gs[:, 0])
    ax_label.axis("off")
    ax_label.text(
        x=-1.0,
        y=0.4,
        s="Cluster",
        rotation=90,
        fontsize=8,
        transform=ax_label.transAxes,
        ha="center",
        va="center",
    )

    fig.tight_layout()

    if save_figure is not None:
        save_figure(save_dir, save_name)

    return fig


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from typing import Mapping, Dict, Any, Sequence, Optional, Tuple, Callable

from WARP.lcd_statistics import get_sig_asterisk_inds
from WARP.lcd_statistics import calc_wilcoxon_significances


def plot_cluster_behavior_corr_split_violins(
    cluster_dict: Mapping[str, Dict[str, Any]],
    color_dict: Mapping[str, Dict[str, Any]],
    fish_data: Mapping[Any, Dict[str, Any]],
    fish_inspect: Sequence[Any],
    stim_keys: Sequence[str],            # 1 or 2 keys
    which: str = "both",                 # "both", "high", "low"
    lowN: int = 3,
    highN: int = 12,
    min_cells_per_cluster: int = 5,
    use_order_label: bool = True,
    rank_by: Optional[str] = None,       # None -> "median" (1 stim) or "delta_median" (2 stim)
    filter_percentile: float = 2.5,      # plotting-only trimming
    bw_method: float = 0.2,
    violin_width: float = 0.42,
    show_medians: bool = True,
    figsize: Tuple[float, float] = (2.2, 1.6),
    save_figure: Optional[Callable[[str, str], None]] = None,
    save_dir: str = "./Figure_Panels/",
    save_name: str = "cluster_behavior_split_violins",
    max_xticks: int = 7,
    show_significance: bool = True,
    csv_file: Optional[str] = None,
    lcd_time_key: str = "lcd_vals_time",   # cluster_dict[cid][lcd_time_key][stim]
    lcd_channel: int = 0,                  # pick channel 0 if LCD array is 3D
    min_valid_timepoints: int = 5,         # per-neuron timepoints required after NaN masking
) -> plt.Figure:
    which = which.lower()
    if which not in ("both", "high", "low"):
        raise ValueError("which must be one of 'both', 'high', or 'low'.")
    stim_keys = list(stim_keys) if stim_keys is not None else []
    if len(stim_keys) < 1 or len(stim_keys) > 2:
        raise ValueError("stim_keys must contain 1 or 2 keys.")
    fish_inspect = list(fish_inspect)
    if rank_by is None:
        rank_by = "median" if len(stim_keys) == 1 else "delta_median"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clean_finite(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        return x[np.isfinite(x)]

    def _trim_for_plotting(x: np.ndarray) -> np.ndarray:
        x = _clean_finite(x)
        if x.size == 0:
            return x
        if filter_percentile is not None and filter_percentile > 0 and x.size >= 5:
            lo, hi = np.percentile(x, [filter_percentile, 100 - filter_percentile])
            x = x[(x >= lo) & (x <= hi)]
        return x

    def _kde(x: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
        if x.size < 2 or np.allclose(np.std(x), 0):
            d = np.zeros_like(xgrid)
            if x.size:
                m = float(np.median(x))
                d[int(np.argmin(np.abs(xgrid - m)))] = 1.0
            return d
        return gaussian_kde(x, bw_method=bw_method)(xgrid)

    def _draw_full_violin(ax, x_plot, y, color):
        if x_plot.size == 0:
            return
        xmin, xmax = float(np.min(x_plot)), float(np.max(x_plot))
        if np.isclose(xmin, xmax):
            xmin -= 1e-3
            xmax += 1e-3
        xgrid = np.linspace(xmin, xmax, 256)
        dens = _kde(x_plot, xgrid)
        dens = dens / max(dens.max(initial=0), 1e-12) * violin_width
        ax.fill_between(xgrid, y - dens, y + dens, facecolor=color, edgecolor="none", alpha=0.85, zorder=2)
        if show_medians:
            m = float(np.median(x_plot))
            ax.plot([m, m], [y - violin_width * 0.95, y + violin_width * 0.95],
                    color=color, lw=0.9, alpha=0.95, zorder=3)

    def _draw_split_violin(ax, x_top_plot, x_bot_plot, y, color):
        x_all = np.concatenate([x_top_plot, x_bot_plot]) if (x_top_plot.size and x_bot_plot.size) else (
            x_top_plot if x_top_plot.size else x_bot_plot
        )
        if x_all.size == 0:
            return
        xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
        if np.isclose(xmin, xmax):
            xmin -= 1e-3
            xmax += 1e-3
        xgrid = np.linspace(xmin, xmax, 256)

        d_top = _kde(x_top_plot, xgrid) if x_top_plot.size else np.zeros_like(xgrid)
        d_bot = _kde(x_bot_plot, xgrid) if x_bot_plot.size else np.zeros_like(xgrid)
        maxd = max(d_top.max(initial=0), d_bot.max(initial=0), 1e-12)
        d_top = d_top / maxd * violin_width
        d_bot = d_bot / maxd * violin_width

        ax.fill_between(xgrid, y, y + d_top, facecolor=color, edgecolor="none", alpha=0.85, zorder=2)
        ax.fill_between(xgrid, y, y - d_bot, facecolor=color, edgecolor="none", alpha=0.35, zorder=2)

        if show_medians:
            if x_top_plot.size:
                m = float(np.median(x_top_plot))
                ax.plot([m, m], [y, y + violin_width * 0.95], lw=0.9, alpha=0.95, zorder=3, color='k')
            if x_bot_plot.size:
                m = float(np.median(x_bot_plot))
                ax.plot([m, m], [y, y - violin_width * 0.95], lw=0.9, alpha=0.65, zorder=3, color='grey')

    def _nice_ticks(xmin: float, xmax: float, max_ticks: int = 7) -> np.ndarray:
        if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
            return np.array([0.0])
        span = xmax - xmin
        raw_step = span / max(1, (max_ticks - 1))
        exp = np.floor(np.log10(raw_step))
        base = raw_step / (10 ** exp)
        candidates = np.array([1.0, 2.0, 2.5, 5.0, 10.0])
        step = candidates[np.argmin(np.abs(candidates - base))] * (10 ** exp)

        start = np.floor(xmin / step) * step
        end = np.ceil(xmax / step) * step
        ticks = np.arange(start, end + 0.5 * step, step)
        ticks = ticks[(ticks >= xmin - 1e-12) & (ticks <= xmax + 1e-12)]
        while ticks.size > max_ticks:
            step *= 2
            start = np.floor(xmin / step) * step
            end = np.ceil(xmax / step) * step
            ticks = np.arange(start, end + 0.5 * step, step)
            ticks = ticks[(ticks >= xmin - 1e-12) & (ticks <= xmax + 1e-12)]
        return ticks

    def _extract_lcd_time_matrix(arr: np.ndarray, channel: int = 0) -> np.ndarray:
        """
        Accept (N,T) or (N,2,T) or (N,T,2) and return (N,T) for the requested channel.
        """
        a = np.asarray(arr[:, 0], dtype=float)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            if a.shape[1] == 2:      # (N,2,T)
                return a[:, channel, :]
            if a.shape[2] == 2:      # (N,T,2)
                return a[:, :, channel]
        raise ValueError(f"Unexpected lcd time array shape {a.shape}; expected (N,T) or (N,2,T) or (N,T,2).")

    def _rowwise_corr_ignore_nan(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Corr(row_i(X), y) ignoring NaNs in X per row and NaNs in y.
        Returns (n_rows,), NaN if too few valid points or zero variance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be (n,T), got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[1]:
            raise ValueError(f"y must be (T,), got {y.shape} while X is {X.shape}")

        y_ok = np.isfinite(y)
        M = np.isfinite(X) & y_ok[None, :]
        n = M.sum(axis=1)

        out = np.full(X.shape[0], np.nan, dtype=float)
        good = n >= min_valid_timepoints
        if not np.any(good):
            return out

        Xf = np.where(M, X, 0.0)
        Y = y[None, :]
        Yf = np.where(M, Y, 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            mx = Xf.sum(axis=1) / n
            my = Yf.sum(axis=1) / n

        Xc = (Xf - mx[:, None]) * M
        Yc = (Yf - my[:, None]) * M

        num = (Xc * Yc).sum(axis=1)
        den = np.sqrt((Xc * Xc).sum(axis=1) * (Yc * Yc).sum(axis=1))

        ok = good & np.isfinite(den) & (den > 0)
        out[ok] = num[ok] / den[ok]
        return out

    def _split_rows_by_fish(cid: str, X: np.ndarray) -> Dict[str, slice]:
        """
        Build slices into X (rows) per fish, assuming X is concatenated
        in the same order as fish_inspect and using neuron_inds_by_fish lengths.
        """
        c = cluster_dict[cid]
        n_by_fish = []
        keys = []
        for fish_n in fish_inspect:
            inds = c.get("neuron_inds_by_fish", {}).get(str(fish_n), None)
            n_i = 0 if inds is None else len(inds)
            n_by_fish.append(n_i)
            keys.append(str(fish_n))

        if int(np.sum(n_by_fish)) != int(X.shape[0]):
            raise ValueError(
                f"Cluster {cid!r}: lcd time rows ({X.shape[0]}) do not match "
                f"sum(len(neuron_inds_by_fish)) ({np.sum(n_by_fish)}). "
                f"Check concatenation order / contents."
            )

        out = {}
        start = 0
        for k, n_i in zip(keys, n_by_fish):
            out[k] = slice(start, start + n_i)
            start += n_i
        return out

    # ------------------------------------------------------------------
    # 2) Collect cluster info + compute per-neuron corr(LCD_time, swim) *from cluster_dict*
    # ------------------------------------------------------------------
    cluster_infos = []
    for cid, c in cluster_dict.items():
        if cid not in color_dict or cid == "all clusters":
            continue

        size = int(c.get("size", c.get("coords", np.empty((0, 3))).shape[0]))
        if size < min_cells_per_cluster:
            continue

        if lcd_time_key not in c:
            raise KeyError(f"Cluster {cid!r} missing key {lcd_time_key!r}.")
        if not isinstance(c[lcd_time_key], dict):
            raise ValueError(f"Cluster {cid!r}[{lcd_time_key!r}] must be a dict stim->array.")

        cd_entry = color_dict.get(cid, {})
        color = np.asarray(cd_entry.get("color", (0, 0, 0, 1.0)))
        order = cd_entry.get("order", 0)
        order = 0 if order is None else int(order)
        label = cd_entry.get("order", cid) if use_order_label else cd_entry.get("name", cid)

        dist_raw_by_stim = {}
        dist_plot_by_stim = {}
        any_nonempty = False

        for stim in stim_keys:
            if stim not in c[lcd_time_key]:
                raise KeyError(f"Cluster {cid!r}[{lcd_time_key!r}] missing stim {stim!r}.")

            # cluster-level lcd time traces (concatenated across fish)
            X_full = _extract_lcd_time_matrix(c[lcd_time_key][stim], channel=lcd_channel)  # (Nrows, T)

            # slice rows per fish, then correlate each row with that fish's swim
            slices = _split_rows_by_fish(cid, X_full)

            corrs_parts = []
            for fish_n in fish_inspect:
                fish_key = str(fish_n)
                sl = slices[fish_key]
                if sl.stop == sl.start:
                    continue

                swim = np.asarray(
                    fish_data[fish_n]["ephys_data"][stim]["swim_signal_downsampled"],
                    dtype=float
                ).ravel()

                X_fish = X_full[sl, :]
                # if lengths mismatch, truncate to min (common in pipelines)
                T = min(X_fish.shape[1], swim.shape[0])
                corr_rows = _rowwise_corr_ignore_nan(X_fish[:, :T], swim[:T])
                corrs_parts.append(corr_rows)

            arr = np.concatenate(corrs_parts, axis=0) if len(corrs_parts) else np.asarray([], dtype=float)
            arr_raw = _clean_finite(arr)
            arr_plot = _trim_for_plotting(arr_raw)

            dist_raw_by_stim[stim] = arr_raw
            dist_plot_by_stim[stim] = arr_plot
            any_nonempty = any_nonempty or (arr_raw.size > 0)

        if not any_nonempty:
            continue

        # score for fallback sorting
        if len(stim_keys) == 1:
            score = float(np.nanmedian(dist_raw_by_stim[stim_keys[0]])) if dist_raw_by_stim[stim_keys[0]].size else np.nan
        else:
            a = dist_raw_by_stim[stim_keys[0]]
            b = dist_raw_by_stim[stim_keys[1]]
            med_a = float(np.nanmedian(a)) if a.size else np.nan
            med_b = float(np.nanmedian(b)) if b.size else np.nan
            if rank_by == "delta_median":
                score = med_a - med_b
            elif rank_by == "stim0_median":
                score = med_a
            elif rank_by == "stim1_median":
                score = med_b
            else:
                raise ValueError("rank_by must be None/'delta_median'/'stim0_median'/'stim1_median' for 2 stim_keys.")

        cluster_infos.append(
            dict(
                cluster_id=cid,
                size=size,
                score=score,
                dist_raw_by_stim=dist_raw_by_stim,
                dist_plot_by_stim=dist_plot_by_stim,
                color=color,
                label=label,
                order=order,
            )
        )

    if len(cluster_infos) == 0:
        raise ValueError("No clusters to plot after filtering.")

    # ------------------------------------------------------------------
    # 3) Wilcoxon p-values vs 0 (per stim_key) + Bonferroni + asterisks
    # ------------------------------------------------------------------
    cluster_ids_all = [ci["cluster_id"] for ci in cluster_infos]
    stars_by_cluster = {stim: {} for stim in stim_keys}

    for stim in stim_keys:
        diff_vals = np.array([ci["dist_raw_by_stim"][stim] for ci in cluster_infos], dtype=object)

        p_vals, p_vals_bonf = calc_wilcoxon_significances(
            diff_vals,
            alternative="two-sided",
            genes=cluster_ids_all,
            verbose=True,
            verbose_ordering=np.arange(len(cluster_ids_all), dtype=int),
            csv_file=csv_file,
            csv_colname="Cluster",
        )
        p_vals_bonf = np.asarray(p_vals_bonf, dtype=float)

        import pandas as pd

        df = pd.DataFrame({
            "gene": cluster_ids_all,
            "p_value_uncorr": p_vals,
            "p_value": p_vals_bonf
        })
        
        df.to_excel(f"swimcorr_pvals_{stim}.xlsx", index=False)

        _, sig_asterisks = get_sig_asterisk_inds(p_vals_bonf)
        for i, cid in enumerate(cluster_ids_all):
            stars_by_cluster[stim][cid] = sig_asterisks.get(i, "")

    # ------------------------------------------------------------------
    # 4) Determine extremes + invert display order
    # ------------------------------------------------------------------
    orders = np.array([ci["order"] for ci in cluster_infos], dtype=int)
    has_ranking = np.any(orders != 0)

    if has_ranking:
        extremes = [ci for ci in cluster_infos if ci["order"] > 0]
        if len(extremes) == 0:
            raise ValueError("Ranking present but none have order > 0.")
        extremes_sorted = sorted(extremes, key=lambda ci: ci["order"])
    else:
        extremes_sorted = sorted(cluster_infos, key=lambda ci: ci["score"])

    n_ext = len(extremes_sorted)
    eff_lowN = min(lowN, n_ext)
    eff_highN = min(highN, max(0, n_ext - eff_lowN))

    low_clusters = extremes_sorted[:eff_lowN]
    high_clusters = extremes_sorted[-eff_highN:] if eff_highN > 0 else []

    if which == "high":
        low_clusters = []
    elif which == "low":
        high_clusters = []

    if len(high_clusters) == 0 and len(low_clusters) == 0:
        raise ValueError("No clusters to plot.")

    low_clusters = list(reversed(low_clusters))
    high_clusters = list(reversed(high_clusters))

    N_high = len(high_clusters)
    N_low = len(low_clusters)
    has_two_blocks = (N_high > 0 and N_low > 0)

    # ------------------------------------------------------------------
    # 5) Shared xlim/ticks from plotted data
    # ------------------------------------------------------------------
    all_vals_plot = []
    for ci in (high_clusters + low_clusters):
        for stim in stim_keys:
            v = ci["dist_plot_by_stim"][stim]
            if v.size:
                all_vals_plot.append(v)
    all_vals_plot = np.concatenate(all_vals_plot, axis=0) if len(all_vals_plot) else np.array([])

    if all_vals_plot.size:
        lo, hi = (np.percentile(all_vals_plot, 1), np.percentile(all_vals_plot, 99)) if all_vals_plot.size >= 20 else (
            all_vals_plot.min(), all_vals_plot.max()
        )
        pad = 0.05 * (hi - lo + 1e-12)
        xlim = (float(lo - pad), float(hi + pad))
    else:
        xlim = (-1.0, 1.0)

    xticks = _nice_ticks(xlim[0], xlim[1], max_ticks=max_xticks)

    # ------------------------------------------------------------------
    # 6) Layout + plotting
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        height_ratios=[N_high if N_high > 0 else 0.01, N_low if N_low > 0 else 0.01],
        width_ratios=[0.3, 1.0],
        hspace=0.1,
        wspace=-0.25,
    )
    ax_high = fig.add_subplot(gs[0, 1]) if N_high > 0 else None
    ax_low = fig.add_subplot(gs[1, 1], sharex=ax_high) if N_low > 0 else None

    def _plot_block(ax, cluster_list, show_xticks: bool):
        n = len(cluster_list)
        ys = np.arange(n)[::-1]
        ax.set_xlim(*xlim)

        for i, ci in enumerate(cluster_list):
            y = ys[i]
            cid = ci["cluster_id"]

            if len(stim_keys) == 1:
                _draw_full_violin(ax, ci["dist_plot_by_stim"][stim_keys[0]], y, ci["color"])
            else:
                _draw_split_violin(
                    ax,
                    ci["dist_plot_by_stim"][stim_keys[0]],
                    ci["dist_plot_by_stim"][stim_keys[1]],
                    y,
                    ci["color"],
                )

            if show_significance:
                x_star = xlim[1] + 0.05 * (xlim[1] - xlim[0] + 1e-12)
                if len(stim_keys) == 1:
                    s = stars_by_cluster[stim_keys[0]].get(cid, "")
                    if s != "":
                        ax.text(x_star, y, s, ha="right", va="center", fontsize=6)
                else:
                    # vertically centered within each half (top/bottom)
                    y_off = violin_width * 0.65
                    s_top = stars_by_cluster[stim_keys[0]].get(cid, "")
                    s_bot = stars_by_cluster[stim_keys[1]].get(cid, "")
                    if s_top != "":
                        ax.text(x_star, y + y_off, s_top, ha="right", va="center", fontsize=5)
                    if s_bot != "":
                        ax.text(x_star, y - y_off, s_bot, ha="right", va="center", fontsize=5, color="gray")

        ax.set_yticks(ys)
        ax.set_yticklabels([ci["label"].split(' ')[0] for ci in cluster_list], fontsize=6)
        ax.grid(True, axis='x', alpha=0.25)
        ax.axvline(0, lw=0.8, alpha=0.25, color='k')

        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y")
        ax.tick_params(axis="x", labelsize=7)

        if show_xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=7)
        else:
            ax.set_xticks([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)

    if ax_high is not None:
        _plot_block(ax_high, high_clusters, show_xticks=(not has_two_blocks))
        if has_two_blocks:
            ax_high.spines["bottom"].set_visible(False)

    if ax_low is not None:
        _plot_block(ax_low, low_clusters, show_xticks=True)
        ax_low.set_xlabel("Correlation", fontsize=8)

    ax_label = fig.add_subplot(gs[:, 0])
    ax_label.axis("off")
    ax_label.text(
        x=-1.0, y=0.4, s="Cluster",
        rotation=90, fontsize=8, transform=ax_label.transAxes,
        ha="center", va="center",
    )

    fig.tight_layout()

    if save_figure is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_figure(save_dir, save_name)

    return fig


