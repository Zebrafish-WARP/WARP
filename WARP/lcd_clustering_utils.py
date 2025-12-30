import numpy as np
from typing import (
    Sequence,
    Dict,
    Any,
    List,
    Optional,
    Tuple,
    Union,
    Mapping,
    Literal,
)

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KDTree

from tqdm.auto import tqdm

from WARP.visualization import make_cmap_dict  # might be useful elsewhere
from WARP.utils import symmetrical_project


# =====================================================================
# Fish support logic
# =====================================================================
def fish_support_ok(
    fish_counts: np.ndarray,
    min_frac_per_fish: float,
    min_count_per_fish: int,
) -> bool:
    """
    Decide whether a cluster (parcel) has sufficient support from all fish.
    """
    fish_counts = np.asarray(fish_counts, dtype=int)
    total = int(fish_counts.sum())
    if total == 0:
        return False

    n_fish = fish_counts.shape[0]
    percs = fish_counts / total

    # Minimum absolute count per fish
    if np.any(fish_counts < min_count_per_fish):
        return False

    # Minimum relative fraction per fish (relative to 1/n_fish)
    threshold = min_frac_per_fish * (1.0 / n_fish)
    if np.any(percs < threshold):
        return False

    return True


# =====================================================================
# Geometry-only parcellation with multiple modes: "ward" / "leiden"
# =====================================================================
def _leiden_cluster_from_connectivity(
    connectivity,
    n_nodes: int,
    n_parcels_guess: int,
    random_state: Optional[int] = None,
    mode_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run Leiden community detection on a kNN connectivity graph.
    """
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as e:
        raise ImportError(
            "Leiden clustering requested (cluster_mode='leiden'), "
            "but 'igraph' and 'leidenalg' are not installed. "
            "Install them via e.g. 'pip install python-igraph leidenalg'."
        ) from e

    if mode_kwargs is None:
        mode_kwargs = {}

    # Build igraph Graph from connectivity
    coo = connectivity.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)

    rng = np.random.default_rng(random_state)
    seed = int(rng.integers(0, 2**31 - 1))

    user_res = mode_kwargs.get("resolution", None)

    if user_res is not None:
        resolutions = [float(user_res)]
    else:
        resolutions = [0.2, 0.5, 1.0, 2.0, 5.0]

    best_labels = None
    best_res = None
    best_diff = None
    best_n_comms = None

    for res in resolutions:
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            resolution_parameter=res,
            seed=seed,
        )
        labels = np.array(part.membership, dtype=int)
        n_comms = len(part)

        diff = abs(n_comms - n_parcels_guess)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_labels = labels
            best_res = res
            best_n_comms = n_comms

    meta = {
        "resolution": best_res,
        "n_communities": best_n_comms,
        "n_parcels_guess": int(n_parcels_guess),
    }
    return best_labels, meta


def make_spatial_parcels(
    coords: np.ndarray,
    n_neighbors: int = 25,
    target_cluster_size: int = 50,
    n_parcels: Optional[int] = None,
    cluster_mode: str = "ward",
    mode_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[Any], Any]:
    """
    Build a fixed spatial partition (parcels) based only on neuron coordinates.
    """
    coords = np.asarray(coords, dtype=float)
    n_neurons = coords.shape[0]

    if n_neurons == 0:
        raise ValueError("make_spatial_parcels: coords is empty.")

    if n_parcels is None:
        target_cluster_size = max(1, int(target_cluster_size))
        n_parcels = max(1, int(round(n_neurons / target_cluster_size)))
    else:
        n_parcels = int(n_parcels)

    cluster_mode = cluster_mode.lower()
    if cluster_mode not in ("ward", "leiden"):
        raise ValueError("cluster_mode must be 'ward' or 'leiden'.")

    if mode_kwargs is None:
        mode_kwargs = {}

    # Build connectivity graph
    if n_neighbors is not None and n_neighbors > 0:
        connectivity = kneighbors_graph(
            coords,
            n_neighbors=n_neighbors,
            include_self=False,
            mode="connectivity",
        )
    else:
        connectivity = None

    if cluster_mode == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_parcels,
            connectivity=connectivity,
            linkage="ward",
        )
        parcel_labels = model.fit_predict(coords)
        return parcel_labels, connectivity, model

    if connectivity is None:
        raise ValueError(
            "Leiden clustering (cluster_mode='leiden') requires a non-None "
            "kNN connectivity graph. Set n_neighbors > 0."
        )

    labels, meta = _leiden_cluster_from_connectivity(
        connectivity=connectivity,
        n_nodes=n_neurons,
        n_parcels_guess=n_parcels,
        random_state=mode_kwargs.get("random_state", None),
        mode_kwargs=mode_kwargs,
    )
    return labels, connectivity, meta


# =====================================================================
# Find spatial LCD clusters for a single gene (single-scale, no perms)
# =====================================================================
def find_lcd_clusters_for_gene(
    coords: np.ndarray,
    vals: np.ndarray,
    fish_ids: Sequence,
    n_neighbors: int = 25,
    target_cluster_size: Union[int, Sequence[int]] = 50,
    n_parcels: Optional[int] = None,
    min_cluster_size: int = 30,
    min_frac_per_fish: float = 0.6,
    min_count_per_fish: int = 5,
    random_state: Optional[int] = None,  # used for Leiden, kept for API
    precomputed_parcel_labels: Optional[np.ndarray] = None,
    cluster_mode: str = "ward",
    mode_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Find spatially localized LCD clusters for a single gene using a spatial
    partition (parcels) and fish-support criteria.

    IMPORTANT: This is **single-scale** and **no longer uses permutations**.
    Each neuron belongs to exactly one spatial parcel.
    """
    # --- normalize target_cluster_size to a single int ---
    if isinstance(target_cluster_size, Sequence) and not isinstance(
        target_cluster_size, (str, bytes)
    ):
        tcs_list = list(target_cluster_size)
        if len(tcs_list) == 0:
            raise ValueError(
                "target_cluster_size was an empty sequence; please pass an int."
            )
        if len(tcs_list) > 1:
            print(
                "Warning: target_cluster_size is a sequence; "
                "only the first element is used in single-scale mode."
            )
        target_cluster_size_eff = int(tcs_list[0])
    else:
        target_cluster_size_eff = int(target_cluster_size)

    coords = np.asarray(coords, dtype=float)
    vals = np.asarray(vals, dtype=float)
    fish_ids = np.asarray(fish_ids)

    if coords.shape[0] != vals.shape[0] or vals.shape[0] != fish_ids.shape[0]:
        raise ValueError("coords, vals, and fish_ids must have the same length.")

    # Joint NaN mask: drop neurons with NaN coords or NaN vals
    nan_mask = np.isnan(vals) | np.isnan(coords).any(axis=1)
    if np.any(nan_mask):
        coords = coords[~nan_mask]
        vals = vals[~nan_mask]
        fish_ids = fish_ids[~nan_mask]

        if precomputed_parcel_labels is not None:
            precomputed_parcel_labels = np.asarray(precomputed_parcel_labels)
            if precomputed_parcel_labels.shape[0] != nan_mask.shape[0]:
                raise ValueError(
                    "precomputed_parcel_labels must have shape (n_neurons,) "
                    "before NaN filtering."
                )
            precomputed_parcel_labels = precomputed_parcel_labels[~nan_mask]

    n_samples = vals.shape[0]
    if n_samples == 0:
        raise ValueError("No valid neurons for this gene after NaN filtering.")

    # Encode fish IDs as consecutive integers 0..n_fish-1
    fish_labels, fish_ids_encoded = np.unique(fish_ids, return_inverse=True)
    n_fish = fish_labels.shape[0]

    # ---------------------------------------------------------------
    # Build parcels (single scale)
    # ---------------------------------------------------------------
    if precomputed_parcel_labels is not None:
        parcel_labels = np.asarray(precomputed_parcel_labels, dtype=int)
        if parcel_labels.shape[0] != n_samples:
            raise ValueError(
                "precomputed_parcel_labels must match length of coords/vals "
                "after NaN filtering."
            )
        connectivity = None
        cluster_model = None
    else:
        if mode_kwargs is None:
            mode_kwargs = {}
        mode_kwargs.setdefault("random_state", random_state)

        parcel_labels, connectivity, cluster_model = make_spatial_parcels(
            coords=coords,
            n_neighbors=n_neighbors,
            target_cluster_size=target_cluster_size_eff,
            n_parcels=n_parcels,
            cluster_mode=cluster_mode,
            mode_kwargs=mode_kwargs,
        )

    # Ensure contiguous labels 0..n_parcels_eff-1
    unique_ids = np.unique(parcel_labels)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    parcel_labels = np.array([id_map[l] for l in parcel_labels], dtype=int)

    n_parcels_eff = unique_ids.size
    parcel_ids = np.arange(n_parcels_eff, dtype=int)

    # ---------------------------------------------------------------
    # Compute per-parcel statistics
    # ---------------------------------------------------------------
    inds_list: List[np.ndarray] = []
    sizes = np.empty(n_parcels_eff, dtype=int)
    fish_counts = np.zeros((n_parcels_eff, n_fish), dtype=int)
    means = np.full(n_parcels_eff, np.nan, dtype=float)
    fish_support_flags = np.zeros(n_parcels_eff, dtype=bool)

    for k in range(n_parcels_eff):
        idx = np.where(parcel_labels == k)[0]
        inds_list.append(idx)
        sizes[k] = idx.size

        if idx.size > 0:
            fish_counts[k] = np.bincount(fish_ids_encoded[idx], minlength=n_fish)
            means[k] = float(np.nanmean(vals[idx]))

        if sizes[k] >= min_cluster_size and fish_support_ok(
            fish_counts[k],
            min_frac_per_fish=min_frac_per_fish,
            min_count_per_fish=min_count_per_fish,
        ):
            fish_support_flags[k] = True

    result: Dict[str, Any] = {
        "parcel_labels": parcel_labels,
        "parcel_ids": parcel_ids,
        "means": means,
        "sizes": sizes,
        "fish_counts": fish_counts,
        "fish_support_ok": fish_support_flags,
        "fish_labels": fish_labels,
        "n_neurons": n_samples,
        "target_cluster_size": int(target_cluster_size_eff),
        "connectivity": connectivity,
        "cluster_model": cluster_model,
        "parcel_indices": inds_list,
    }

    return result


# =====================================================================
# NEW helper: assemble global neuron indices (per fish) for gene-expressing neurons
# (keeps existing assemble_* APIs untouched)
# =====================================================================
def assemble_gene_neuron_inds_across_fish(
    gene: str,
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the *global* neuron indices (into full fish_data arrays) for all
    neurons expressing `gene` across fish, in the exact same fish-by-fish
    concatenation order used by assemble_gene_lcd_data_across_fish(...).

    Returns
    -------
    neuron_inds_gene : (N,) int array
        Global neuron indices (per fish) for each concatenated neuron.
    fish_ids_gene : (N,) object array
        Fish id for each concatenated neuron.
    """
    neuron_inds_list: List[np.ndarray] = []
    fish_ids_list: List[np.ndarray] = []

    for fish_n in fish_inspect:
        gene_names = fish_data[fish_n]["gene_data"]["gene_names"]
        gene_inds = np.where(gene_names == gene)[0]
        if gene_inds.size == 0:
            continue
        gene_ind = int(gene_inds[0])

        gc_binary = fish_data[fish_n]["gene_data"]["gene_counts_binary"]
        gene_neuron_inds = np.where(gc_binary[:, gene_ind])[0]
        if gene_neuron_inds.size == 0:
            continue

        neuron_inds_list.append(gene_neuron_inds.astype(int))
        fish_ids_list.append(np.array([fish_n] * gene_neuron_inds.size, dtype=object))

    if len(neuron_inds_list) == 0:
        raise ValueError(f"No neurons found for gene {gene} in the specified fish.")

    neuron_inds_gene = np.concatenate(neuron_inds_list, axis=0)
    fish_ids_gene = np.concatenate(fish_ids_list, axis=0)
    return neuron_inds_gene, fish_ids_gene


# =====================================================================
# Summarize clusters for a gene (ALL parcels, with fish_support flag)
# =====================================================================
def summarize_significant_clusters_for_gene(
    coords: np.ndarray,
    gene_counts: np.ndarray,
    stim_responses_avg: np.ndarray,
    stim_responses_sem: np.ndarray,
    lcd_vals_by_paradigm: Dict[str, np.ndarray],
    fish_ids: Sequence,
    result: Dict[str, Any],
    lcd_vals_time_by_paradigm: Optional[Dict[str, np.ndarray]] = None,
    primary_stim_key: str = "visrap",
    neuron_inds_gene: Optional[np.ndarray] = None,  # NEW, optional; keeps API compatible
) -> List[Dict[str, Any]]:
    """
    Summarize each spatial cluster (parcel) for a gene.

    NEW (optional):
        If `neuron_inds_gene` is provided (global indices into fish_data arrays,
        aligned with the concatenated ordering used for coords/vals/fish_ids),
        then each cluster entry will additionally include:

            cluster["neuron_inds_gene"]      -> (n_cells_cluster,) int
            cluster["neuron_inds_by_fish"]   -> dict fish_id -> (n_cells_fish_cluster,) int

    This lets you re-fetch coords, stim traces, gene counts, etc. from fish_data
    without storing heavy arrays in the cluster dict.
    """
    coords = np.asarray(coords, dtype=float)
    gene_counts = np.asarray(gene_counts, dtype=float)
    stim_responses_avg = np.asarray(stim_responses_avg, dtype=float)
    stim_responses_sem = np.asarray(stim_responses_sem, dtype=float)
    fish_ids = np.asarray(fish_ids)

    if neuron_inds_gene is not None:
        neuron_inds_gene = np.asarray(neuron_inds_gene, dtype=int)
        if neuron_inds_gene.shape[0] != coords.shape[0]:
            raise ValueError(
                "neuron_inds_gene must have the same length as coords before NaN filtering."
            )

    if primary_stim_key not in lcd_vals_by_paradigm:
        raise KeyError(
            f"primary_stim_key '{primary_stim_key}' not found in lcd_vals_by_paradigm."
        )

    vals_primary = np.asarray(lcd_vals_by_paradigm[primary_stim_key], dtype=float)

    if coords.shape[0] != vals_primary.shape[0] or coords.shape[0] != fish_ids.shape[0]:
        raise ValueError(
            "coords, vals_primary, and fish_ids must have the same length before NaN filtering."
        )

    # NaN mask must match logic in find_lcd_clusters_for_gene
    if vals_primary.ndim == 1:
        nan_mask = np.isnan(vals_primary) | np.isnan(coords).any(axis=1)
    else:
        nan_mask = np.isnan(vals_primary[:, 0]) | np.isnan(coords).any(axis=1)

    if np.any(nan_mask):
        coords = coords[~nan_mask]
        gene_counts = gene_counts[~nan_mask]
        stim_responses_avg = stim_responses_avg[~nan_mask]
        stim_responses_sem = stim_responses_sem[~nan_mask]
        fish_ids = fish_ids[~nan_mask]

        if neuron_inds_gene is not None:
            neuron_inds_gene = neuron_inds_gene[~nan_mask]

        # Apply same mask to all paradigms
        for key in list(lcd_vals_by_paradigm.keys()):
            v = np.asarray(lcd_vals_by_paradigm[key])
            if v.shape[0] == nan_mask.shape[0]:
                lcd_vals_by_paradigm[key] = v[~nan_mask]
            else:
                raise ValueError(
                    f"lcd_vals_by_paradigm['{key}'] length {v.shape[0]} "
                    f"does not match coords length {nan_mask.shape[0]} "
                    "before NaN filtering."
                )

        if lcd_vals_time_by_paradigm is not None:
            for key in list(lcd_vals_time_by_paradigm.keys()):
                vt = np.asarray(lcd_vals_time_by_paradigm[key])
                if vt.shape[0] == nan_mask.shape[0]:
                    lcd_vals_time_by_paradigm[key] = vt[~nan_mask]
                else:
                    raise ValueError(
                        f"lcd_vals_time_by_paradigm['{key}'] length {vt.shape[0]} "
                        f"does not match coords length {nan_mask.shape[0]} "
                        "before NaN filtering."
                    )

    parcel_labels = np.asarray(result["parcel_labels"], dtype=int)
    sizes = np.asarray(result["sizes"], dtype=int)
    fish_counts = np.asarray(result["fish_counts"], dtype=int)
    fish_support_flags = np.asarray(result["fish_support_ok"], dtype=bool)
    fish_labels = np.asarray(result["fish_labels"])
    target_cluster_size = int(result["target_cluster_size"])

    n_parcels = parcel_labels.max() + 1
    clusters: List[Dict[str, Any]] = []

    cluster_counter = 1

    for k in range(n_parcels):
        inds = np.where(parcel_labels == k)[0]
        if inds.size == 0:
            continue

        fc_vec = fish_counts[k]
        total = int(fc_vec.sum())

        fish_counts_dict = {
            str(fish_labels[i]): int(fc_vec[i]) for i in range(len(fish_labels))
        }
        fish_percs_dict = {
            str(fish_labels[i]): (fc_vec[i] / total if total > 0 else np.nan)
            for i in range(len(fish_labels))
        }

        vals_primary_cluster = lcd_vals_by_paradigm[primary_stim_key][inds]
        vals_primary_arr = np.asarray(vals_primary_cluster)
        if vals_primary_arr.ndim == 1:
            mean_lcd = float(np.nanmean(vals_primary_arr))
        else:
            mean_lcd = float(np.nanmean(vals_primary_arr[:, 0]))

        size_k = int(sizes[k])
        fish_support_ok_flag = bool(fish_support_flags[k])

        lcd_vals_cluster: Dict[str, np.ndarray] = {}
        for stim_key, v_all in lcd_vals_by_paradigm.items():
            lcd_vals_cluster[stim_key] = v_all[inds]

        lcd_vals_time_cluster: Dict[str, np.ndarray] = {}
        if lcd_vals_time_by_paradigm is not None:
            for stim_key, vt_all in lcd_vals_time_by_paradigm.items():
                lcd_vals_time_cluster[stim_key] = vt_all[inds]

        cluster_entry: Dict[str, Any] = {
            "cluster_id": int(cluster_counter),  # 1..N per gene
            "parcel_idx": int(k),
            "target_cluster_size": target_cluster_size,
            "size": size_k,
            "mean_lcd": mean_lcd,
            "fish_support_ok": fish_support_ok_flag,
            "fish_counts": fish_counts_dict,
            "fish_percs": fish_percs_dict,

            # Existing heavy fields (unchanged behavior)
            "coords": coords[inds],
            "gene_counts": gene_counts[inds],
            "stim_responses_avg": stim_responses_avg[inds],
            "stim_responses_sem": stim_responses_sem[inds],

            "lcd_vals": lcd_vals_cluster,
            "indices": inds,
            "fish_ids": fish_ids[inds],
        }

        if lcd_vals_time_cluster:
            cluster_entry["lcd_vals_time"] = lcd_vals_time_cluster

        # NEW: store global neuron indices per fish if provided
        if neuron_inds_gene is not None:
            neuron_inds_cluster = neuron_inds_gene[inds].astype(int)
            fish_ids_cluster = np.asarray(cluster_entry["fish_ids"], dtype=object)

            neuron_inds_by_fish: Dict[str, np.ndarray] = {}
            for f in np.unique(fish_ids_cluster):
                m = fish_ids_cluster == f
                neuron_inds_by_fish[str(f)] = neuron_inds_cluster[m].astype(int)

            cluster_entry["neuron_inds_gene"] = neuron_inds_cluster
            cluster_entry["neuron_inds_by_fish"] = neuron_inds_by_fish

        clusters.append(cluster_entry)
        cluster_counter += 1

    return clusters


# =====================================================================
# Assemble gene LCD data across fish (unchanged APIs)
# =====================================================================
def assemble_gene_lcd_data_across_fish(
    gene: str,
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_key: str = "visrap",
    distance_bin: int = 20,
    proj_axis: int = 1,
    ref_fish_for_projection: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Assemble coordinates, LCD values and fish IDs for a single gene across multiple fish.
    (API unchanged)
    """
    coords_gene_list: List[np.ndarray] = []
    gc_gene_list: List[np.ndarray] = []
    stim_response_avg_gene_list: List[np.ndarray] = []
    stim_response_sem_gene_list: List[np.ndarray] = []
    vals_gene_list: List[np.ndarray] = []
    fish_ids_gene_list: List[np.ndarray] = []

    for fish_n in fish_inspect:
        gene_names = fish_data[fish_n]["gene_data"]["gene_names"]
        gene_inds = np.where(gene_names == gene)[0]
        if gene_inds.size == 0:
            continue
        gene_ind = gene_inds[0]

        gene_neuron_inds = np.where(
            fish_data[fish_n]["gene_data"]["gene_counts_binary"][:, gene_ind]
        )[0]
        if gene_neuron_inds.size == 0:
            continue

        coords_fish = fish_data[fish_n]["cell_centers_data"]["cell_centers_zb"][
            gene_neuron_inds
        ]
        gc_fish = fish_data[fish_n]["gene_data"]["gene_counts"][gene_neuron_inds]

        stim_response_avg_fish = fish_data[fish_n]["stim_response_data"]["visrap"][
            "avg_stim_responses"
        ][gene_neuron_inds]
        stim_response_sem_fish = fish_data[fish_n]["stim_response_data"]["visrap"][
            "sem_stim_responses"
        ][gene_neuron_inds]

        vals_fish = LCD_data[stim_key][fish_n][gene]["observed_stats"]["distance"][
            distance_bin
        ]["vals"]
        vals_fish = np.asarray(vals_fish, dtype=float)

        if vals_fish.shape[0] != coords_fish.shape[0]:
            raise ValueError(
                f"Mismatch coords vs LCD vals for gene {gene} in fish {fish_n}: "
                f"{coords_fish.shape[0]} coords vs {vals_fish.shape[0]} vals."
            )

        coords_gene_list.append(coords_fish)
        gc_gene_list.append(gc_fish)
        stim_response_avg_gene_list.append(stim_response_avg_fish)
        stim_response_sem_gene_list.append(stim_response_sem_fish)
        vals_gene_list.append(vals_fish)
        fish_ids_gene_list.append(
            np.array([fish_n] * coords_fish.shape[0], dtype=object)
        )

    if len(coords_gene_list) == 0:
        raise ValueError(f"No neurons found for gene {gene} in the specified fish.")

    coords_gene = np.concatenate(coords_gene_list, axis=0)
    gc_gene = np.concatenate(gc_gene_list, axis=0)
    stim_response_avg_gene = np.concatenate(stim_response_avg_gene_list, axis=0)
    stim_response_sem_gene = np.concatenate(stim_response_sem_gene_list, axis=0)
    vals_gene = np.concatenate(vals_gene_list, axis=0)
    fish_ids_gene = np.concatenate(fish_ids_gene_list, axis=0)

    if ref_fish_for_projection is None:
        ref_fish_for_projection = fish_inspect[0]

    coords_all_ref = fish_data[ref_fish_for_projection]["cell_centers_data"][
        "cell_centers_zb"
    ]

    coords_gene_projected = symmetrical_project(
        coords_gene,
        coords_all=coords_all_ref,
        proj_axis=proj_axis,
    )

    return (
        coords_gene,
        gc_gene,
        stim_response_avg_gene,
        stim_response_sem_gene,
        vals_gene,
        fish_ids_gene,
        coords_gene_projected,
    )


def assemble_gene_lcd_data_across_fish_with_time(
    gene: str,
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_key: str = "visrap",
    distance_bin: int = 20,
    proj_axis: int = 1,
    ref_fish_for_projection: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """
    Wrapper around `assemble_gene_lcd_data_across_fish` that also tries to
    assemble time-resolved LCD contributions for this gene across fish.
    (API unchanged)
    """
    (
        coords_gene,
        gc_gene,
        stim_response_avg_gene,
        stim_response_sem_gene,
        vals_gene,
        fish_ids_gene,
        coords_gene_projected,
    ) = assemble_gene_lcd_data_across_fish(
        gene=gene,
        LCD_data=LCD_data,
        fish_data=fish_data,
        fish_inspect=fish_inspect,
        stim_key=stim_key,
        distance_bin=distance_bin,
        proj_axis=proj_axis,
        ref_fish_for_projection=ref_fish_for_projection,
    )

    vals_time_list: List[np.ndarray] = []
    all_have_time = True

    for fish_n in fish_inspect:
        fish_entry = LCD_data.get(stim_key, {}).get(fish_n, {})
        gene_entry = fish_entry.get(gene)
        if gene_entry is None:
            continue

        gene_names = fish_data[fish_n]["gene_data"]["gene_names"]
        gene_inds = np.where(gene_names == gene)[0]
        if gene_inds.size == 0:
            continue
        gene_ind = gene_inds[0]

        gc_binary = fish_data[fish_n]["gene_data"]["gene_counts_binary"]
        gene_neuron_inds = np.where(gc_binary[:, gene_ind])[0]
        if gene_neuron_inds.size == 0:
            continue

        obs_stats_all = gene_entry.get("observed_stats", {})
        obs_stats_family = obs_stats_all.get("distance", {})
        obs_stats = obs_stats_family.get(distance_bin, None)
        if obs_stats is None:
            all_have_time = False
            break

        vals_time_fish = obs_stats.get("vals_time", None)

        if vals_time_fish is None and "vals_time_full" in obs_stats:
            vtf = np.asarray(obs_stats["vals_time_full"], dtype=float)
            if vtf.ndim != 3 or vtf.shape[1] < 1:
                all_have_time = False
                break
            vals_time_fish = vtf[:, :, :]

        if vals_time_fish is None:
            all_have_time = False
            break

        vals_time_fish = np.asarray(vals_time_fish, dtype=float)
        if vals_time_fish.shape[0] != gene_neuron_inds.shape[0]:
            all_have_time = False
            break

        vals_time_list.append(vals_time_fish)

    if (not all_have_time) or len(vals_time_list) == 0:
        vals_time_gene = None
    else:
        vals_time_gene = np.concatenate(vals_time_list, axis=0)
        if vals_time_gene.shape[0] != coords_gene.shape[0]:
            vals_time_gene = None

    return (
        coords_gene,
        gc_gene,
        stim_response_avg_gene,
        stim_response_sem_gene,
        vals_gene,
        vals_time_gene,
        fish_ids_gene,
        coords_gene_projected,
    )


def assemble_gene_lcd_data_across_fish_all_paradigms_with_time(
    gene: str,
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    primary_stim_key: str = "visrap",
    distance_bin: int = 20,
    proj_axis: int = 1,
    ref_fish_for_projection: Optional[str] = None,
    paradigms: Optional[Sequence[str]] = None,
) -> Tuple[
    np.ndarray,                  # coords_gene
    np.ndarray,                  # gc_gene
    np.ndarray,                  # stim_response_avg_gene (for primary stim)
    np.ndarray,                  # stim_response_sem_gene (for primary stim)
    Dict[str, np.ndarray],       # lcd_vals_by_paradigm[paradigm] -> (N, D)
    Dict[str, np.ndarray],       # lcd_vals_time_by_paradigm[paradigm] -> (N, ..., T)
    np.ndarray,                  # fish_ids_gene
    np.ndarray,                  # coords_gene_projected
]:
    """
    Assemble gene-level LCD data across fish for a *primary* stimulus key
    and gather LCD values (and time-resolved LCD, if available) for all paradigms.
    (API unchanged)
    """
    (
        coords_gene,
        gc_gene,
        stim_response_avg_gene,
        stim_response_sem_gene,
        vals_gene_primary,
        vals_time_gene_primary,
        fish_ids_gene,
        coords_gene_projected,
    ) = assemble_gene_lcd_data_across_fish_with_time(
        gene=gene,
        LCD_data=LCD_data,
        fish_data=fish_data,
        fish_inspect=fish_inspect,
        stim_key=primary_stim_key,
        distance_bin=distance_bin,
        proj_axis=proj_axis,
        ref_fish_for_projection=ref_fish_for_projection,
    )

    lcd_vals_by_paradigm: Dict[str, np.ndarray] = {}
    lcd_vals_time_by_paradigm: Dict[str, np.ndarray] = {}

    lcd_vals_by_paradigm[primary_stim_key] = vals_gene_primary
    if vals_time_gene_primary is not None:
        lcd_vals_time_by_paradigm[primary_stim_key] = vals_time_gene_primary

    if paradigms is None:
        paradigms = list(LCD_data.keys())

    for stim_key in paradigms:
        if stim_key == primary_stim_key:
            continue
        if stim_key not in LCD_data:
            continue

        try:
            (
                coords_p,
                _gc_p,
                _stim_avg_p,
                _stim_sem_p,
                vals_p,
                vals_time_p,
                fish_ids_p,
                coords_proj_p,
            ) = assemble_gene_lcd_data_across_fish_with_time(
                gene=gene,
                LCD_data=LCD_data,
                fish_data=fish_data,
                fish_inspect=fish_inspect,
                stim_key=stim_key,
                distance_bin=distance_bin,
                proj_axis=proj_axis,
                ref_fish_for_projection=ref_fish_for_projection,
            )
        except ValueError:
            continue

        if (
            coords_p.shape != coords_gene.shape
            or not np.allclose(coords_p, coords_gene, equal_nan=True)
            or fish_ids_p.shape != fish_ids_gene.shape
            or not np.array_equal(fish_ids_p, fish_ids_gene)
        ):
            print(
                f"Warning: skipping paradigm '{stim_key}' for gene '{gene}' "
                f"because neuron ordering does not match primary_stim_key "
                f"('{primary_stim_key}')."
            )
            continue

        lcd_vals_by_paradigm[stim_key] = vals_p
        if vals_time_p is not None:
            lcd_vals_time_by_paradigm[stim_key] = vals_time_p

    return (
        coords_gene,
        gc_gene,
        stim_response_avg_gene,
        stim_response_sem_gene,
        lcd_vals_by_paradigm,
        lcd_vals_time_by_paradigm,
        fish_ids_gene,
        coords_gene_projected,
    )


# =====================================================================
# Run LCD clustering across all genes (single-scale, no permutations)
# =====================================================================
def run_multiscale_lcd_clustering_across_genes(
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_key: str = "visrap",
    distance_bin: int = 20,
    target_cluster_size: Union[int, Sequence[int]] = 150,
    n_neighbors: int = 25,
    min_cluster_size: int = 25,
    min_frac_per_fish: float = 0.5,
    random_state: int = 0,
    proj_axis: int = 1,
    cluster_mode: str = "ward",
    mode_kwargs: Optional[Dict[str, Any]] = None,
    paradigms: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Run LCD clustering across all genes and aggregate per-cluster data.

    NEW (compatible):
      - clusters additionally store neuron indices that index into fish_data:
            cluster["neuron_inds_gene"]
            cluster["neuron_inds_by_fish"]
        These are aligned to the same NaN filtering used for clustering.
    """
    if paradigms is None:
        paradigms = list(LCD_data.keys())

    first_fish = fish_inspect[0]
    gene_list = list(LCD_data[stim_key][first_fish].keys())

    _ = make_cmap_dict(gene_list)

    cluster_dict: Dict[str, Dict[str, Any]] = {}
    clusters_by_gene: Dict[str, Dict[str, Any]] = {}

    for gene in tqdm(gene_list, desc="Running LCD clustering across genes"):
        try:
            (
                coords_gene,
                gene_counts_gene,
                stim_response_avg_gene,
                stim_response_sem_gene,
                lcd_vals_by_paradigm,
                lcd_vals_time_by_paradigm,
                fish_ids_gene,
                coords_gene_projected,
            ) = assemble_gene_lcd_data_across_fish_all_paradigms_with_time(
                gene=gene,
                LCD_data=LCD_data,
                fish_data=fish_data,
                fish_inspect=fish_inspect,
                primary_stim_key=stim_key,
                distance_bin=distance_bin,
                proj_axis=proj_axis,
                ref_fish_for_projection=first_fish,
                paradigms=paradigms,
            )
        except ValueError as e:
            print(f"Skipping gene {gene}: {e}")
            continue

        # NEW: assemble global neuron indices aligned to the same concatenation order
        try:
            neuron_inds_gene, fish_ids_inds = assemble_gene_neuron_inds_across_fish(
                gene=gene,
                fish_data=fish_data,
                fish_inspect=fish_inspect,
            )
        except ValueError as e:
            print(f"Skipping gene {gene} (index assembly failed): {e}")
            continue

        # sanity: should match fish_ids_gene ordering exactly
        if fish_ids_inds.shape != fish_ids_gene.shape or not np.array_equal(fish_ids_inds, fish_ids_gene):
            raise ValueError(
                f"Fish-id ordering mismatch for gene '{gene}': "
                "assemble_gene_neuron_inds_across_fish did not align with assembled arrays."
            )

        vals_primary = lcd_vals_by_paradigm[stim_key]
        result = find_lcd_clusters_for_gene(
            coords=coords_gene_projected,
            vals=vals_primary[:, 0],
            fish_ids=fish_ids_gene,
            n_neighbors=n_neighbors,
            target_cluster_size=target_cluster_size,
            min_cluster_size=min_cluster_size,
            min_frac_per_fish=min_frac_per_fish,
            min_count_per_fish=5,
            random_state=random_state,
            cluster_mode=cluster_mode,
            mode_kwargs=mode_kwargs,
        )

        clusters = summarize_significant_clusters_for_gene(
            coords=coords_gene,
            gene_counts=gene_counts_gene,
            stim_responses_avg=stim_response_avg_gene,
            stim_responses_sem=stim_response_sem_gene,
            lcd_vals_by_paradigm=lcd_vals_by_paradigm,
            fish_ids=fish_ids_gene,
            result=result,
            lcd_vals_time_by_paradigm=lcd_vals_time_by_paradigm,
            primary_stim_key=stim_key,
            neuron_inds_gene=neuron_inds_gene,  # NEW
        )

        clusters_by_gene[gene] = {}
        for c in clusters:
            cluster_id_int = c["cluster_id"]
            cluster_id_str = f"{gene}_C{cluster_id_int}"

            c["cluster_id_str"] = cluster_id_str
            c["gene"] = gene
            c["stim_key"] = stim_key
            c["distance_bin"] = distance_bin
            c["proj_axis"] = proj_axis

            cluster_dict[cluster_id_str] = c
            clusters_by_gene[gene][cluster_id_str] = c

    return cluster_dict, clusters_by_gene


# =====================================================================
# Helpers: filter clusters by fish support
# =====================================================================
def filter_supported_clusters(
    cluster_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    return {
        cid: c
        for cid, c in cluster_dict.items()
        if c.get("fish_support_ok", False)
    }


def filter_supported_clusters_by_gene(
    clusters_by_gene: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    filtered: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for gene, cdict in clusters_by_gene.items():
        kept = {cid: c for cid, c in cdict.items() if c.get("fish_support_ok", False)}
        if kept:
            filtered[gene] = kept
    return filtered


# =====================================================================
# Helper: split clusters by fish
# (kept compatible; note: your original version references c["vals"]/c["vals_time"]
# but you now store per-paradigm in c["lcd_vals"]/c["lcd_vals_time"].
# Leaving this unchanged would likely break; below is a conservative fix that
# keeps output fields similar by using the cluster's primary stim_key.
# =====================================================================
def split_clusters_by_fish(
    cluster_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Split a pooled cluster_dict into per-fish cluster dictionaries.

    Uses cluster['lcd_vals'][cluster['stim_key']] as "vals" analog.
    Uses cluster['lcd_vals_time'][cluster['stim_key']] as "vals_time" analog (if present).
    """
    fish_clusters: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for cid, c in cluster_dict.items():
        fish_ids = np.asarray(c.get("fish_ids"))
        if fish_ids.size == 0:
            continue

        stim_key = c.get("stim_key", "visrap")

        vals_block = None
        if "lcd_vals" in c and stim_key in c["lcd_vals"]:
            vals_block = np.asarray(c["lcd_vals"][stim_key])

        vals_time_full = None
        if "lcd_vals_time" in c and stim_key in c["lcd_vals_time"]:
            vals_time_full = np.asarray(c["lcd_vals_time"][stim_key])

        unique_fish = np.unique(fish_ids)

        for f in unique_fish:
            mask = fish_ids == f
            if not np.any(mask):
                continue

            coords_f = c["coords"][mask]
            gene_counts_f = c["gene_counts"][mask]
            stim_avg_f = c["stim_responses_avg"][mask]
            stim_sem_f = c["stim_responses_sem"][mask]
            indices_f = c["indices"][mask]

            vals_f = vals_block[mask] if vals_block is not None else None
            vals_time_f = vals_time_full[mask] if vals_time_full is not None else None

            if vals_f is None:
                mean_lcd_f = np.nan
            else:
                vals_f_arr = np.asarray(vals_f)
                mean_lcd_f = float(np.nanmean(vals_f_arr[:, 0])) if vals_f_arr.ndim > 1 else float(np.nanmean(vals_f_arr))

            size_f = int(mask.sum())

            c_f: Dict[str, Any] = {
                "cluster_id_global": c["cluster_id"],
                "cluster_id": cid,
                "fish_id": f,
                "size": size_f,
                "size_global": c["size"],
                "mean_lcd": mean_lcd_f,
                "mean_lcd_global": c["mean_lcd"],
                "fish_support_ok_global": c.get("fish_support_ok", False),
                "coords": coords_f,
                "gene_counts": gene_counts_f,
                "stim_responses_avg": stim_avg_f,
                "stim_responses_sem": stim_sem_f,
                "vals": vals_f,
                "indices": indices_f,
            }

            if vals_time_f is not None:
                c_f["vals_time"] = vals_time_f

            fish_clusters.setdefault(f, {})[cid] = c_f

    return fish_clusters


# =====================================================================
# NEW helper: fetch arrays from fish_data using neuron indices
# =====================================================================
from typing import Dict, Any, Mapping, Optional, Literal, Sequence, Set
import numpy as np


def fetch_cluster_arrays_from_fish_data(
    cluster: Mapping[str, Any],
    fish_data: Mapping[Any, Dict[str, Any]],
    stim_key: Optional[str] = None,
    coords_key: str = "cell_centers_zb",
    gene_counts_key: str = "gene_counts",
    stim_avg_key: str = "avg_stim_responses",
    stim_sem_key: str = "sem_stim_responses",
    stim_avg_cat_key: str = "avg_stim_responses_cat",
    stim_sem_cat_key: str = "sem_stim_responses_cat",
    prefer_cat: bool = False,
    order: Literal["cluster", "fish_sorted"] = "cluster",
    paradigms: Optional[Sequence[str]] = None,          # NEW (optional)
    include_partial_paradigms: bool = True,             # NEW (optional)
) -> Dict[str, Any]:
    """
    Reconstruct the heavy arrays you currently store in cluster objects from fish_data,
    using cluster['neuron_inds_by_fish'] (preferred) or cluster['neuron_inds_gene'].

    NEW:
        Also returns functional data for *all paradigms* under:
            out["functional_data"][paradigm] = {
                "stim_responses_avg": ...,
                "stim_responses_sem": ...,
            }

    Returns dict with:
        coords, gene_counts, stim_responses_avg, stim_responses_sem,
        fish_ids, neuron_inds,
        functional_data (NEW)

    Parameters
    ----------
    stim_key
        Primary stimulus key to also return at top level as stim_responses_avg/sem.
        If None, uses cluster.get("stim_key", "visrap").
    paradigms
        Which paradigms to fetch into out["functional_data"].
        If None, uses union of paradigms across fish in this cluster.
    include_partial_paradigms
        If True: paradigms can be present for only some fish; those fish contribute.
        If False: only include paradigms present for *all* fish in this cluster.
    order
        "cluster"    : preserve fish-block order as it appears in cluster['fish_ids']
                      (stable order by first occurrence). This should match your
                      assembly order in practice.
        "fish_sorted": concatenate fish blocks in sorted fish-id order.
    """
    if stim_key is None:
        stim_key = cluster.get("stim_key", "visrap")

    if "neuron_inds_by_fish" not in cluster and "neuron_inds_gene" not in cluster:
        raise ValueError(
            "Cluster does not contain neuron indices. "
            "Re-run clustering with the updated code so clusters include "
            "'neuron_inds_by_fish'/'neuron_inds_gene'."
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def _stable_unique(arr: np.ndarray) -> np.ndarray:
        """Unique values in order of first appearance."""
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

    def _get_stim_arrays_for_fish(fd: Dict[str, Any], paradigm: str):
        """
        Returns (avg, sem) arrays for one fish and one paradigm.
        Tries CAT keys if prefer_cat=True, otherwise defaults to non-cat keys.
        Falls back when only one variant exists.
        """
        if "stim_response_data" not in fd or paradigm not in fd["stim_response_data"]:
            raise KeyError(f"Fish missing stim_response_data['{paradigm}'].")

        stim_entry = fd["stim_response_data"][paradigm]

        if prefer_cat and (stim_avg_cat_key in stim_entry):
            avg_all = stim_entry[stim_avg_cat_key]
            sem_all = stim_entry.get(stim_sem_cat_key, stim_entry.get(stim_sem_key, None))
        else:
            avg_all = stim_entry.get(stim_avg_key, stim_entry.get(stim_avg_cat_key, None))
            sem_all = stim_entry.get(stim_sem_key, stim_entry.get(stim_sem_cat_key, None))

        if avg_all is None:
            raise KeyError(
                f"Could not find avg stim responses for paradigm '{paradigm}'. "
                f"Tried keys: {stim_avg_key!r}, {stim_avg_cat_key!r}."
            )
        if sem_all is None:
            # SEM is optional in some pipelines; allow None
            sem_all = None

        avg_all = np.asarray(avg_all, dtype=float)
        if sem_all is not None:
            sem_all = np.asarray(sem_all, dtype=float)

        return avg_all, sem_all

    # -----------------------------
    # Determine fish_ids + neuron_inds in desired order
    # -----------------------------
    if order == "cluster" and "neuron_inds_gene" in cluster and "fish_ids" in cluster:
        fish_ids = np.asarray(cluster["fish_ids"], dtype=object)
        neuron_inds = np.asarray(cluster["neuron_inds_gene"], dtype=int)
        if fish_ids.shape[0] != neuron_inds.shape[0]:
            # fall back to per-fish blocks below
            fish_ids = None
            neuron_inds = None
    else:
        fish_ids = None
        neuron_inds = None

    if fish_ids is None or neuron_inds is None:
        # Build (fish_ids, neuron_inds) from per-fish dict (or from gene+fish_ids if needed)
        neuron_inds_by_fish = cluster.get("neuron_inds_by_fish", None)
        if neuron_inds_by_fish is None:
            fish_ids_tmp = np.asarray(cluster.get("fish_ids"), dtype=object)
            neuron_inds_tmp = np.asarray(cluster.get("neuron_inds_gene"), dtype=int)
            neuron_inds_by_fish = {}
            for f in np.unique(fish_ids_tmp):
                m = fish_ids_tmp == f
                neuron_inds_by_fish[str(f)] = neuron_inds_tmp[m]

        # Choose fish order
        if order == "fish_sorted":
            fish_order = sorted(neuron_inds_by_fish.keys())
        else:
            # stable order by first appearance in cluster['fish_ids'] if available,
            # else fall back to sorted
            if "fish_ids" in cluster:
                fish_ids_tmp = np.asarray(cluster["fish_ids"], dtype=object)
                fish_order = [str(f) for f in _stable_unique(fish_ids_tmp)]
            else:
                fish_order = sorted(neuron_inds_by_fish.keys())

        fish_ids_list = []
        neuron_inds_list = []
        for f_str in fish_order:
            inds_f = np.asarray(neuron_inds_by_fish.get(f_str, []), dtype=int)
            if inds_f.size == 0:
                continue
            fish_ids_list.append(np.array([f_str] * inds_f.size, dtype=object))
            neuron_inds_list.append(inds_f)

        fish_ids = np.concatenate(fish_ids_list, axis=0) if fish_ids_list else np.array([], dtype=object)
        neuron_inds = np.concatenate(neuron_inds_list, axis=0) if neuron_inds_list else np.array([], dtype=int)

    # -----------------------------
    # Fetch coords + gene counts (always)
    # -----------------------------
    coords_list = []
    gc_list = []

    # use stable fish order (so we match fish_ids/neuron_inds block order)
    fish_order_vals = _stable_unique(fish_ids) if order == "cluster" else np.unique(fish_ids)
    if order == "fish_sorted":
        fish_order_vals = np.array(sorted([str(f) for f in fish_order_vals]), dtype=object)

    for f in fish_order_vals:
        m = fish_ids == f
        if not np.any(m):
            continue
        inds_f = neuron_inds[m]
        fd = fish_data[f]

        coords_list.append(np.asarray(fd["cell_centers_data"][coords_key], dtype=float)[inds_f])
        gc_list.append(np.asarray(fd["gene_data"][gene_counts_key], dtype=float)[inds_f])

    coords = np.concatenate(coords_list, axis=0) if coords_list else None
    gene_counts = np.concatenate(gc_list, axis=0) if gc_list else None

    # -----------------------------
    # Fetch primary stim_key at top level (backward compatible)
    # -----------------------------
    stim_avg_list = []
    stim_sem_list = []

    for f in fish_order_vals:
        m = fish_ids == f
        if not np.any(m):
            continue
        inds_f = neuron_inds[m]
        fd = fish_data[f]

        avg_all, sem_all = _get_stim_arrays_for_fish(fd, stim_key)
        stim_avg_list.append(avg_all[inds_f])
        if sem_all is None:
            stim_sem_list.append(None)
        else:
            stim_sem_list.append(sem_all[inds_f])

    stim_responses_avg = np.concatenate(stim_avg_list, axis=0) if stim_avg_list else None
    if stim_sem_list and all(x is not None for x in stim_sem_list):
        stim_responses_sem = np.concatenate(stim_sem_list, axis=0)
    else:
        stim_responses_sem = None

    # -----------------------------
    # NEW: Fetch functional data for all paradigms
    # -----------------------------
    # Determine paradigms if not provided
    if paradigms is None:
        paradigms_set: Set[str] = set()
        for f in fish_order_vals:
            fd = fish_data[f]
            if "stim_response_data" in fd:
                paradigms_set.update(list(fd["stim_response_data"].keys()))
        paradigms = sorted(paradigms_set)

    functional_data: Dict[str, Dict[str, Any]] = {}

    for p in paradigms:
        p_avg_list = []
        p_sem_list = []
        fish_used = 0
        fish_missing = 0

        for f in fish_order_vals:
            m = fish_ids == f
            if not np.any(m):
                continue
            inds_f = neuron_inds[m]
            fd = fish_data[f]

            try:
                avg_all, sem_all = _get_stim_arrays_for_fish(fd, p)
            except KeyError:
                fish_missing += 1
                continue

            fish_used += 1
            p_avg_list.append(avg_all[inds_f])
            if sem_all is None:
                p_sem_list.append(None)
            else:
                p_sem_list.append(sem_all[inds_f])

        if fish_used == 0:
            continue

        if (not include_partial_paradigms) and (fish_missing > 0):
            # require paradigm present for all fish in this cluster
            continue

        p_avg = np.concatenate(p_avg_list, axis=0) if p_avg_list else None
        if p_sem_list and all(x is not None for x in p_sem_list):
            p_sem = np.concatenate(p_sem_list, axis=0)
        else:
            p_sem = None

        functional_data[p] = {
            "stim_responses_avg": p_avg,
            "stim_responses_sem": p_sem,
        }

    return {
        "coords": coords,
        "gene_counts": gene_counts,
        "stim_responses_avg": stim_responses_avg,
        "stim_responses_sem": stim_responses_sem,
        "fish_ids": fish_ids,
        "neuron_inds": neuron_inds,
        "functional_data": functional_data,  # NEW
    }



# =====================================================================
# compute_cluster_neighbor_traces (kept compatible)
# =====================================================================
def compute_cluster_neighbor_traces(
    cluster_dict: Mapping[str, Dict[str, Any]],
    fish_data: Mapping[str, Dict[str, Any]],
    neighbor_radius: Optional[float] = None,
    stim_key_default: str = "visrap",
    cluster_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Precompute neighbor-average stimulus traces for each cluster.
    """
    neighbor_traces: Dict[str, Dict[str, np.ndarray]] = {}
    neighbor_cache: Dict[Any, Optional[Dict[str, Any]]] = {}

    if cluster_ids is None:
        cluster_ids_iter = list(cluster_dict.keys())
    else:
        cluster_ids_iter = list(cluster_ids)

    for cid in cluster_ids_iter:
        if cid not in cluster_dict:
            continue

        c = cluster_dict[cid]

        fish_ids = np.asarray(c.get("fish_ids", []))
        coords_cluster = np.asarray(c.get("coords", []), dtype=float)

        if fish_ids.size == 0 or coords_cluster.size == 0:
            neighbor_traces[cid] = {"neighbor_mean_trace": None, "neighbor_sem_trace": None}
            continue

        gene = c.get("gene", None)
        if gene is None:
            gene = str(cid).split("_")[0]

        stim_key = c.get("stim_key", stim_key_default)
        dist_bin = c.get("distance_bin", None)

        radius = neighbor_radius
        if radius is None:
            radius = float(dist_bin) if dist_bin is not None else 20.0

        neighbor_traces_list = []

        for f in np.unique(fish_ids):
            if f not in fish_data:
                continue

            if f not in neighbor_cache:
                fish_entry = fish_data[f]

                coords_all_full = np.asarray(
                    fish_entry["cell_centers_data"]["cell_centers_zb"], dtype=float
                )

                stim_entry = fish_entry["stim_response_data"][stim_key]
                if "avg_stim_responses_cat" in stim_entry:
                    stim_resp_all_full = np.asarray(stim_entry["avg_stim_responses_cat"], dtype=float)
                else:
                    stim_resp_all_full = np.asarray(stim_entry["avg_stim_responses"], dtype=float)

                gene_names_f = fish_entry["gene_data"]["gene_names"]
                gene_counts_binary_full = np.asarray(
                    fish_entry["gene_data"]["gene_counts_binary"], dtype=bool
                )

                valid_mask = ~np.isnan(coords_all_full).any(axis=1)
                coords_all_f = coords_all_full[valid_mask]
                if coords_all_f.shape[0] == 0:
                    neighbor_cache[f] = None
                else:
                    stim_resp_all_f = stim_resp_all_full[valid_mask]
                    gene_counts_binary_valid = gene_counts_binary_full[valid_mask]
                    tree = KDTree(coords_all_f)
                    neighbor_cache[f] = {
                        "coords_all": coords_all_f,
                        "stim_resp_all": stim_resp_all_f,
                        "gene_names": gene_names_f,
                        "gene_counts_binary": gene_counts_binary_valid,
                        "expr_bins": {},
                        "tree": tree,
                    }

            ctx = neighbor_cache[f]
            if ctx is None:
                continue

            coords_all_f = ctx["coords_all"]
            stim_resp_all_f = ctx["stim_resp_all"]
            gene_names_f = ctx["gene_names"]
            gene_counts_binary_valid = ctx["gene_counts_binary"]
            tree = ctx["tree"]

            gene_key = str(gene)
            if gene_key not in ctx["expr_bins"]:
                gene_inds = np.where(gene_names_f == gene)[0]
                if gene_inds.size == 0:
                    ctx["expr_bins"][gene_key] = None
                else:
                    gi = int(gene_inds[0])
                    expr_bin = gene_counts_binary_valid[:, gi].astype(bool)
                    ctx["expr_bins"][gene_key] = expr_bin

            expr_bin = ctx["expr_bins"][gene_key]
            if expr_bin is None:
                continue

            mask_f = (fish_ids == f)
            coords_cluster_f = coords_cluster[mask_f]
            if coords_cluster_f.size == 0:
                continue

            mask_valid_cf = ~np.isnan(coords_cluster_f).any(axis=1)
            coords_cluster_f = coords_cluster_f[mask_valid_cf]
            if coords_cluster_f.shape[0] == 0:
                continue

            ind_arrays = tree.query_radius(coords_cluster_f, r=radius)

            neighbors_mask = np.zeros(coords_all_f.shape[0], dtype=bool)
            for inds in ind_arrays:
                if inds.size == 0:
                    continue
                inds_neg = inds[~expr_bin[inds]]
                if inds_neg.size == 0:
                    continue
                neighbors_mask[inds_neg] = True

            neighbor_idx = np.where(neighbors_mask)[0]
            if neighbor_idx.size == 0:
                continue

            neighbor_block = np.asarray(stim_resp_all_f[neighbor_idx], dtype=float)
            if neighbor_block.ndim == 1:
                neighbor_block = neighbor_block[None, :]

            neighbor_block_flat = neighbor_block.reshape(neighbor_block.shape[0], -1)
            neighbor_traces_list.append(neighbor_block_flat)

        if neighbor_traces_list:
            neighbor_mat = np.concatenate(neighbor_traces_list, axis=0)
            n_non_nan = np.sum(~np.isnan(neighbor_mat), axis=0)
            n_non_nan = np.maximum(1, n_non_nan)
            neighbor_mean = np.nanmean(neighbor_mat, axis=0)
            neighbor_std = np.nanstd(neighbor_mat, axis=0)
            neighbor_sem = neighbor_std / np.sqrt(n_non_nan)

            neighbor_mean_trace = neighbor_mean.astype(float)
            neighbor_sem_trace = neighbor_sem.astype(float)
        else:
            neighbor_mean_trace = None
            neighbor_sem_trace = None

        neighbor_traces[cid] = {
            "neighbor_mean_trace": neighbor_mean_trace,
            "neighbor_sem_trace": neighbor_sem_trace,
        }

    return neighbor_traces


# =====================================================================
# compute_cluster_stim_lcd_stats (unchanged)
# =====================================================================
def compute_cluster_stim_lcd_stats(
    cluster_dict: Mapping[str, Dict[str, Any]],
    fish_data: Mapping[Any, Dict[str, Any]],
    stim_key: str = "visrap",
    pre_duration: int = 0,
    stim_duration: int = 7,
    post_duration: int = 16,
    component: Literal["lcd", "same", "other"] = "lcd",
) -> Dict[str, Dict[str, Any]]:
    """
    For each cluster in cluster_dict, compute per-stimulus statistics of
    time-resolved LCD contributions (or related components).
    """
    try:
        from WARP.stimulus_response_utils import (
            calc_stim_responses,
            calc_avg_stim_responses,
        )
    except ImportError as e:
        raise ImportError(
            "compute_cluster_stim_lcd_stats requires "
            "WARP.stimulus_response_utils (calc_stim_responses, "
            "calc_avg_stim_responses)."
        ) from e

    comp_idx_map = {"lcd": 0, "same": 1, "other": 2}
    if component not in comp_idx_map:
        raise ValueError(
            f"Unknown component {component!r}, must be one of {list(comp_idx_map.keys())}."
        )
    comp_idx = comp_idx_map[component]

    epoch_names = np.array(["pre", "stim", "post"], dtype=object)
    cluster_stim_stats: Dict[str, Dict[str, Any]] = {}

    for cid, c in cluster_dict.items():
        stim_for_cluster = c.get("stim_key", stim_key)
        lcd_time_all = c.get("lcd_vals_time", None)
        if lcd_time_all is None or stim_for_cluster not in lcd_time_all:
            continue

        lcd_time_full = np.asarray(lcd_time_all[stim_for_cluster])
        if lcd_time_full.ndim != 3 or lcd_time_full.shape[1] != 3:
            continue

        n_cells, _, T_full = lcd_time_full.shape

        fish_ids = np.asarray(c.get("fish_ids", []))
        if fish_ids.shape[0] != n_cells:
            raise ValueError(
                f"Cluster '{cid}' has lcd_vals_time[{stim_for_cluster!r}].shape[0]={n_cells} "
                f"but fish_ids.shape[0]={fish_ids.shape[0]}."
            )

        epoch_means_all_cells = []
        time_responses_all_cells = []

        unique_fish = np.unique(fish_ids)
        first_n_stim = None
        first_block_len = None

        for f in unique_fish:
            mask_f = (fish_ids == f)
            if not np.any(mask_f):
                continue

            data_f = lcd_time_full[mask_f, comp_idx, :]

            try:
                stim_vec = np.asarray(
                    fish_data[f]["ephys_data"][stim_for_cluster]["stimulus"], dtype=int
                )
            except Exception as e:
                print(
                    f"[compute_cluster_stim_lcd_stats] Could not get stimulus "
                    f"for fish '{f}' in cluster '{cid}': {e}"
                )
                continue

            T_stim = stim_vec.shape[0]
            if T_stim != T_full:
                T_eff = min(T_full, T_stim)
                if T_eff <= 0:
                    continue
                data_f = data_f[:, :T_eff]
                stim_vec = stim_vec[:T_eff]

            stim_responses = calc_stim_responses(
                data_f,
                stim_vec,
                pre_duration=pre_duration,
                stim_duration=stim_duration,
                post_duration=post_duration,
            )

            avg_responses, _ = calc_avg_stim_responses(stim_responses)
            n_cells_fish, n_stim, block_len = avg_responses.shape

            if first_n_stim is None:
                first_n_stim = n_stim
                first_block_len = block_len
            else:
                if n_stim != first_n_stim or block_len != first_block_len:
                    raise ValueError(
                        f"Inconsistent stimulus response shape for cluster '{cid}': "
                        f"previous (n_stim, block_len)=({first_n_stim}, {first_block_len}), "
                        f"now ({n_stim}, {block_len}) for fish {f!r}."
                    )

            idx_pre = np.arange(0, min(pre_duration, block_len), dtype=int)
            idx_stim = np.arange(pre_duration, min(pre_duration + stim_duration, block_len), dtype=int)
            idx_post = np.arange(
                pre_duration + stim_duration,
                min(pre_duration + stim_duration + post_duration, block_len),
                dtype=int,
            )
            epoch_indices = [idx_pre, idx_stim, idx_post]

            epoch_means_f = np.full((n_cells_fish, n_stim, 3), np.nan, dtype=float)
            for e_idx, idxs in enumerate(epoch_indices):
                if idxs.size == 0:
                    continue
                epoch_means_f[:, :, e_idx] = np.nanmedian(avg_responses[:, :, idxs], axis=2)

            epoch_means_all_cells.append(epoch_means_f)
            time_responses_all_cells.append(avg_responses)

        if not epoch_means_all_cells:
            continue

        epoch_means_all_cells = np.concatenate(epoch_means_all_cells, axis=0)
        time_responses_all_cells = np.concatenate(time_responses_all_cells, axis=0)

        mean = np.nanmean(epoch_means_all_cells, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.nanstd(epoch_means_all_cells, axis=0)
            n_eff = np.sum(np.isfinite(epoch_means_all_cells), axis=0)
            sem = std / np.sqrt(np.maximum(n_eff, 1))
            sem[n_eff <= 1] = np.nan

        time_mean = np.nanmean(time_responses_all_cells, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            time_std = np.nanstd(time_responses_all_cells, axis=0)
            n_eff_t = np.sum(np.isfinite(time_responses_all_cells), axis=0)
            time_sem = time_std / np.sqrt(np.maximum(n_eff_t, 1))
            time_sem[n_eff_t <= 1] = np.nan

        stim_ids = np.arange(first_n_stim, dtype=int) + 1

        cluster_stim_stats[cid] = {
            "stim_ids": stim_ids,
            "mean": mean,
            "sem": sem,
            "time_mean": time_mean,
            "time_sem": time_sem,
            "epoch_names": epoch_names.copy(),
            "component": component,
            "pre_duration": int(pre_duration),
            "stim_duration": int(stim_duration),
            "post_duration": int(post_duration),
        }

    return cluster_stim_stats
