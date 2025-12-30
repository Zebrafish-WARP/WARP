import numpy as np
from typing import Dict, Iterable, Tuple, Union
from statsmodels.stats.multitest import fdrcorrection


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def empirical_p_value(statistic, null_values, alternative='two-sided', center=None):
    """
    Compute an empirical (permutation-style) p-value.

    Parameters
    ----------
    statistic : float
        Observed test statistic from real data (scalar).
    null_values : array-like
        Null distribution values (e.g., from permutations/shuffles).
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
    center : {None, 'empirical', float}, optional
        Centering for both statistic and null:
        - None: no centering
        - 'empirical': center by null mean
        - float: center by given value

    Returns
    -------
    p_value : float
    """
    null_values = np.asarray(null_values).ravel()
    n_null = len(null_values)
    if n_null == 0:
        return np.nan

    # Centering
    if center == 'empirical':
        c = float(null_values.mean())
    elif center is None:
        c = 0.0
    else:
        c = float(center)

    if c != 0.0:
        statistic = statistic - c
        null_values = null_values - c

    if alternative == 'greater':
        return (np.sum(null_values >= statistic) + 1) / (n_null + 1)
    elif alternative == 'less':
        return (np.sum(null_values <= statistic) + 1) / (n_null + 1)
    elif alternative == 'two-sided':
        return (np.sum(np.abs(null_values) >= abs(statistic)) + 1) / (n_null + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def _extract_diff_stat(x) -> float:
    """
    Extract LCD difference component from a scalar or 1D vector.

    - If scalar: return as float
    - If 1D: return x[0] as float  (assumed LCD difference)
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0])
    raise ValueError(f"Expected scalar or 1D array; got shape {arr.shape}")


def _extract_diff_null(null_values) -> np.ndarray:
    """
    Extract LCD difference component from null array:

    - (N,)      -> returned as float array
    - (N, 3)    -> return column 0 as float array
    """
    arr = np.asarray(null_values)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        return arr[:, 0].astype(float)
    raise ValueError(f"Expected null array of shape (N,) or (N,3); got {arr.shape}")


import numpy as np
from typing import Dict, Iterable, Tuple, Any
from statsmodels.stats.multitest import fdrcorrection

# assumes these exist in your codebase:
# - empirical_p_value(statistic, null_values, alternative='two-sided', center=None)
# - _extract_diff_null(null_raw)  -> 1D array of null diffs (feature 0)
# - _extract_diff_stat(obs_val)   -> scalar or small vector of observed diffs (feature 0)


# ---------------------------------------------------------------------
# Small utility: treat size-1 arrays/lists as true scalars (0d arrays)
# ---------------------------------------------------------------------
def _coerce_size1_to_scalar(v):
    """
    If v is array-like with exactly one element (e.g. [x], np.array([x])),
    return a 0d numpy scalar array; otherwise return np.asarray(v) unchanged.
    """
    a = np.asarray(v)
    if a.size == 1:
        # reshape to 0d; keeps dtype; float(a) is too aggressive sometimes
        return np.asarray(a.reshape(()))
    return a


def compute_pvals(
    LCD_data: dict,
    fish_inspect,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    paradigms=None,
    mask_types=("distance", "region"),
) -> dict:
    """
    Compute empirical and FDR-corrected p-values for all scalar / small-vector
    statistics found in observed_stats, using the LCD *difference* component
    (feature index 0) for the permutation null.

    IMPORTANT: FDR correction is done *across all genes* within a fish,
    separately for each (paradigm, fish, mask_type, stat_name) combination.

    Additionally, creates a pseudo-gene key 'all_genes' PER FISH that:
      1) keeps observed_stats fields (including 'vals') by pooling across genes
         WITHIN each (mask_type, mask, stat_name)
      2) computes a pooled p-value per (mask_type, mask, stat_name) by pooling
         observed diffs and null diffs across genes (within mask), BUT NOT FDR-corrected.
    """
    if paradigms is None:
        paradigms = LCD_data.keys()

    # ------------------------------------------------------------------
    # FIRST PASS: compute raw p-values and cache them (for FDR)
    # key: (paradigm, fish_n, mask_type, stat_name)
    # val: list of (gene, mask, p_raw)
    # ------------------------------------------------------------------
    raw_pvals: Dict[Tuple[Any, Any, str, str], list] = {}

    # ------------------------------------------------------------------
    # POOLED observed cache for 'all_genes' (includes large fields like 'vals'):
    # key: (paradigm, fish_n, mask_type, mask, stat_name)
    # val: list of np.arrays (raw observed stat values)
    # ------------------------------------------------------------------
    pooled_obs_cache: Dict[Tuple[Any, Any, str, Any, str], list] = {}

    # ------------------------------------------------------------------
    # POOLED diff cache for 'all_genes' p-values (ONLY small/scalar stats):
    # key: (paradigm, fish_n, mask_type, mask, stat_name)
    # val: {"obs": [1D arrays...], "null": [1D arrays...]}  (diff-extracted)
    # ------------------------------------------------------------------
    pooled_diff_cache: Dict[Tuple[Any, Any, str, Any, str], Dict[str, list]] = {}

    for paradigm in paradigms:
        paradigm_data = LCD_data.get(paradigm, {})
        if not paradigm_data:
            continue

        for fish_n in fish_inspect:
            fish_data = paradigm_data.get(fish_n)
            if not fish_data:
                continue

            for gene, gene_data in fish_data.items():
                if gene == "all_genes":
                    continue

                observed_all = gene_data.get("observed_stats")
                perm_all = gene_data.get("perm_stats")
                if observed_all is None or perm_all is None:
                    continue

                for mask_type in mask_types:
                    observed_block = observed_all.get(mask_type)
                    perm_block = perm_all.get(mask_type)
                    if not observed_block or not perm_block:
                        continue

                    masks = list(observed_block.keys())
                    if not masks:
                        continue

                    stat_names = sorted({
                        key
                        for mask in masks
                        for key in observed_block[mask].keys()
                    })

                    for mask in masks:
                        obs_stats = observed_block[mask]
                        perm_stats = perm_block.get(mask)
                        if perm_stats is None:
                            continue

                        for s in stat_names:
                            if s not in obs_stats:
                                continue

                            obs_val = obs_stats[s]
                            if obs_val is None:
                                continue

                            arr = _coerce_size1_to_scalar(obs_val)

                            # --------------------------------------------------
                            # 0) Always collect observed stat for all_genes,
                            #    including large arrays like 'vals'
                            # --------------------------------------------------
                            key_obs_pool = (paradigm, fish_n, mask_type, mask, s)
                            pooled_obs_cache.setdefault(key_obs_pool, []).append(arr)

                            # Identify "large" stats (per-neuron / per-pair arrays)
                            is_large = (arr.ndim > 1) or (arr.ndim == 1 and arr.shape[0] > 3)
                            if is_large:
                                continue  # keep in observed_stats, but no p-value

                            # --------------------------------------------------
                            # 1) Determine null distribution for this stat
                            # --------------------------------------------------
                            null_raw = None
                            key_perm = f"perm_{s}"
                            if key_perm in perm_stats:
                                null_raw = perm_stats[key_perm]
                            elif "perm_mean" in perm_stats:
                                null_raw = perm_stats["perm_mean"]
                            else:
                                continue

                            null_diff = _extract_diff_null(null_raw)
                            obs_diff = _extract_diff_stat(obs_val)

                            # --------------------------------------------------
                            # 2) per-gene/mask p-value (used for FDR)
                            # --------------------------------------------------
                            p_raw = empirical_p_value(
                                statistic=obs_diff,
                                null_values=null_diff,
                                alternative=alternative,
                            )
                            if not np.isnan(p_raw):
                                key = (paradigm, fish_n, mask_type, s)
                                raw_pvals.setdefault(key, []).append((gene, mask, float(p_raw)))

                            # --------------------------------------------------
                            # 3) pooled across genes p-value cache (NO FDR)
                            # --------------------------------------------------
                            key_diff_pool = (paradigm, fish_n, mask_type, mask, s)
                            d = pooled_diff_cache.setdefault(key_diff_pool, {"obs": [], "null": []})

                            obs_arr = np.asarray(obs_diff).ravel()
                            null_arr = np.asarray(null_diff).ravel()
                            if obs_arr.size and null_arr.size:
                                d["obs"].append(obs_arr)
                                d["null"].append(null_arr)

    # ------------------------------------------------------------------
    # SECOND PASS: apply FDR per (paradigm, fish, mask_type, stat_name)
    # across all genes & masks for that fish/stat combo
    # ------------------------------------------------------------------
    for (paradigm, fish_n, mask_type, stat_name), entries in raw_pvals.items():
        if not entries:
            continue

        p_raw_arr = np.array([p for (_, _, p) in entries], dtype=float)
        signif_corr, p_corr_arr = fdrcorrection(p_raw_arr, alpha=alpha)

        for (gene, mask, p_raw), p_corr, sig_corr in zip(entries, p_corr_arr, signif_corr):
            gene_data = LCD_data[paradigm][fish_n][gene]
            gene_pvals = gene_data.setdefault("p_vals", {})
            mask_type_pvals = gene_pvals.setdefault(mask_type, {})
            mask_stats = mask_type_pvals.setdefault(mask, {})

            mask_stats[stat_name] = {
                "standard": float(p_raw),
                "corrected": float(p_corr),
                "significant_standard": bool(p_raw < alpha),
                "significant_corr": bool(sig_corr),
            }

    # ------------------------------------------------------------------
    # THIRD PASS: write pooled observed_stats for 'all_genes' (incl. 'vals')
    # ------------------------------------------------------------------
    for (paradigm, fish_n, mask_type, mask, stat_name), arrs in pooled_obs_cache.items():
        if not arrs:
            continue

        # coerce any size-1 arrays to scalars BEFORE combining
        arrs = [_coerce_size1_to_scalar(a) for a in arrs]

        fish_block = LCD_data.setdefault(paradigm, {}).setdefault(fish_n, {})
        all_gene_block = fish_block.setdefault("pooled", {})

        obs_stats_block = all_gene_block.setdefault("observed_stats", {})
        out = obs_stats_block.setdefault(mask_type, {}).setdefault(mask, {})

        # Combine across genes:
        # - all scalars (0d) -> scalar mean
        # - mix of scalars + small fixed vectors (len<=3, same len) -> elementwise mean
        # - all 2D with same second dim -> vstack
        # - else -> concatenate flattened
        if all(a.ndim == 0 for a in arrs):
            out[stat_name] = float(np.nanmean([float(a) for a in arrs]))

        elif all(a.ndim in (0, 1) for a in arrs):
            vecs = [a for a in arrs if a.ndim == 1]
            if len(vecs) == 0:
                out[stat_name] = float(np.nanmean([float(a) for a in arrs]))
            else:
                lengths = [v.shape[0] for v in vecs]
                max_len, min_len = max(lengths), min(lengths)
                if max_len <= 3 and max_len == min_len:
                    stacked = np.stack(
                        [a if a.ndim == 1 else np.full((max_len,), float(a)) for a in arrs],
                        axis=0,
                    )
                    merged = np.nanmean(stacked, axis=0)
                    # if it’s length-1, store a scalar, not a list
                    out[stat_name] = float(merged[0]) if merged.size == 1 else merged
                else:
                    out[stat_name] = np.concatenate([np.asarray(a).ravel() for a in arrs])

        elif all(a.ndim == 2 for a in arrs) and len({a.shape[1] for a in arrs}) == 1:
            out[stat_name] = np.vstack(arrs)

        else:
            out[stat_name] = np.concatenate([np.asarray(a).ravel() for a in arrs])

    # ------------------------------------------------------------------
    # FOURTH PASS: store pooled 'all_genes' p-values (NO FDR)
    # ------------------------------------------------------------------
    for (paradigm, fish_n, mask_type, mask, stat_name), pools in pooled_diff_cache.items():
        obs_list = pools.get("obs", [])
        null_list = pools.get("null", [])
        if not obs_list or not null_list:
            continue

        obs_pool = np.concatenate(obs_list).astype(float, copy=False)
        null_pool = np.concatenate(null_list).astype(float, copy=False)
        if obs_pool.size == 0 or null_pool.size == 0:
            continue

        pooled_stat = float(np.nanmean(obs_pool))
        p_pool = empirical_p_value(
            statistic=pooled_stat,
            null_values=null_pool,
            alternative=alternative,
        )
        if np.isnan(p_pool):
            continue

        fish_block = LCD_data.setdefault(paradigm, {}).setdefault(fish_n, {})
        all_gene_block = fish_block.setdefault("pooled", {})

        pvals_block = all_gene_block.setdefault("p_vals", {})
        out = pvals_block.setdefault(mask_type, {}).setdefault(mask, {})

        out[stat_name] = {
            "standard": float(p_pool),
            "corrected": float(p_pool),  # explicitly not FDR-corrected
            "significant_standard": bool(p_pool < alpha),
            "significant_corr": bool(p_pool < alpha),
        }

    return LCD_data


# ---------------------------------------------------------------------
# Across-fish aggregation
# ---------------------------------------------------------------------
def _aggregate_observed(values):
    """
    Average across fish:
    - If all scalars (including size-1 arrays): scalar mean
    - Otherwise: elementwise mean (preserving shape)
    """
    arrs = [_coerce_size1_to_scalar(v) for v in values]
    arrs = [np.asarray(v) for v in arrs]
    if not arrs:
        return np.nan

    if all(a.ndim == 0 for a in arrs):
        return float(np.mean([float(a) for a in arrs]))

    stacked = np.stack(arrs, axis=0)
    merged = np.nanmean(stacked, axis=0)
    if np.asarray(merged).size == 1:
        return float(np.asarray(merged).reshape(()))
    return merged


def _classify_stat(values_list):
    """
    Decide whether a statistic should be 'average' or 'concat' across fish.

    Heuristic (no hard-coded names):
    - All scalars (including size-1 arrays) -> 'average'
    - Scalars + small 1D vectors (len <= 3, same length) -> 'average'
    - 1D vectors with length > 3 or varying lengths -> 'concat'
    - ndims > 1 (e.g. per-neuron 2D arrays) -> 'concat'
    """
    arrs = [_coerce_size1_to_scalar(v) for v in values_list if v is not None]
    arrs = [np.asarray(v) for v in arrs]
    if not arrs:
        return "average"

    if all(a.ndim == 0 for a in arrs):
        return "average"

    if all(a.ndim in (0, 1) for a in arrs):
        lengths = [a.shape[0] for a in arrs if a.ndim == 1]
        if not lengths:
            return "average"
        max_len = max(lengths)
        min_len = min(lengths)
        if max_len <= 3 and max_len == min_len:
            return "average"
        return "concat"

    return "concat"


def _combine_pvals(p_std_list, p_corr_list, signif_std_list, signif_corr_list, alpha):
    """
    Conservative combination across fish:
    - merged p = max p across fish
    - significance only if all fish significant and merged p < alpha
    """
    if len(p_std_list):
        p_std = float(np.nanmax(p_std_list))
        s_std = all(signif_std_list) and (p_std < alpha)
    else:
        p_std, s_std = np.nan, False

    if len(p_corr_list):
        p_corr = float(np.nanmax(p_corr_list))
        s_corr = all(signif_corr_list) and (p_corr < alpha)
    else:
        p_corr, s_corr = np.nan, False

    return {
        "standard": p_std,
        "corrected": p_corr,
        "significant_standard": s_std,
        "significant_corr": s_corr,
    }


def merge_LCD_dicts(
    LCD_data: Dict,
    fish_inspect: Iterable,
    alpha: float = 0.05,
    paradigms: Iterable = None,
    mask_types: Tuple[str, ...] = ("distance", "region"),
) -> Dict:
    """
    Merge LCD_data across fish.

    If compute_pvals() added pseudo-gene 'all_genes', it will be merged like any other gene:
    - observed_stats: averaged/concatenated via the same heuristics
    - p_vals: combined conservatively with _combine_pvals
    """
    if paradigms is None:
        paradigms = LCD_data.keys()

    merged: Dict = {}

    for paradigm in paradigms:
        p_data = LCD_data.get(paradigm, {})
        if not p_data:
            continue

        merged[paradigm] = {}

        genes = set().union(*(p_data.get(f, {}).keys() for f in fish_inspect))

        for gene in genes:
            merged_gene = {"observed_stats": {}, "p_vals": {}}
            merged[paradigm][gene] = merged_gene

            for mask_type in mask_types:
                masks = set()
                for f in fish_inspect:
                    obs = (
                        p_data.get(f, {})
                        .get(gene, {})
                        .get("observed_stats", {})
                        .get(mask_type)
                    )
                    if obs:
                        masks.update(obs.keys())

                if not masks:
                    continue

                obs_out = {}
                pvals_out = {}

                stat_names = set()
                for f in fish_inspect:
                    for mask in masks:
                        stats = (
                            p_data.get(f, {})
                            .get(gene, {})
                            .get("observed_stats", {})
                            .get(mask_type, {})
                            .get(mask, {})
                        )
                        if stats:
                            stat_names.update(stats.keys())
                stat_names = sorted(stat_names)

                for mask in masks:
                    merged_obs = {}
                    merged_p = {}

                    per_stat_vals = {s: [] for s in stat_names}
                    for f in fish_inspect:
                        stats = (
                            p_data.get(f, {})
                            .get(gene, {})
                            .get("observed_stats", {})
                            .get(mask_type, {})
                            .get(mask, {})
                        )
                        if not stats:
                            continue
                        for s in stat_names:
                            if s in stats:
                                per_stat_vals[s].append(stats[s])

                    for s, vals in per_stat_vals.items():
                        if not vals:
                            continue
                        mode = _classify_stat(vals)
                        arrs = [_coerce_size1_to_scalar(v) for v in vals]
                        arrs = [np.asarray(v) for v in arrs]

                        if mode == "average":
                            merged_obs[s] = _aggregate_observed(arrs)
                        else:
                            if all(a.ndim == 2 for a in arrs) and len({a.shape[1] for a in arrs}) == 1:
                                merged_obs[s] = np.vstack(arrs)
                            else:
                                merged_obs[s] = np.concatenate([a.ravel() for a in arrs])

                    for s in stat_names:
                        p_std_list, p_corr_list = [], []
                        signif_std_list, signif_corr_list = [], []

                        for f in fish_inspect:
                            p_block = (
                                p_data.get(f, {})
                                .get(gene, {})
                                .get("p_vals", {})
                                .get(mask_type, {})
                                .get(mask, {})
                                .get(s)
                            )
                            if not p_block:
                                continue

                            p_std_list.append(p_block.get("standard", np.nan))
                            p_corr_list.append(p_block.get("corrected", np.nan))
                            signif_std_list.append(bool(p_block.get("significant_standard", False)))
                            signif_corr_list.append(bool(p_block.get("significant_corr", False)))

                        if p_std_list:
                            merged_p[s] = _combine_pvals(
                                np.asarray(p_std_list),
                                np.asarray(p_corr_list),
                                signif_std_list,
                                signif_corr_list,
                                alpha,
                            )

                    if merged_obs:
                        obs_out[mask] = merged_obs
                    if merged_p:
                        pvals_out[mask] = merged_p

                if obs_out:
                    merged_gene["observed_stats"][mask_type] = obs_out
                if pvals_out:
                    merged_gene["p_vals"][mask_type] = pvals_out

    return merged



import numpy as np
from typing import Dict, Any, Sequence, Tuple, Optional


# ---------------------------------------------------------------------
# Existing utility from your codebase (assumed available)
# ---------------------------------------------------------------------
def empirical_p_value(statistic, null_values, alternative='two-sided', center=None):
    """
    Compute an empirical (permutation-style) p-value.

    Parameters
    ----------
    statistic : float
        Observed test statistic from real data (scalar).
    null_values : array-like
        Null distribution values (e.g., from permutations/shuffles).
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
    center : {None, 'empirical', float}, optional
        Centering for both statistic and null:
        - None: no centering
        - 'empirical': center by null mean
        - float: center by given value

    Returns
    -------
    p_value : float
    """
    null_values = np.asarray(null_values).ravel()
    n_null = len(null_values)
    if n_null == 0:
        return np.nan

    # Centering
    if center == 'empirical':
        c = float(null_values.mean())
    elif center is None:
        c = 0.0
    else:
        c = float(center)

    if c != 0.0:
        statistic = statistic - c
        null_values = null_values - c

    if alternative == 'greater':
        return (np.sum(null_values >= statistic) + 1) / (n_null + 1)
    elif alternative == 'less':
        return (np.sum(null_values <= statistic) + 1) / (n_null + 1)
    elif alternative == 'two-sided':
        return (np.sum(np.abs(null_values) >= abs(statistic)) + 1) / (n_null + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


# ---------------------------------------------------------------------
# Helper: extract diff-component from perm_vals
# ---------------------------------------------------------------------
def _extract_diff_perm_vals_array(perm_vals: np.ndarray) -> np.ndarray:
    """
    Extract LCD *difference* component (column 0) from perm_vals.

    Expected shapes:
      - (n_perm, n_neurons, 3)  -> returns (n_perm, n_neurons) = perm_vals[:, :, 0]
      - (n_perm, n_neurons)     -> assumed to already be diff values

    Returns
    -------
    diff_perm_vals : (n_perm, n_neurons) float array
    """
    arr = np.asarray(perm_vals)
    if arr.ndim == 3:
        return arr[:, :, 0].astype(float)
    if arr.ndim == 2:
        return arr.astype(float)
    raise ValueError(
        f"perm_vals must be 2D or 3D (n_perm, n_neurons[, 3]); got shape {arr.shape}"
    )


# ---------------------------------------------------------------------
# Assemble observed + permuted LCD (diff component) across fish for one gene
# ---------------------------------------------------------------------
def assemble_gene_lcd_and_perm_across_fish(
    gene: str,
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_key: str = "visrap",
    mask_type: str = "distance",
    mask: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble for a single gene across multiple fish:

      coords_gene_filt : (N_valid, 3)   coordinates (after NaN filtering)
      vals_gene_filt   : (N_valid, 3)   observed LCD values per neuron
      perm_diff_filt   : (n_perm, N_valid) permuted LCD *difference* per neuron

    IMPORTANT
    ---------
    - We only include fish / neurons for which BOTH observed vals and perm_vals
      exist for this (stim_key, mask_type, mask).
    - NaN filtering is applied AFTER concatenation in a way that matches how
      clustering/summarization do it:
          nan_mask = np.isnan(vals_gene[:, 0]) | np.isnan(coords_gene).any(axis=1)
    """
    coords_gene_list = []
    vals_gene_list = []
    perm_vals_list = []
    fish_ids_list = []

    for fish_n in fish_inspect:
        # Skip fish with no entry for this paradigm or gene
        if fish_n not in LCD_data.get(stim_key, {}):
            continue
        if gene not in LCD_data[stim_key][fish_n]:
            continue

        # Gene index & neuron selection for this fish
        gene_names = fish_data[fish_n]['gene_data']['gene_names']
        gene_inds = np.where(gene_names == gene)[0]
        if gene_inds.size == 0:
            continue
        gene_ind = gene_inds[0]

        gene_neuron_inds = np.where(
            fish_data[fish_n]['gene_data']['gene_counts_binary'][:, gene_ind]
        )[0]
        if gene_neuron_inds.size == 0:
            continue

        # Coordinates for gene-expressing neurons
        coords_fish = fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][gene_neuron_inds]

        # Observed vals: (N_gene_fish, 3)
        try:
            vals_fish = np.asarray(
                LCD_data[stim_key][fish_n][gene]['observed_stats'][mask_type][mask]['vals'],
                dtype=float,
            )
        except KeyError as e:
            raise KeyError(
                f"Missing observed vals for gene '{gene}', fish '{fish_n}', "
                f"mask_type='{mask_type}', mask={mask}: {e}"
            )

        # Permuted vals: (n_perm, N_gene_fish, 3) or (n_perm, N_gene_fish)
        try:
            perm_vals_fish = np.asarray(
                LCD_data[stim_key][fish_n][gene]['perm_stats'][mask_type][mask]['perm_vals'],
                dtype=float,
            )
        except KeyError as e:
            raise KeyError(
                f"Missing perm_vals for gene '{gene}', fish '{fish_n}', "
                f"mask_type='{mask_type}', mask={mask}: {e}"
            )

        # The neuron axis of perm_vals must match the neuron axis of vals
        if perm_vals_fish.shape[-2] != vals_fish.shape[0]:
            raise ValueError(
                f"perm_vals neuron axis does not match vals for gene '{gene}', fish '{fish_n}': "
                f"{perm_vals_fish.shape[-2]} perm neurons vs {vals_fish.shape[0]} vals."
            )

        # Sanity check with coords: we assume LCD vals were computed on
        # exactly the same subset/order as coords_fish / gene_neuron_inds
        if vals_fish.shape[0] != coords_fish.shape[0]:
            raise ValueError(
                f"Mismatch coords vs vals for gene '{gene}' in fish '{fish_n}': "
                f"{coords_fish.shape[0]} coords vs {vals_fish.shape[0]} vals."
            )

        coords_gene_list.append(coords_fish)
        vals_gene_list.append(vals_fish)
        perm_vals_list.append(perm_vals_fish)
        fish_ids_list.append(np.array([fish_n] * coords_fish.shape[0], dtype=object))

    if not coords_gene_list:
        raise ValueError(f"No neurons found for gene '{gene}' in the specified fish.")

    # Concatenate across fish
    coords_gene = np.concatenate(coords_gene_list, axis=0)     # (N_total, 3)
    vals_gene = np.concatenate(vals_gene_list, axis=0)         # (N_total, 3)
    fish_ids_gene = np.concatenate(fish_ids_list, axis=0)      # (N_total,)

    # Ensure common n_perm and concatenate perm vals along neuron axis
    perm_shapes = [pv.shape[0] for pv in perm_vals_list]
    if len(set(perm_shapes)) != 1:
        raise ValueError(
            f"perm_vals have different n_perm across fish for gene '{gene}': {perm_shapes}."
        )
    perm_vals_3d = []
    for pv in perm_vals_list:
        if pv.ndim == 2:
            pv = pv[:, :, None]  # (n_perm, N, 1)
        perm_vals_3d.append(pv)
    perm_vals_concat = np.concatenate(perm_vals_3d, axis=1)    # (n_perm, N_total, 3)

    # Apply same NaN mask as in summarize_significant_clusters_for_gene
    nan_mask = np.isnan(vals_gene[:, 0]) | np.isnan(coords_gene).any(axis=1)
    if np.any(nan_mask):
        coords_gene = coords_gene[~nan_mask]
        vals_gene = vals_gene[~nan_mask]
        fish_ids_gene = fish_ids_gene[~nan_mask]
        perm_vals_concat = perm_vals_concat[:, ~nan_mask, :]

    # Extract diff component from perm_vals -> (n_perm, N_valid)
    perm_diff = _extract_diff_perm_vals_array(perm_vals_concat)

    return coords_gene, vals_gene, perm_diff



# ---------------------------------------------------------------------
# Cluster-level p-values for a single gene
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Cluster-level p-values for a single gene
# ---------------------------------------------------------------------
def compute_cluster_pvals_for_gene(
    gene: str,
    cluster_dict: Dict[str, Dict[str, Any]],
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_keys=None,
    stim_key: str = "visrap",          # kept for backwards compatibility
    mask_type: str = "distance",
    mask: int = 20,
    alternative: str = "two-sided",
    center=None,  # passed to empirical_p_value; e.g. None or 'empirical'
    # ---------------------- NEW (optional, internal) ----------------------
    pooled_accum: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    For a single gene, compute cluster-level p-values for the LCD difference
    using neuron-wise permutation distributions, and update `cluster_dict`
    in place for clusters belonging to this gene.

    If `pooled_accum` is provided, it is updated with per-cluster observed
    and null statistics so that a pooled p-value across clusters can be
    computed outside of `cluster_dict`.
    """
    # -----------------------------
    # Normalize stim_keys argument
    # -----------------------------
    if stim_keys is None:
        if isinstance(stim_key, (list, tuple, set)):
            stim_keys_list = list(stim_key)
        else:
            stim_keys_list = [stim_key]
    else:
        if isinstance(stim_keys, str):
            stim_keys_list = [stim_keys]
        else:
            stim_keys_list = list(stim_keys)

    # Clusters belonging to this gene
    prefix = f"{gene}_C"
    gene_cluster_ids = [cid for cid in cluster_dict.keys() if cid.startswith(prefix)]
    if not gene_cluster_ids:
        return

    # ----------------------------------------------
    # Loop over paradigms and compute p-vals per one
    # ----------------------------------------------
    for sk in stim_keys_list:
        try:
            coords_gene, vals_gene, perm_diff = assemble_gene_lcd_and_perm_across_fish(
                gene=gene,
                LCD_data=LCD_data,
                fish_data=fish_data,
                fish_inspect=fish_inspect,
                stim_key=sk,
                mask_type=mask_type,
                mask=mask,
            )
        except (KeyError, ValueError):
            continue

        diff_obs = vals_gene[:, 0]   # (N_valid,)
        N_valid = diff_obs.shape[0]

        for cid in gene_cluster_ids:
            cdict = cluster_dict[cid]
            inds = np.asarray(cdict.get("indices", []), dtype=int)

            if inds.size == 0:
                cdict.setdefault("cluster_pval", {})
                cdict["cluster_pval"][sk] = {
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "alternative": alternative,
                }
                continue

            valid_mask = (inds >= 0) & (inds < N_valid)
            inds_valid = inds[valid_mask]

            if inds_valid.size == 0:
                cdict.setdefault("cluster_pval", {})
                cdict["cluster_pval"][sk] = {
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "alternative": alternative,
                    "note": f"no valid neurons overlapping permuted LCD array for stim {sk!r}",
                }
                continue

            obs_stat = float(np.nanmean(diff_obs[inds_valid]))
            null_cluster = np.nanmean(perm_diff[:, inds_valid], axis=1)  # (n_perm,)

            p_val = empirical_p_value(
                statistic=obs_stat,
                null_values=null_cluster,
                alternative=alternative,
                center=center,
            )

            cdict.setdefault("cluster_pval", {})
            cdict["cluster_pval"][sk] = {
                "statistic": obs_stat,
                "p_value": float(p_val),
                "alternative": alternative,
            }

            # -------------------------------
            # NEW: accumulate for pooled pval
            # -------------------------------
            if pooled_accum is not None:
                acc = pooled_accum.setdefault(
                    sk,
                    {"obs": [], "null": []},  # lists of scalars and (n_perm,) arrays
                )
                if np.isfinite(obs_stat):
                    acc["obs"].append(float(obs_stat))
                null_cluster = np.asarray(null_cluster, dtype=float).ravel()
                if null_cluster.size > 0 and np.all(np.isfinite(null_cluster)):
                    acc["null"].append(null_cluster)


# ---------------------------------------------------------------------
# Convenience: compute p-values for all clusters (all genes)
# ---------------------------------------------------------------------
def compute_cluster_pvals_all_clusters(
    cluster_dict: Dict[str, Dict[str, Any]],
    LCD_data: Dict[str, Any],
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_keys=None,
    mask_type: str = "distance",
    mask: int = 20,
    alternative: str = "two-sided",
    center=None,
    alpha: float = 0.05,
    # ---------------------- NEW ----------------------
    return_pooled: bool = True,
) -> Union[Dict[str, Dict[str, Any]], Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]]:
    """
    Loop over all genes represented in `cluster_dict`, compute cluster-level
    p-values for each gene, and update `cluster_dict` in place.

    NEW: Also computes a pooled p-value across *all clusters* per paradigm,
    returned separately as `cluster_pooled_pval` (so we do NOT add a 'pooled'
    entry to cluster_dict).

    Returns
    -------
    If return_pooled is False:
        cluster_dict
    If return_pooled is True:
        (cluster_dict, cluster_pooled_pval)

    where cluster_pooled_pval[stim] = {
        "statistic": pooled_mean_of_cluster_means,
        "p_value": pooled_p_value,
        "alternative": ...,
        "significant_standard": ...,
        "n_clusters": ...,
    }
    """
    # -----------------------------
    # Normalize stim_keys argument
    # -----------------------------
    if stim_keys is None:
        stim_keys_list = ["visrap"]
    else:
        if isinstance(stim_keys, str):
            stim_keys_list = [stim_keys]
        else:
            stim_keys_list = list(stim_keys)

    # Infer genes from cluster IDs of the form "GENE_C<number>"
    genes = set()
    for cid in cluster_dict.keys():
        if "_C" in cid:
            g, _ = cid.rsplit("_C", 1)
            genes.add(g)

    # This will store per-stim lists of:
    #  - observed cluster stats (scalars)
    #  - null cluster stats (arrays of shape (n_perm,))
    pooled_accum: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------
    # For each paradigm: compute per-cluster pvals + FDR
    # -------------------------------------------------------
    for sk in stim_keys_list:
        # 1) compute raw p-values per (gene, cluster) for this stim
        for gene in sorted(genes):
            compute_cluster_pvals_for_gene(
                gene=gene,
                cluster_dict=cluster_dict,
                LCD_data=LCD_data,
                fish_data=fish_data,
                fish_inspect=fish_inspect,
                stim_keys=[sk],
                mask_type=mask_type,
                mask=mask,
                alternative=alternative,
                center=center,
                pooled_accum=pooled_accum,   # <-- accumulate pooled across clusters
            )

        # 2) Gather p-values across all clusters for this stim
        cluster_ids = []
        p_raw_list = []

        for cid, cdict in cluster_dict.items():
            cp_all = cdict.get("cluster_pval", {})
            cp = cp_all.get(sk)
            if cp is None:
                continue
            p = cp.get("p_value")
            if p is None or np.isnan(p):
                continue
            cluster_ids.append(cid)
            p_raw_list.append(float(p))

        if len(p_raw_list) > 0:
            p_raw_arr = np.asarray(p_raw_list, dtype=float)
            signif_corr, p_corr_arr = fdrcorrection(p_raw_arr, alpha=alpha)

            # 3) Write back corrected values (per stim)
            for cid, p_raw, p_corr, sig_corr in zip(cluster_ids, p_raw_arr, p_corr_arr, signif_corr):
                cp = cluster_dict[cid]["cluster_pval"][sk]
                cp["p_value_raw"] = float(p_raw)
                cp["p_value_corr"] = float(p_corr)
                cp["significant_standard"] = bool(p_raw < alpha)
                cp["significant_corr"] = bool(sig_corr)

    # -------------------------------------------------------
    # Pooled p-value across ALL clusters (per stim), NO FDR
    # -------------------------------------------------------
    cluster_pooled_pval: Dict[str, Dict[str, Any]] = {}

    levels = pooled_accum  # alias
    for sk in stim_keys_list:
        acc = levels.get(sk, None)
        if acc is None or len(acc.get("obs", [])) == 0 or len(acc.get("null", [])) == 0:
            cluster_pooled_pval[sk] = {
                "statistic": np.nan,
                "p_value": np.nan,
                "alternative": alternative,
                "significant_standard": False,
                "n_clusters": 0,
                "note": "no pooled data available (no clusters or missing nulls)",
            }
            continue

        obs_list = np.asarray(acc["obs"], dtype=float)
        null_list = acc["null"]

        # Require consistent n_perm so we can pool by averaging cluster-level null stats
        n_perm_set = {np.asarray(v).shape[0] for v in null_list}
        if len(n_perm_set) != 1:
            cluster_pooled_pval[sk] = {
                "statistic": float(np.nanmean(obs_list)) if obs_list.size else np.nan,
                "p_value": np.nan,
                "alternative": alternative,
                "significant_standard": False,
                "n_clusters": int(obs_list.size),
                "note": f"inconsistent n_perm across clusters: {sorted(n_perm_set)}",
            }
            continue

        null_mat = np.vstack([np.asarray(v, dtype=float).ravel() for v in null_list])  # (n_clusters, n_perm)

        pooled_stat = float(np.nanmean(obs_list))
        pooled_null = np.nanmean(null_mat, axis=0)  # (n_perm,)

        p_pool = empirical_p_value(
            statistic=pooled_stat,
            null_values=pooled_null,
            alternative=alternative,
            center=center,
        )

        cluster_pooled_pval[sk] = {
            "statistic": float(pooled_stat),
            "p_value": float(p_pool),
            "alternative": alternative,
            "significant_standard": bool(np.isfinite(p_pool) and (p_pool < alpha)),
            "n_clusters": int(obs_list.size),
            # explicitly not FDR-corrected:
            "p_value_corr": float(p_pool) if np.isfinite(p_pool) else np.nan,
            "significant_corr": bool(np.isfinite(p_pool) and (p_pool < alpha)),
        }

    if return_pooled:
        return cluster_dict, cluster_pooled_pval
    return cluster_dict




def calc_wilcoxon_significances(vals, alternative='two-sided', bonferroni: bool = True, alpha: float = 0.05, verbose=False, genes=None, verbose_ordering=None, csv_file=None, csv_colname=None):
    
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import os
    import pandas as pd
    
    # Calculate p-values using wilcoxon signed rank test
    p_vals = np.array([wilcoxon(v.astype('float'), alternative=alternative, method='auto', nan_policy='omit').pvalue
                       for v in vals], dtype=float)
    
    # Apply bonferroni correction if specified
    if bonferroni:
        p_vals_bonferroni = multipletests(p_vals, alpha=alpha, method='bonferroni', is_sorted=False, returnsorted=False)[1]
    
#     if verbose:
#         for i, [v, p, pb] in enumerate(zip(vals, p_vals, p_vals_bonferroni)):
#             print(f"{i}: N values: {len(v)} | p-val: {p} | p-val (bonferroni): {pb}")
            
    if verbose and genes:
        
        if verbose_ordering is not None: ordering = verbose_ordering
        else: ordering = np.arange(len(genes))
        
        if alternative == 'two-sided': sidedness = 'Two-sided'
        elif alternative == 'smaller': sidedness = 'One-sided (smaller)'
        elif alternative == 'greater': sidedness = 'One-sided (greater)' 
        
        main_str = f'{sidedness} Wilcoxon Signed Rank Test.'
        main_str_bf = f'{sidedness} Wilcoxon Signed Rank Test, Bonferroni-corrected.'
        
        for i, [v, p, pb, gene] in enumerate(zip(vals[ordering], p_vals[ordering], 
                                                 p_vals_bonferroni[ordering], np.array(genes)[ordering])):
            if p == 0: main_str += f' {gene}, p<ε;'
            else: main_str += f' {gene}, {p:.3e};'
            if pb == 0: main_str_bf += f' {gene}, p<ε;'
            else: main_str_bf += f' {gene}, {pb:.3e};'
            
        print(main_str + '\n')
        print(main_str_bf)
    
    # if csv_file is not None and csv_colname is not None:
    #     df_out_data = {
    #         csv_colname: genes if genes is not None else np.arange(len(p_vals)),
    #         'p-value': [p if p > 0 else 'p<ε' for p in p_vals]
    #     }

    #     if bonferroni:
    #         df_out_data['p-value (bonferroni-corrected)'] = [
    #             p if p > 0 else 'p<ε' for p in p_vals_bonferroni
    #         ]

    #     df_out = pd.DataFrame(df_out_data)
    #     df_out.to_csv(csv_file, index=False)
    #     if verbose:
    #         print(f"CSV file saved to {csv_file}")
    
    return p_vals, p_vals_bonferroni


def get_sig_asterisk_inds(p_vals, alphas: list = [0.05, 0.01, 0.001]):
    alphas = alphas + [0]
    sig_asterisk_inds = [np.where(p_vals>alphas[0])[0]] + [np.where((p_vals >= alphas[i+1]) & (p_vals <= alphas[i]))[0] for i in range(len(alphas)-1)]

    asterisks = ['ns'] + ['*'*(i+1) for i in range(len(alphas))]
    sig_asterisks = [{ind: asterisks[a_ind] for ind in inds} for a_ind, inds in enumerate(sig_asterisk_inds)]
    sig_asterisks = {k: v for d in sig_asterisks for k, v in d.items()}
    
    return sig_asterisk_inds, sig_asterisks