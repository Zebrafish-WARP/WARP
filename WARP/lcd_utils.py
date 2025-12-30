import numpy as np
from scipy.spatial import distance_matrix
from statsmodels.stats.multitest import multipletests
from numba import njit, prange
from joblib import Parallel, delayed

from typing import Dict, Any, Sequence, Optional, Tuple, Mapping, Callable

from tqdm_joblib import tqdm_joblib
from tqdm.auto import tqdm

from WARP.utils import calc_neighbor_list

from WARP.lcd_statistics import calc_wilcoxon_significances
from WARP.lcd_statistics import get_sig_asterisk_inds
from WARP.visualization import make_cmap_dict

# -------------------------------
# Correlation utilities
# -------------------------------
def compute_corrmat(X, Y=None, dtype=np.float32):
    if Y is None:
        Y = X
    X_z = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    Y_z = (Y - Y.mean(axis=1, keepdims=True)) / Y.std(axis=1, keepdims=True)
    X_z = X_z.astype(dtype)
    Y_z = Y_z.astype(dtype)
    corrmat = np.dot(X_z, Y_z.T) / X_z.shape[1]
    return np.clip(corrmat, -1, 1)


def compute_corrmat_blockwise(X, Y=None, block_size=5000, dtype=np.float32):
    """
    Compute blockwise correlation matrix between rows of X and Y.

    Parameters
    ----------
    X : ndarray, shape (n_x, n_t)
        Input data matrix (each row is a variable, columns are samples).
    Y : ndarray, shape (n_y, n_t), optional
        Second matrix to correlate with X. If None, uses X.
    block_size : int
        Number of rows per block to process at once.
    dtype : np.dtype
        Precision for intermediate computations to save memory.

    Returns
    -------
    corrmat : ndarray, shape (n_x, n_y)
        Correlation matrix between rows of X and Y.
    """
    if Y is None:
        Y = X

    # Ensure float32/64
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=dtype)

    n_x, n_t = X.shape
    n_y = Y.shape[0]

    # Precompute means and stds to avoid recomputation
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_std = Y.std(axis=1, keepdims=True)

    corrmat = np.empty((n_x, n_y), dtype=dtype)

    # Normalize Y once (since reused across X blocks)
    Y_z = (Y - Y_mean) / Y_std

    # Process X in blocks
    for i in range(0, n_x, block_size):
        X_block = X[i:i + block_size]
        X_z = (X_block - X_mean[i:i + block_size]) / X_std[i:i + block_size]
        corr_block = np.dot(X_z, Y_z.T) / n_t
        np.clip(corr_block, -1, 1, out=corr_block)
        corrmat[i:i + block_size] = corr_block
        del X_block, X_z, corr_block

    return corrmat

# -------------------------------
# Fisher transform
# -------------------------------
def fisher_z(r): 
    return np.arctanh(np.clip(r, -0.999999, 0.999999))

def inv_fisher_z(z): 
    return np.tanh(z)


@njit
def nanmean_2d(mat):
    n_rows, n_cols = mat.shape
    res = np.empty(n_rows, dtype=mat.dtype)
    for i in range(n_rows):
        s = 0.0
        n = 0
        for j in range(n_cols):
            v = mat[i, j]
            if not np.isnan(v):
                s += v
                n += 1
        if n > 0:
            res[i] = s / n
        else:
            res[i] = np.nan
    return res

# -------------------------------
# Compute LCD matrix for all neurons
# -------------------------------
@njit(parallel=True)
def compute_LCD(gene_neuron_inds, neighbor_list_arr, neighbor_list_len, corrmat):
    """
    gene_neuron_inds: array of neuron indices for this gene
    neighbor_list_arr: pre-padded neighbor indices (shape: n_neurons x max_neighbors, -1 padding)
    neighbor_list_len: number of valid neighbors per neuron
    corrmat: full correlation matrix (NxN)
    
    Returns: LCD_mat (n_neurons x n_neurons x 3)
        [:, :, 0] = LCD (same - other)
        [:, :, 1] = corr_same
        [:, :, 2] = corr_other
    """
    n_neurons = gene_neuron_inds.shape[0]
    LCD_mat = np.full((n_neurons, n_neurons, 3), np.nan, dtype=np.float32)
    
    for idx in prange(n_neurons):
        neuron_i = gene_neuron_inds[idx]
        
        # corr_same: correlation with this neuron
        corr_same = corrmat[:, neuron_i]
        
        # corr_other: correlation with neighbors excluding same-type neurons
        n_neighbors = neighbor_list_len[idx]
        if n_neighbors > 0:
            neighbor_inds = neighbor_list_arr[idx, :n_neighbors]
            corr_other = nanmean_2d(corrmat[:, neighbor_inds])
        else:
            corr_other = np.full(n_neurons, np.nan, dtype=np.float32)
        
        # LCD vector: same - other
        lcd_vec = np.empty((n_neurons, 3), dtype=np.float32)
        for j in range(n_neurons):
            lcd_vec[j, 0] = corr_same[j] - corr_other[j]
            lcd_vec[j, 1] = corr_same[j]
            lcd_vec[j, 2] = corr_other[j]
        
        lcd_vec[idx, :] = np.nan  # NaN on diagonal
        LCD_mat[:, idx, :] = lcd_vec
        
    return LCD_mat


import numpy as np
from numba import njit, prange

# -------------------------------
# Mask statistics utility
# -------------------------------

@njit(fastmath=False)
def nanmean_axis0_3(vals):
    """
    vals: (n_rows, 3)
    returns: (3,) nanmean over rows
    """
    n_rows = vals.shape[0]
    out = np.empty(3, dtype=np.float64)

    for k in range(3):
        s = 0.0
        n = 0
        for i in range(n_rows):
            v = vals[i, k]
            if v == v:  # not NaN
                s += v
                n += 1
        out[k] = s / n if n > 0 else np.nan

    return out


@njit(fastmath=False)
def nanmedian_axis0_3(vals):
    """
    vals: (n_rows, 3)
    returns: (3,) nanmedian over rows
    """
    n_rows = vals.shape[0]
    out = np.empty(3, dtype=np.float64)
    tmp = np.empty(n_rows, dtype=np.float64)

    for k in range(3):
        m = 0
        # collect non-NaN values in tmp[:m]
        for i in range(n_rows):
            v = vals[i, k]
            if v == v:  # not NaN
                tmp[m] = v
                m += 1

        if m == 0:
            out[k] = np.nan
        else:
            arr = tmp[:m].copy()
            arr.sort()
            mid = m // 2
            if m % 2 == 1:
                out[k] = arr[mid]
            else:
                out[k] = 0.5 * (arr[mid - 1] + arr[mid])

    return out


# -------------------------------
# Jitted mask statistics
# -------------------------------
@njit(parallel=True, fastmath=False)
def compute_mask_statistics_jit(data_mat, mask_idx_list, mask_len_list):
    """
    data_mat: (n_rows, n_cols, 3)
    mask_idx_list: (n_rows, max_len)
    mask_len_list: (n_rows,)
    """
    n_rows = mask_idx_list.shape[0]
    vals = np.empty((n_rows, 3), dtype=np.float64)

    for i in prange(n_rows):
        valid = mask_len_list[i]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        n = 0

        for j in range(valid):
            idx = mask_idx_list[i, j]
            if idx < 0:
                continue

            v0 = data_mat[i, idx, 0]
            v1 = data_mat[i, idx, 1]
            v2 = data_mat[i, idx, 2]

            # skip neighbor if any feature is NaN
            if (v0 == v0) and (v1 == v1) and (v2 == v2):
                s0 += v0
                s1 += v1
                s2 += v2
                n += 1

        if n > 0:
            vals[i, 0] = s0 / n
            vals[i, 1] = s1 / n
            vals[i, 2] = s2 / n
        else:
            vals[i, 0] = np.nan
            vals[i, 1] = np.nan
            vals[i, 2] = np.nan

    mean_val = nanmean_axis0_3(vals)
    median_val = nanmedian_axis0_3(vals)
    return mean_val, median_val, vals


# -------------------------------
# Main wrapper (handles multiple masks)
# -------------------------------
def compute_mask_statistics(LCD_mat, mask_dicts, store_mask_values=None):
    """
    Compute mean/median over masked values for multiple masks.

    LCD_mat: (n_rows x n_cols x 3)
    mask_dicts: dict of dict of masks (list-of-arrays per row)
    store_mask_values: dict specifying whether to store all per-row values
    """
    if LCD_mat is None or mask_dicts is None:
        return {}

    if store_mask_values is None:
        store_mask_values = {k: False for k in mask_dicts}

    mask_statistics = {}
    n_rows = LCD_mat.shape[0]

    for k_i, masks_dict in mask_dicts.items():
        mask_statistics[k_i] = {}
        store_vals = store_mask_values.get(k_i, False)

        for k_j, mask in masks_dict.items():
            mask_statistics[k_i][k_j] = {}

            if not mask:
                continue

            # mask is list of [ind, row_indices]
            max_len = max(len(row) for (ind, row) in mask)

            mask_idx_arr = np.full((n_rows, max_len), -1, dtype=np.int32)
            mask_len_arr = np.zeros(n_rows, dtype=np.int32)

            for ind, row in mask:
                L = len(row)
                mask_len_arr[ind] = L
                if L > 0:
                    mask_idx_arr[ind, :L] = row

            mean_val, median_val, vals = compute_mask_statistics_jit(
                LCD_mat, mask_idx_arr, mask_len_arr
            )

            mask_statistics[k_i][k_j]["mean"] = mean_val
            mask_statistics[k_i][k_j]["median"] = median_val

            if store_vals:
                mask_statistics[k_i][k_j]["vals"] = vals

    return mask_statistics



# -------------------------------
# Main wrapper (handles multiple masks)
# -------------------------------
# -------------------------------
# Main wrapper (handles multiple masks)
# -------------------------------
def compute_mask_statistics(LCD_mat, mask_dicts, store_mask_values=None):
    """
    Compute mean/median over masked values for multiple masks.

    LCD_mat: (n_rows x n_cols x 3)
    mask_dicts: dict of dict of masks (list-of-arrays per row)
    store_mask_values: dict specifying whether to store all per-row values
    """
    if LCD_mat is None or mask_dicts is None:
        return {}

    if store_mask_values is None:
        store_mask_values = {k: False for k in mask_dicts}

    mask_statistics = {}
    n_rows = LCD_mat.shape[0]

    for k_i, masks_dict in mask_dicts.items():
        mask_statistics[k_i] = {}
        store_vals = store_mask_values.get(k_i, False)

        for k_j, mask in masks_dict.items():
            mask_statistics[k_i][k_j] = {}

            if not mask:
                continue

            # mask is list of [ind, row_indices]
            max_len = max(len(row) for (ind, row) in mask)

            mask_idx_arr = np.full((n_rows, max_len), -1, dtype=np.int32)
            mask_len_arr = np.zeros(n_rows, dtype=np.int32)

            for ind, row in mask:
                L = len(row)
                mask_len_arr[ind] = L
                if L > 0:
                    mask_idx_arr[ind, :L] = row

            mean_val, median_val, vals = compute_mask_statistics_jit(
                LCD_mat, mask_idx_arr, mask_len_arr
            )

            mask_statistics[k_i][k_j]["mean"] = mean_val
            mask_statistics[k_i][k_j]["median"] = median_val

            if store_vals:
                mask_statistics[k_i][k_j]["vals"] = vals

    return mask_statistics
    

@njit(parallel=True, fastmath=False)  # fastmath=False because of NaN checks
def compute_lcd_block(
    corrmat,
    perm_neurons,
    perm_neighbors,
    neighbor_counts,
):
    """
    corrmat:        (N_gene_rows, N_all_neurons)  (float32 or float64)
    perm_neurons:   (n_gene_neurons, p_cur)
    perm_neighbors: (n_gene_neurons, p_cur, max_neighbors)
    neighbor_counts:(n_gene_neurons,)

    Returns
    -------
    lcd_block: (N_gene_rows, n_gene_neurons, 3, p_cur)
        For each row r, neuron i, perm p:
            [0] = cs - cs_o
            [1] = cs           (self corr)
            [2] = cs_o         (neighbor mean)
    """
    N_gene_rows = corrmat.shape[0]            # = n_gene_neurons in your setup
    n_gene_neurons, p_cur = perm_neurons.shape
    max_neighbors = perm_neighbors.shape[2]

    lcd_block = np.empty((N_gene_rows, n_gene_neurons, 3, p_cur),
                         dtype=corrmat.dtype)

    for i in prange(n_gene_neurons):
        n_use = neighbor_counts[i]
        max_k = n_use if n_use < max_neighbors else max_neighbors

        for p in range(p_cur):
            col = perm_neurons[i, p]

            # invalid neuron index → fill with NaNs
            if col < 0:
                for r in range(N_gene_rows):
                    lcd_block[r, i, 0, p] = np.nan
                    lcd_block[r, i, 1, p] = np.nan
                    lcd_block[r, i, 2, p] = np.nan
                continue

            for r in range(N_gene_rows):
                cs = corrmat[r, col]

                # if cs is NaN → all three are NaN
                if not (cs == cs):
                    lcd_block[r, i, 0, p] = np.nan
                    lcd_block[r, i, 1, p] = np.nan
                    lcd_block[r, i, 2, p] = np.nan
                    continue

                s = 0.0
                c = 0

                # neighbors
                for k in range(max_k):
                    nidx = perm_neighbors[i, p, k]
                    if nidx < 0:
                        break
                    v = corrmat[r, nidx]
                    if v == v:  # not NaN
                        s += v
                        c += 1

                if c == 0:
                    lcd_block[r, i, 0, p] = np.nan
                    lcd_block[r, i, 1, p] = np.nan
                    lcd_block[r, i, 2, p] = np.nan
                else:
                    cs_o = s / c
                    lcd_block[r, i, 0, p] = cs - cs_o
                    lcd_block[r, i, 1, p] = cs
                    lcd_block[r, i, 2, p] = cs_o

    return lcd_block


def choose_p_chunk(n_gene_neurons, base_p_chunk, dtype=np.float32,
                   max_lcd_mem_bytes=512 * 1024**2):
    """
    Choose an adaptive p_chunk so that lcd_block roughly fits within
    max_lcd_mem_bytes for a given gene.

    lcd_block has shape (n_gene_neurons, n_gene_neurons, 3, p_chunk).
    """
    if n_gene_neurons <= 0:
        return 0

    bytes_per_float = np.dtype(dtype).itemsize
    bytes_per_perm = n_gene_neurons * n_gene_neurons * 3 * bytes_per_float

    if bytes_per_perm <= 0:
        return int(base_p_chunk)

    max_p_by_mem = max_lcd_mem_bytes // bytes_per_perm
    if max_p_by_mem < 1:
        max_p_by_mem = 1

    return int(min(base_p_chunk, max_p_by_mem))

# ---------- High-level wrapper ----------
def compute_permuted_LCD_stats(
    gene_name,
    gene_neuron_inds,
    neighbor_list_arr,      # full neighbor list as object array
    neighbor_list_gene,     # neighbor list entries for gene neurons (object array)
    corrmat,
    N_permute,
    mask_dicts=None,
    store_mask_values=None,
    rng=None,
    min_neighbors=5,
    p_chunk=50,
    max_permutation_GB=0.5,
):
    """
    Chunked permutation computation, using numba kernel.

    If store_mask_values[m] is True for a mask family m, then in addition to
    perm_mean / perm_median, this function will also store per-neuron
    permutation values in perm_vals:

        perm_stats_all[m][k]['perm_vals'] : (N_permute, n_rows, 3)

    where n_rows = number of rows in the LCD matrix (usually n_gene_neurons).
    """

    if rng is None:
        rng = np.random.default_rng()

    if mask_dicts is None or len(mask_dicts) == 0:
        return None

    # normalize store_mask_values
    if store_mask_values is None:
        store_mask_values = {m: False for m in mask_dicts.keys()}

    n_gene_neurons = len(gene_neuron_inds)
    n_rows = corrmat.shape[0]  # rows in LCD_mat / lcd_perm

    # how many "other" neighbors per gene neuron (excluding same-type neurons)
    N_neighbors_other_list = np.array(
        [len(np.setdiff1d(l, gene_neuron_inds)) for l in neighbor_list_gene],
        dtype=np.int64
    )

    # convert GB → bytes for lcd_block budget
    max_lcd_mem_bytes = int(max_permutation_GB * (1024**3))

    # choose adaptive chunk size based on n_gene_neurons and dtype
    effective_p_chunk = choose_p_chunk(
        n_gene_neurons,
        base_p_chunk=p_chunk,
        dtype=corrmat.dtype,
        max_lcd_mem_bytes=max_lcd_mem_bytes,
    )

    # optional extra clamp for very large genes
    if n_gene_neurons > 10000:
        effective_p_chunk = min(effective_p_chunk, 5)

    # prepare output structure
    perm_stats_all = {
        m: {
            k: {
                'perm_mean': np.zeros((N_permute, 3), dtype=float),
                'perm_median': np.zeros((N_permute, 3), dtype=float),
            }
            for k in mask_dicts[m].keys()
        }
        for m in mask_dicts.keys()
    }

    # allocate perm_vals only for mask families flagged in store_mask_values
    for m in mask_dicts.keys():
        if store_mask_values.get(m, False):
            for k in mask_dicts[m].keys():
                perm_stats_all[m][k]['perm_vals'] = np.full(
                    (N_permute, n_rows, 3),
                    np.nan,
                    dtype=float,
                )

    # pre-allocate an all-NaN LCD for trivial chunks (no neighbors)
    lcd_nan = np.full((n_rows, n_gene_neurons, 3), np.nan, dtype=corrmat.dtype)

    for start_p in tqdm(
        range(0, N_permute, effective_p_chunk),
        desc=f'perm chunks for gene {gene_name} ({n_gene_neurons} neurons)'
    ):
        end_p = min(N_permute, start_p + effective_p_chunk)
        p_cur = end_p - start_p

        # Build perm_neurons (n_gene_neurons x p_cur), -1 if invalid
        perm_neurons = np.full((n_gene_neurons, p_cur), -1, dtype=np.int64)
        for i in range(n_gene_neurons):
            neighs = neighbor_list_gene[i]
            if len(neighs) > min_neighbors:
                perm_neurons[i, :] = rng.choice(neighs, size=p_cur, replace=True)
            else:
                # leave as -1
                pass

        # compute max neighbors per gene neuron
        max_neighbors_pad = int(np.max(N_neighbors_other_list)) if N_neighbors_other_list.size > 0 else 0
        if max_neighbors_pad == 0:
            # nothing to compute in this chunk: stats come from all-NaN LCD
            for local_idx in range(p_cur):
                perm_idx = start_p + local_idx
                perm_mask_stats = compute_mask_statistics(lcd_nan, mask_dicts, store_mask_values)
                for mname, sub in perm_mask_stats.items():
                    for k, vals in sub.items():
                        perm_stats_all[mname][k]['perm_mean'][perm_idx] = vals['mean']
                        perm_stats_all[mname][k]['perm_median'][perm_idx] = vals['median']
                        if 'perm_vals' in perm_stats_all[mname][k]:
                            perm_stats_all[mname][k]['perm_vals'][perm_idx, :, :] = vals['vals']
            continue

        # Build perm_neighbors padded array (n_gene_neurons x p_cur x max_neighbors_pad)
        perm_neighbors = np.full((n_gene_neurons, p_cur, max_neighbors_pad), -1, dtype=np.int64)

        for i in range(n_gene_neurons):
            n_other = int(N_neighbors_other_list[i])
            if n_other == 0:
                continue
            for p in range(p_cur):
                pn = perm_neurons[i, p]
                if pn < 0:
                    continue
                neigh_list_pn = neighbor_list_arr[pn]
                if len(neigh_list_pn) == 0:
                    continue
                draw = rng.choice(neigh_list_pn, size=n_other, replace=True)
                perm_neighbors[i, p, :n_other] = draw

        neighbor_counts = N_neighbors_other_list.copy()

        # Call numba kernel to compute lcd_block
        lcd_block = compute_lcd_block(
            corrmat,
            perm_neurons,
            perm_neighbors,
            neighbor_counts
        )

        # For each permutation in chunk, compute mask stats
        for local_idx in range(p_cur):
            perm_idx = start_p + local_idx
            # slice over permutation axis, keep 3 features
            lcd_perm = lcd_block[:, :, :, local_idx]  # (n_rows, n_gene_neurons, 3)
            perm_mask_stats = compute_mask_statistics(lcd_perm, mask_dicts, store_mask_values)

            for mname, sub in perm_mask_stats.items():
                for k, vals in sub.items():
                    perm_stats_all[mname][k]['perm_mean'][perm_idx] = vals['mean']
                    perm_stats_all[mname][k]['perm_median'][perm_idx] = vals['median']
                    # if we allocated perm_vals for this mask family, also store per-neuron vals
                    if 'perm_vals' in perm_stats_all[mname][k]:
                        # vals['vals'] is (n_rows, 3)
                        perm_stats_all[mname][k]['perm_vals'][perm_idx, :, :] = vals['vals']

    return perm_stats_all


# -------------------------------
# Compute LCD for one gene
# -------------------------------
def process_LCD_gene(
    fish_n,
    gene_name,
    functional_timeseries,
    gene_counts_binary,
    gene_names,
    neighbor_list_arr,
    N_permute=None,
    fisher_transform=False,
    p_chunk=100,
    seed=42,
    mask_dicts=None,
    store_full_LCD=False,
    store_mask_values=None,
    fish_paths=None,
    max_permutation_GB=0.5,
):

    # Lazily load data if paths provided
    if (functional_timeseries is None or gene_counts_binary is None) and fish_paths is not None:
        functional_timeseries = np.load(fish_paths[0])
        gene_counts_binary = np.load(fish_paths[1])

    rng = np.random.default_rng(seed)

    gene_ind = np.where(gene_names == gene_name)[0][0]
    gene_neuron_inds = np.where(gene_counts_binary[:, gene_ind])[0]

    n_neurons = len(gene_neuron_inds)

    if n_neurons == 0:
        return {'LCD_mat': None, 'observed_stats': None, 'perm_stats': None}

    # neighbor_list entries for gene neurons
    neighbor_list_gene = neighbor_list_arr[gene_neuron_inds]

    # compute correlations for gene neurons vs all neurons
    X = functional_timeseries[gene_neuron_inds]
    Y = functional_timeseries

    corrmat = compute_corrmat(X, Y) if n_neurons <= 5000 else compute_corrmat_blockwise(X, Y, block_size=5000)
    # use float32 everywhere to reduce memory and bandwidth
    corrmat = corrmat.astype(np.float32, copy=False)

    # Build per-gene-neuron neighbor arrays (excluding same-type neurons)
    max_neighbors = 0
    neighbor_list_arr_gene = []
    neighbor_list_len = np.empty(n_neurons, dtype=np.int32)

    for i in range(n_neurons):
        inds = np.setdiff1d(neighbor_list_gene[i], gene_neuron_inds)
        neighbor_list_arr_gene.append(inds)
        L = len(inds)
        neighbor_list_len[i] = L
        if L > max_neighbors:
            max_neighbors = L

    neighbor_list_arr_padded = np.full((n_neurons, max_neighbors), -1, dtype=np.int32)
    for i in range(n_neurons):
        inds = neighbor_list_arr_gene[i]
        L = len(inds)
        if L > 0:
            neighbor_list_arr_padded[i, :L] = inds

    # Compute observed LCD
    LCD_mat = compute_LCD(
        np.array(gene_neuron_inds, dtype=np.int32),
        neighbor_list_arr_padded,
        neighbor_list_len,
        corrmat,
    )

    # Observed mask statistics
    obs_mask_stats = None
    if mask_dicts:
        if LCD_mat is None:
            LCD_mat_temp = np.full((n_neurons, n_neurons, 3), np.nan, dtype=np.float32)
            obs_mask_stats = compute_mask_statistics(LCD_mat_temp, mask_dicts, store_mask_values)
        else:
            obs_mask_stats = compute_mask_statistics(LCD_mat, mask_dicts, store_mask_values)

    # Permutation statistics
    perm_stats = None
    if N_permute is not None:
        perm_stats = compute_permuted_LCD_stats(
            gene_name,
            gene_neuron_inds,
            neighbor_list_arr=neighbor_list_arr,
            neighbor_list_gene=neighbor_list_gene,
            corrmat=corrmat,
            N_permute=N_permute,
            mask_dicts=mask_dicts,
            store_mask_values=store_mask_values,
            rng=rng,
            p_chunk=p_chunk,
            max_permutation_GB=max_permutation_GB,
        )

    if not store_full_LCD:
        LCD_mat = None

    return {'LCD_mat': LCD_mat, 'observed_stats': obs_mask_stats, 'perm_stats': perm_stats}

# -------------------------------
# Compute LCD for all genes/fish
# -------------------------------

def compute_LCD_all(
    fish_inspect,
    fish_data,
    gene_names,
    gene_names_inspect,
    paradigm,
    neighbor_radius_inner,
    neighbor_radius_outer,
    use_genes=True,
    shuffle=False,
    N_permute=None,
    n_jobs_genes=1,
    p_chunk=100,
    fisher_transform=False,
    store_full_LCD=False,
    mask_dicts=None,
    store_mask_values=None,
    seed=1,
    cluster=None,
    cluster_kwargs=None,
    tmp_root='/',
    max_permutation_GB=0.5,
):

    LCD_results = {}

    if cluster_kwargs:
        from dask import delayed, compute
        from ClusterWrap.decorator import cluster as cluster_decorator
        import tempfile

        tmp_dir_obj = tempfile.TemporaryDirectory(dir=tmp_root)

        fish_paths = {}
        for fish_n in fish_inspect:
            func_path = f"{tmp_dir_obj.name}/func_{fish_n}.npy"
            np.save(func_path, fish_data[fish_n]['functional_data'][paradigm])

            gene_path = f"{tmp_dir_obj.name}/gene_{fish_n}.npy"
            np.save(gene_path, fish_data[fish_n]['gene_data']['gene_counts_binary'])

            fish_paths[fish_n] = (func_path, gene_path)

        @cluster_decorator
        @cluster_decorator
        def submit_all_fish_all_genes(cluster=None, cluster_kwargs={}):
            tasks = []
            for fish_n in tqdm(fish_inspect, desc='Building task list for individual fish'):
                neighbor_list = calc_neighbor_list(
                    fish_data[fish_n]['cell_centers_data']['cell_centers_zb'],
                    dist_thr_inner=neighbor_radius_inner,
                    dist_thr_outer=neighbor_radius_outer
                )
                neighbor_list_arr = np.array(neighbor_list, dtype=object)

                for gene_name in gene_names_inspect:
                    mask_dicts_gene = {k: d[fish_n][gene_name] for k, d in mask_dicts.items()}

                    task = delayed(process_LCD_gene)(
                        fish_n=fish_n,
                        gene_name=gene_name,
                        gene_names=gene_names,
                        functional_timeseries=None,
                        gene_counts_binary=None,
                        fish_paths=fish_paths[fish_n],
                        neighbor_list_arr=neighbor_list_arr,
                        N_permute=N_permute,
                        fisher_transform=fisher_transform,
                        p_chunk=p_chunk,
                        seed=seed,
                        mask_dicts=mask_dicts_gene,
                        store_full_LCD=store_full_LCD,
                        store_mask_values=store_mask_values,
                        max_permutation_GB=max_permutation_GB,
                    )
                    tasks.append((fish_n, gene_name, task))

            futures = compute(*[t for _, _, t in tasks])

            LCD_results = {}
            for (fish_n, gene_name, _), fut in zip(tasks, futures):
                if fish_n not in LCD_results:
                    LCD_results[fish_n] = {}
                LCD_results[fish_n][gene_name] = fut

            tmp_dir_obj.cleanup()
            return LCD_results

        LCD_results = submit_all_fish_all_genes(
            cluster=cluster,
            cluster_kwargs=cluster_kwargs,
        )

    else:
        for fish_n in fish_inspect:
            print(f"Processing fish: {fish_n}")
            neighbor_list = calc_neighbor_list(
                fish_data[fish_n]['cell_centers_data']['cell_centers_zb'],
                dist_thr_inner=neighbor_radius_inner,
                dist_thr_outer=neighbor_radius_outer
            )
            neighbor_list_arr = np.array(neighbor_list, dtype=object)

            if n_jobs_genes in (1, None):
                iterator = tqdm(gene_names_inspect, desc="Calculating LCDs for all genes")
                gene_results = {}
                for gene_name in iterator:
                    gene_results[gene_name] = process_LCD_gene(
                        fish_n=fish_n,
                        gene_name=gene_name,
                        functional_timeseries=fish_data[fish_n]['functional_data'][paradigm],
                        gene_counts_binary=fish_data[fish_n]['gene_data']['gene_counts_binary'],
                        gene_names=gene_names,
                        neighbor_list_arr=neighbor_list_arr,
                        N_permute=N_permute,
                        fisher_transform=fisher_transform,
                        p_chunk=p_chunk,
                        seed=seed,
                        mask_dicts={k: d[fish_n][gene_name] for k, d in mask_dicts.items()},
                        store_full_LCD=store_full_LCD,
                        store_mask_values=store_mask_values,
                        max_permutation_GB=max_permutation_GB,  # <--- here
                    )
            else:
                with tqdm_joblib(desc="Calculating LCDs for all genes", total=len(gene_names_inspect)) as progress_bar:
                    results = Parallel(n_jobs=n_jobs_genes)(
                        delayed(process_LCD_gene)(
                            fish_n=fish_n,
                            gene_name=gene_name,
                            functional_timeseries=fish_data[fish_n]['functional_data'][paradigm],
                            gene_counts_binary=fish_data[fish_n]['gene_data']['gene_counts_binary'],
                            gene_names=gene_names,
                            neighbor_list_arr=neighbor_list_arr,
                            N_permute=N_permute,
                            fisher_transform=fisher_transform,
                            p_chunk=p_chunk,
                            seed=seed,
                            mask_dicts={k: d[fish_n][gene_name] for k, d in mask_dicts.items()},
                            store_full_LCD=store_full_LCD,
                            store_mask_values=store_mask_values,
                            max_permutation_GB=max_permutation_GB,  # <--- here
                        )
                        for gene_name in gene_names_inspect
                    )
                gene_results = {gene_name: result for gene_name, result in zip(gene_names_inspect, results)}

            LCD_results[fish_n] = gene_results

    return LCD_results



# ==================================================================================
# Average LCD over distances
# ==================================================================================


def compute_LCD_distance_average(fish_data, LCD_data, dist_max_list=[20], dist_min_list=[0], min_neurons=5):

    LCD_data_dist_avg = {}
    for paradigm in tqdm(LCD_data.keys()):
        LCD_data_dist_avg[paradigm] = {}
        for fish_n in tqdm(LCD_data[paradigm].keys(), leave=False):

            LCD_data_dist_avg[paradigm][fish_n] = {}

            gene_names = fish_data[fish_n]['gene_data']['gene_names']

            for gene in tqdm(LCD_data[paradigm][fish_n].keys(), leave=False):

                LCD_data_dist_avg[paradigm][fish_n][gene] = {}

                gene_ind = np.where(gene_names == gene)[0][0]
                gc_binary = fish_data[fish_n]['gene_data']['gene_counts_binary']
                gene_neuron_inds = np.where(gc_binary[:, gene_ind])[0]

                distmat = distance_matrix(fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][gene_neuron_inds], 
                                          fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][gene_neuron_inds])

                for dist_min, dist_max in zip(dist_min_list, dist_max_list):

                    LCD_data_dist_avg[paradigm][fish_n][gene][dist_max] = {}

                    distmat_mask = (distmat>=dist_min) & (distmat<=dist_max)

                    distmat_mask[distmat_mask.sum(1)<min_neurons] = False

                    LCD_mat_mask = np.where(distmat_mask, LCD_data[paradigm][fish_n][gene][..., 0], np.nan)
                    LCD_data_dist_avg[paradigm][fish_n][gene][dist_max] = np.nanmean(LCD_mat_mask, axis=1)
    
    return LCD_data_dist_avg

from scipy.spatial import cKDTree
def get_LCD_distance_average_masks(
        fish_data, 
        dist_max_list=[20], 
        dist_min_list=[0], 
        min_neurons=5
    ):

    LCD_data_dist_avg_masks = {}

    for fish_n in tqdm(fish_data.keys(), leave=False):
        LCD_data_dist_avg_masks[fish_n] = {}

        gene_names = fish_data[fish_n]['gene_data']['gene_names']
        gc_binary  = fish_data[fish_n]['gene_data']['gene_counts_binary']
        centers    = fish_data[fish_n]['cell_centers_data']['cell_centers_zb']

        for gene in tqdm(gene_names, leave=False):
            LCD_data_dist_avg_masks[fish_n][gene] = {}

            gene_ind = np.where(gene_names == gene)[0][0]
            gene_neuron_inds = np.where(gc_binary[:, gene_ind])[0]

            if len(gene_neuron_inds) < min_neurons:
                continue

            # Construct KD-tree over the expressing neurons
            local_centers = centers[gene_neuron_inds]
            tree = cKDTree(local_centers)

            # Map: local → global
            # global index = gene_neuron_inds[local_index]

            for dist_min, dist_max in zip(dist_min_list, dist_max_list):

                # Query all unordered pairs within dist_max
                pairs = np.array(list(tree.query_pairs(dist_max)), dtype=np.int32)

                if pairs.size == 0:
                    # No masks
                    LCD_data_dist_avg_masks[fish_n][gene][dist_max] = [np.array([], dtype=np.int32)
                                                                       for _ in range(len(gene_neuron_inds))]
                    continue

                # Make pairs bidirectional
                pairs = np.vstack([pairs, pairs[:, ::-1]])

                # Apply lower bound
                if dist_min > 0:
                    diffs = local_centers[pairs[:, 0]] - local_centers[pairs[:, 1]]
                    dists = np.sqrt(np.sum(diffs**2, axis=1))
                    pairs = pairs[dists >= dist_min]

                    if len(pairs) == 0:
                        LCD_data_dist_avg_masks[fish_n][gene][dist_max] = \
                            [np.array([], dtype=np.int32) for _ in range(len(gene_neuron_inds))]
                        continue

                # Filter low-neighborhood neurons
                neighbor_counts = np.bincount(pairs[:, 0], minlength=len(gene_neuron_inds))
                valid = np.where(neighbor_counts >= min_neurons)[0]

                mask_valid = np.isin(pairs[:, 0], valid) & np.isin(pairs[:, 1], valid)
                pairs = pairs[mask_valid]

                # Build final mask: list-of-arrays
                final_masks = []
                for i in range(len(gene_neuron_inds)):
                    targets = pairs[pairs[:, 0] == i, 1]
                    final_masks.append((i, targets.astype(np.int32)))

                LCD_data_dist_avg_masks[fish_n][gene][dist_max] = final_masks

    return LCD_data_dist_avg_masks


# ==================================================================================
# Average LCD between brain regions
# ==================================================================================

def get_LCD_region_pair_average_masks(
        fish_data, 
        regions_inspect_inds, 
        region_inspect_names, 
        min_neurons=5
    ):

    LCD_data_region_avg_masks = {}

    for fish_n in tqdm(fish_data.keys(), leave=False):

        LCD_data_region_avg_masks[fish_n] = {}

        gene_names   = fish_data[fish_n]['gene_data']['gene_names']
        gc_binary    = fish_data[fish_n]['gene_data']['gene_counts_binary']
        region_labels = fish_data[fish_n]['region_data']['region_labels']

        for gene in tqdm(gene_names, leave=False):

            LCD_data_region_avg_masks[fish_n][gene] = {}

            gene_ind = np.where(gene_names == gene)[0][0]
            gene_neuron_inds = np.where(gc_binary[:, gene_ind])[0]

            # Skip if too few expressing neurons
            if len(gene_neuron_inds) < min_neurons:
                continue

            # mapping: neuron_id -> local index in gene_neuron_inds array
            inv_map = {nid: i for i, nid in enumerate(gene_neuron_inds)}

            for i, r_i in enumerate(regions_inspect_inds):

                region_inds_i = np.where(region_labels[:, r_i])[0]
                region_gene_i = np.intersect1d(gene_neuron_inds, region_inds_i)

                if len(region_gene_i) < min_neurons:
                    continue

                for j, r_j in enumerate(regions_inspect_inds):

                    region_inds_j = np.where(region_labels[:, r_j])[0]
                    region_gene_j = np.intersect1d(gene_neuron_inds, region_inds_j)

                    if len(region_gene_j) < min_neurons:
                        continue

                    # Final mask (list of lists):
                    # For each SOURCE neuron in region i,
                    # store all TARGET neurons in region j.
                    mask = []
                    target_idx = np.array([inv_map[nid] for nid in region_gene_j], dtype=np.int32)

                    for src_nid in region_gene_i:
                        mask.append((inv_map[src_nid], target_idx.copy()))

                    LCD_data_region_avg_masks[fish_n][gene][
                        f"{region_inspect_names[i]}-{region_inspect_names[j]}"
                    ] = mask

    return LCD_data_region_avg_masks


def process_gene_region(gene_name, gene_names, LCD_mats_fish, fish_subdata, regions_inspect_inds, use_genes=True, split=False):
    """
    Compute region–region LCD matrices (full and average) for one gene in one fish.

    Parameters
    ----------
    gene_name : str
        Gene to process.
    LCD_mats_fish : dict
        Dictionary of LCD matrices for this fish: LCD_mats_fish[gene_name].
    fish_subdata : dict
        Dictionary with fields: 'region_names_split', 'region_labels_split',
        'gene_names', and 'gene_counts_binary'.
    region_names_inspect_full : list of str
        Base region names (without '_left'/'_right') to include.

    Returns
    -------
    gene_name : str
        The processed gene name.
    gene_result : dict or None
        Dictionary with keys 'full' and 'avg', or None if gene has no expressing neurons.
    """

    # Find neurons expressing this gene
    gene_ind = np.where(gene_names == gene_name)[0][0]
    if use_genes: gene_neuron_inds = np.where(fish_subdata['gene_data']['gene_counts_binary'][:, gene_ind])[0]
    else: gene_neuron_inds = np.where(fish_subdata['subtype_data']['subtype_binary'][:, gene_ind])[0]
    if len(gene_neuron_inds) == 0:
        return gene_name, None

    # Extract region membership and LCD matrix
    if split: gene_region_labels = fish_subdata['region_data']['region_labels_split'][gene_neuron_inds][:, regions_inspect_inds]
    else: gene_region_labels = fish_subdata['region_data']['region_labels'][gene_neuron_inds][:, regions_inspect_inds]
    lcd_gene = LCD_mats_fish[gene_name][..., 0]

    n_regions = gene_region_labels.shape[1]
    gene_result = {
        'full': np.empty((n_regions, n_regions), dtype=object),
        'avg': np.full((n_regions, n_regions), np.nan)
    }

    # Compute LCD per region pair
    for i, j in np.ndindex((n_regions, n_regions)):
        inds_i = gene_region_labels[:, i]
        inds_j = gene_region_labels[:, j]

        if inds_i.sum() > 0 and inds_j.sum() > 0:
            submat = lcd_gene[inds_i][:, inds_j]
            gene_result['full'][i, j] = submat
            gene_result['avg'][i, j] = np.nanmean(submat)

    return gene_name, gene_result


def compute_LCD_region(paradigm, fish_list, LCD_mats, fish_data, regions_inspect_inds, use_genes=True, split=False, n_jobs=None):
    """
    Parallelize LCD region computation across genes per fish for a given paradigm.

    Returns
    -------
    lcd_region : dict
        lcd_region[paradigm][fish_n][gene_name] = {'full', 'avg'}.
    """
    lcd_region = {}

    for fish_n in tqdm(fish_list, desc=f"Processing fish for {paradigm}"):
        fish_subdata = fish_data[fish_n]
        if use_genes: gene_names = fish_subdata['gene_data']['gene_names']
        else: gene_names = fish_subdata['subtype_data']['subtype_names']
        LCD_mats_fish = LCD_mats[paradigm][fish_n]
        
        if n_jobs is None:
            # Run in parallel across genes for this fish
            results = [process_gene_region(gene_name, gene_names, LCD_mats_fish, fish_subdata, regions_inspect_inds, use_genes, split)
                    for gene_name in tqdm(gene_names, total=len(gene_names), leave=False)
                    ]

        else:
            # Run in parallel across genes for this fish
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_gene_region)(gene_name, gene_names, LCD_mats_fish,
                                              fish_subdata, regions_inspect_inds,
                                                use_genes, split)
                for gene_name in gene_names
            )

        # Collect valid results
        lcd_region[fish_n] = {
            gene_name: gene_result
            for gene_name, gene_result in results if gene_result is not None
        }

    return lcd_region



def compute_lcd_spontaneous_vs_visstim_stats(
    LCD_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    mask_type: str = "distance",
    mask: int = 20,
    spont_key: str = "random",
    engaged_key: str = "visrap",
    csv_file: str = "./data/statistical_testing_random_visstim_increase.csv",
) -> Dict[str, Any]:
    """
    Compute LCD statistics comparing spontaneous vs visually stimulated conditions.

    For each gene:
      - Gather LCD difference values (column 0 of 'vals') across fish for
        spontaneous (spont_key) and engaged (engaged_key) paradigms.
      - Restrict to neurons that have finite values in *both* paradigms.
      - Compute medians, SEMs, and per-neuron differences (engaged - spont).
      - Run Wilcoxon signed-rank tests on the per-neuron differences.
      - Include a pooled 'all genes' row.

    Parameters
    ----------
    LCD_data
        Dict with LCD data per paradigm, fish, gene, distance/mask, etc.
    fish_inspect
        Fish IDs to consider.
    mask_type
        Key under observed_stats (e.g. "distance").
    mask
        Mask value under observed_stats[mask_type] (e.g. distance bin).
    spont_key
        Key in LCD_data for spontaneous condition (e.g. "random").
    engaged_key
        Key in LCD_data for visually engaged condition (e.g. "visrap").
    csv_file
        Path passed through to calc_wilcoxon_significances for CSV dumping.

    Returns
    -------
    stats : dict
        Contains:
          - 'gene_names_tot'      : list of genes (without pooled row)
          - 'gene_names_all'      : list of genes + ['all genes']
          - 'engaged_vals'        : array of object arrays (per gene, pooled)
          - 'spont_vals'          : array of object arrays (per gene, pooled)
          - 'diff_vals'           : array of object arrays (per gene, pooled)
          - 'engaged_meds'        : medians (per gene, pooled)
          - 'spont_meds'          : medians (per gene, pooled)
          - 'meds'                : list of (spont, engaged) tuples
          - 'engaged_SEMs'        : SEMs (per gene, pooled)
          - 'spont_SEMs'          : SEMs (per gene, pooled)
          - 'SEMs'                : list of (spont, engaged) SEM tuples
          - 'diff_meds'           : median differences (engaged - spont)
          - 'p_vals'              : raw p-values (Wilcoxon)
          - 'p_vals_bonferroni'   : Bonferroni-corrected p-values
          - 'effect_sizes'        : Cohen's d for diff_vals
          - 'sig_asterisks'       : significance strings ("", "*", "**", "***")
          - 'mask_type', 'mask', 'spont_key', 'engaged_key'
    """
    spont_key = spont_key.lower()
    engaged_key = engaged_key.lower()

    if spont_key not in LCD_data or engaged_key not in LCD_data:
        raise ValueError(f"LCD_data must contain both '{spont_key}' and '{engaged_key}' keys.")

    spont_data = LCD_data[spont_key]
    engaged_data = LCD_data[engaged_key]

    # ------------------------------------------------------------------
    # 1) Determine genes that have data in BOTH paradigms for the given fish
    # ------------------------------------------------------------------
    genes_spont = set()
    genes_engaged = set()

    for f in fish_inspect:
        genes_spont.update(spont_data.get(f, {}).keys())
        genes_engaged.update(engaged_data.get(f, {}).keys())

    gene_names_tot = sorted(genes_spont & genes_engaged)
    if len(gene_names_tot) == 0:
        raise ValueError("No overlapping genes found between spontaneous and visstim paradigms.")

    # ------------------------------------------------------------------
    # 2) Gather per-gene LCD difference values across fish
    # ------------------------------------------------------------------
    engaged_vals_list = []
    spont_vals_list = []

    for gene in gene_names_tot:
        gene_spont_vals = []
        gene_engaged_vals = []

        for fish_n in fish_inspect:
            spont_gene_data = spont_data.get(fish_n, {}).get(gene)
            engaged_gene_data = engaged_data.get(fish_n, {}).get(gene)
            if spont_gene_data is None or engaged_gene_data is None:
                continue

            try:
                spont_vals = np.asarray(
                    spont_gene_data["observed_stats"][mask_type][mask]["vals"],
                    dtype=float,
                )
                engaged_vals = np.asarray(
                    engaged_gene_data["observed_stats"][mask_type][mask]["vals"],
                    dtype=float,
                )
            except KeyError:
                # Missing stats for this mask_type/mask
                continue

            if spont_vals.shape[0] != engaged_vals.shape[0]:
                raise ValueError(
                    f"Neuron count mismatch for gene '{gene}', fish '{fish_n}': "
                    f"{spont_vals.shape[0]} (spont) vs {engaged_vals.shape[0]} (engaged)."
                )

            # Extract difference component (column 0 if Nx3, else scalar/1D)
            if spont_vals.ndim == 2:
                spont_diff = spont_vals[:, 0]
            else:
                spont_diff = spont_vals.astype(float).ravel()

            if engaged_vals.ndim == 2:
                engaged_diff = engaged_vals[:, 0]
            else:
                engaged_diff = engaged_vals.astype(float).ravel()

            # Require finite in both paradigms
            mask_both = np.isfinite(spont_diff) & np.isfinite(engaged_diff)
            if not np.any(mask_both):
                continue

            gene_spont_vals.append(spont_diff[mask_both])
            gene_engaged_vals.append(engaged_diff[mask_both])

        if gene_spont_vals and gene_engaged_vals:
            spont_concat = np.concatenate(gene_spont_vals)
            engaged_concat = np.concatenate(gene_engaged_vals)
        else:
            spont_concat = np.array([], dtype=float)
            engaged_concat = np.array([], dtype=float)

        spont_vals_list.append(spont_concat)
        engaged_vals_list.append(engaged_concat)

    # ------------------------------------------------------------------
    # 3) Build pooled "all genes" row
    # ------------------------------------------------------------------
    spont_vals_nonempty = [v for v in spont_vals_list if v.size > 0]
    engaged_vals_nonempty = [v for v in engaged_vals_list if v.size > 0]

    if spont_vals_nonempty and engaged_vals_nonempty:
        spont_all = np.concatenate(spont_vals_nonempty)
        engaged_all = np.concatenate(engaged_vals_nonempty)
    else:
        spont_all = np.array([], dtype=float)
        engaged_all = np.array([], dtype=float)

    spont_vals = np.array(spont_vals_list + [spont_all], dtype=object)
    engaged_vals = np.array(engaged_vals_list + [engaged_all], dtype=object)

    # ------------------------------------------------------------------
    # 4) Medians, SEMs, diff values
    # ------------------------------------------------------------------
    def _median_and_sem(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return np.nan, np.nan
        x = np.asarray(x, dtype=float)
        med = float(np.nanmedian(x))
        n_eff = np.sum(~np.isnan(x))
        if n_eff <= 1:
            sem = np.nan
        else:
            sem = float(np.nanstd(x) / np.sqrt(n_eff))
        return med, sem

    engaged_meds = []
    spont_meds = []
    engaged_SEMs = []
    spont_SEMs = []
    diff_vals = []

    for sv, ev in zip(spont_vals, engaged_vals):
        sv = np.asarray(sv, dtype=float)
        ev = np.asarray(ev, dtype=float)

        # Ensure same length (they should be by construction)
        if sv.shape != ev.shape:
            n = min(sv.size, ev.size)
            sv = sv[:n]
            ev = ev[:n]

        diff = ev - sv
        diff_vals.append(diff)

        sm, ss = _median_and_sem(sv)
        em, es = _median_and_sem(ev)

        spont_meds.append(sm)
        engaged_meds.append(em)
        spont_SEMs.append(ss)
        engaged_SEMs.append(es)

    engaged_meds = np.array(engaged_meds, dtype=float)
    spont_meds = np.array(spont_meds, dtype=float)
    engaged_SEMs = np.array(engaged_SEMs, dtype=float)
    spont_SEMs = np.array(spont_SEMs, dtype=float)

    meds = [(sp, en) for sp, en in zip(spont_meds, engaged_meds)]
    SEMs = [(sp, en) for sp, en in zip(spont_SEMs, engaged_SEMs)]

    diff_vals = np.array(diff_vals, dtype=object)
    diff_meds = np.array(
        [np.nanmedian(v) if v.size > 0 else np.nan for v in diff_vals],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 5) Statistical tests (Wilcoxon + Bonferroni, Cohen's d)
    # ------------------------------------------------------------------
    gene_names_all = gene_names_tot + ["all genes"]

    # Order for verbose output: sort descending by diff_meds, pooled at the end
    sort_inds = np.flip(np.argsort(diff_meds))
    sort_inds_verbose = np.insert(sort_inds, 0, -1)

    p_vals, p_vals_bonferroni = calc_wilcoxon_significances(
        diff_vals,
        alternative="two-sided",
        genes=gene_names_all,
        verbose=True,
        verbose_ordering=sort_inds_verbose,
        csv_file=csv_file,
        csv_colname="Gene",
    )

    # Significance asterisks for plotting
    _, sig_asterisks = get_sig_asterisk_inds(p_vals_bonferroni)

    # Cohen's d per gene/pooled row
    effect_sizes = []
    for dv in diff_vals:
        dv = np.asarray(dv, dtype=float)
        dv = dv[np.isfinite(dv)]
        if dv.size <= 1:
            effect_sizes.append(np.nan)
        else:
            m = np.nanmean(dv)
            s = np.nanstd(dv, ddof=1)
            effect_sizes.append(m / s if s > 0 else np.nan)
    effect_sizes = np.asarray(effect_sizes, dtype=float)

    stats = {
        "gene_names_tot": gene_names_tot,
        "gene_names_all": gene_names_all,
        "engaged_vals": engaged_vals,
        "spont_vals": spont_vals,
        "diff_vals": diff_vals,
        "engaged_meds": engaged_meds,
        "spont_meds": spont_meds,
        "meds": meds,
        "engaged_SEMs": engaged_SEMs,
        "spont_SEMs": spont_SEMs,
        "SEMs": SEMs,
        "diff_meds": diff_meds,
        "p_vals": p_vals,
        "p_vals_bonferroni": p_vals_bonferroni,
        "sig_asterisks": sig_asterisks,
        "effect_sizes": effect_sizes,
        "mask_type": mask_type,
        "mask": mask,
        "spont_key": spont_key,
        "engaged_key": engaged_key,
    }
    return stats

import numpy as np
from typing import Dict, Any, Sequence, Tuple, Optional
from sklearn.neighbors import KDTree


def compute_neighbor_correlation_vs_radius(
    fish_data: Dict[str, Any],
    fish_inspect: Sequence[str],
    stim_key: str = "visrap",          # now mostly used as a label in the output
    radii: Sequence[float] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
    coords_key: str = "cell_centers_zb",
    traces_key: str = "dff_traces",
    dtype: str = "float32",
    subsample: int = 1
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute average correlation to neighbors and average neighbor count as a
    function of radius, across all neurons (per fish and pooled across fish),
    in a memory-efficient way.

    Key changes vs. previous version
    --------------------------------
    * Correlations are computed **per neuron, per neighbor set** without ever
      materializing the full (P, T) matrix of pairwise traces.
    * By default, correlations are computed on **full dF/F traces** taken from
      fish_data[fish]["neural_data"][traces_key].
      As a fallback, if that is missing, we try the old location:
      fish_data[fish]["stim_response_data"][stim_key][traces_key].

    For each fish:
      - Use 3D coordinates: fish_data[fish]["cell_centers_data"][coords_key]
      - Use full traces (dF/F) as described above (shape: (n_neurons, T))
      - Build KDTree on coords, query neighbors up to max(radii).
      - Z-score traces per neuron (in place) so that correlation(i,j) is
        proportional to dot(z_i, z_j).
      - For each neuron i:
          * Compute correlations to its neighbors within max(radii) once.
          * For each radius r:
              - Determine which neighbors fall within r (using sorted distances).
              - Compute the mean correlation to neighbors for that neuron.
              - Count the number of neighbors for that neuron at radius r.
      - Aggregate across neurons:
          * avg_corr[r] = mean over neurons (with ≥1 neighbor) of their
            per-neuron mean neighbor correlation at radius r.
          * SEM[r]     = SEM across neurons of those per-neuron means.
          * avg_neighbors[r], std_neighbors[r] across all neurons.

    Returns
    -------
    average_correlations_radii : dict
        {
          "per_fish": {
             fish_id: {
                stim_key: {
                   "radii": np.ndarray [R],
                   "avg":   np.ndarray [R],   # mean per radius
                   "SEM":   np.ndarray [R],   # SEM across neurons (per fish)
                },
             },
          },
          "fish_concat": {
             stim_key: {
                "radii": np.ndarray [R],
                "avg":   np.ndarray [R],     # mean across fish of per-fish means
                "SEM":   np.ndarray [R],     # SEM across fish of per-fish means
             }
          }
        }

    neighbors_radii : dict
        {
          "per_fish": {
             fish_id: {
                "radii": np.ndarray [R],
                "avg":   np.ndarray [R],     # mean neighbors per neuron
                "std":   np.ndarray [R],     # std across neurons
             }
          },
          "fish_concat": {
             "radii": np.ndarray [R],
             "avg":   np.ndarray [R],       # mean across fish of per-fish means
             "std":   np.ndarray [R],       # std across fish of per-fish means
          }
        }
    """
    radii = np.asarray(sorted(set(radii)), dtype=float)
    if radii.size == 0:
        raise ValueError("radii must contain at least one radius.")
    max_radius = float(np.max(radii))
    R = radii.size

    avg_corr_per_fish: Dict[str, Dict[str, Any]] = {}
    neigh_per_fish: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Per-fish computation, streaming over neighbors
    # ------------------------------------------------------------------
    for fish_id in fish_inspect:
        fish_entry = fish_data[fish_id]

        # ---- coords ----
        coords_all = np.asarray(
            fish_entry["cell_centers_data"][coords_key],
            dtype=float,
        )

        # ---- traces: prefer full dF/F in neural_data, fall back to stim_response_data ----
        traces = None
        if "functional_data" in fish_entry and traces_key in fish_entry["functional_data"]:
            traces = fish_entry["functional_data"][traces_key]
        elif (
            "stim_response_data" in fish_entry
            and stim_key in fish_entry["stim_response_data"]
            and traces_key in fish_entry["stim_response_data"][stim_key]
        ):
            traces = fish_entry["stim_response_data"][stim_key][traces_key]
        else:
            raise KeyError(
                f"Could not find traces for fish {fish_id}. "
                f"Tried fish_data['neural_data']['{traces_key}'] and "
                f"fish_data['stim_response_data']['{stim_key}']['{traces_key}']."
            )

        traces = np.asarray(traces, dtype=dtype)

        if coords_all.shape[0] != traces.shape[0]:
            raise ValueError(
                f"coords vs traces mismatch for fish {fish_id}: "
                f"{coords_all.shape[0]} coords vs {traces.shape[0]} traces."
            )

        # Drop neurons with NaN coords or NaN traces
        nan_mask = np.isnan(coords_all).any(axis=1) | np.isnan(traces).any(axis=1)
        if np.any(nan_mask):
            coords = coords_all[~nan_mask]
            X = traces[~nan_mask]
        else:
            coords = coords_all
            X = traces

        coords = coords[::subsample]
        X = X[::subsample]

        n_neurons, T = X.shape
        if n_neurons == 0:
            raise ValueError(
                f"No valid neurons for fish {fish_id} after NaN filtering."
            )

        # ---- Z-score per neuron, in-place, ddof=1 ----
        # X: (n_neurons, T)
        X_mean = X.mean(axis=1, keepdims=True)
        X -= X_mean
        denom = np.sum(X**2, axis=1, keepdims=True)
        denom /= max(T - 1, 1)
        std = np.sqrt(denom)
        # avoid division by zero
        std[std == 0] = np.nan
        X /= std

        # ---- KDTree neighbors up to max_radius ----
        tree = KDTree(coords)
        idxs_list, dists_list = tree.query_radius(
            coords,
            r=max_radius,
            return_distance=True,
            sort_results=True,
        )

        # ------------------------------------------------------------------
        # Streaming accumulation of per-neuron statistics for each radius
        # ------------------------------------------------------------------
        # For correlations: we aggregate per-neuron mean neighbor correlation
        #   to get mean and SEM across neurons.
        corr_sum = np.zeros(R, dtype=np.float64)
        corr_sq_sum = np.zeros(R, dtype=np.float64)
        corr_n = np.zeros(R, dtype=np.int64)

        # For neighbor counts: we aggregate neighbors per neuron (including 0)
        neigh_sum = np.zeros(R, dtype=np.float64)
        neigh_sq_sum = np.zeros(R, dtype=np.float64)
        neigh_n = np.full(R, n_neurons, dtype=np.int64)  # every neuron contributes a count

        # loop over neurons
        for i in tqdm(range(n_neurons)):
            idxs = idxs_list[i]
            dists = dists_list[i]

            if idxs.size == 0:
                # no neighbors at all within max_radius; counts will be 0
                continue

            # remove self-neighbor if present
            self_mask = idxs != i
            idxs = idxs[self_mask]
            dists = dists[self_mask]

            n_neighbors_all = idxs.size
            if n_neighbors_all == 0:
                continue

            Xi = X[i]  # (T,)

            # correlations to all neighbors within max_radius
            # shape: (n_neighbors_all,)
            # Corr(i,j) = (z_i * z_j).sum / (T-1)
            X_neighbors = X[idxs]
            corrs_all = (X_neighbors @ Xi) / max(T - 1, 1)

            # For each radius, restrict neighbors and compute per-neuron mean corr
            # and neighbor count
            for r_idx, r in enumerate(radii):
                # since distances are sorted, we can use searchsorted
                m = np.searchsorted(dists, r, side="right")  # neighbors with dist <= r
                n_nbr = int(m) * subsample

                # neighbor count contribution (including zero)
                neigh_sum[r_idx] += n_nbr
                neigh_sq_sum[r_idx] += n_nbr**2

                if n_nbr == 0:
                    # no neighbors at this radius for this neuron -> no corr sample
                    continue

                corrs_r = corrs_all[:m]
                # drop non-finite values if any
                corrs_r = corrs_r[np.isfinite(corrs_r)]
                if corrs_r.size == 0:
                    continue

                mean_corr_i_r = float(np.mean(corrs_r))

                corr_sum[r_idx] += mean_corr_i_r
                corr_sq_sum[r_idx] += mean_corr_i_r**2
                corr_n[r_idx] += 1

        # ------------------------------------------------------------------
        # Convert accumulators to per-fish summary stats
        # ------------------------------------------------------------------
        avg_corr = np.full(R, np.nan, dtype=float)
        sem_corr = np.full(R, np.nan, dtype=float)

        avg_neigh = np.full(R, np.nan, dtype=float)
        std_neigh = np.full(R, np.nan, dtype=float)

        # correlations
        for r_idx in range(R):
            if corr_n[r_idx] > 0:
                mean_r = corr_sum[r_idx] / corr_n[r_idx]
                mean_sq_r = corr_sq_sum[r_idx] / corr_n[r_idx]
                var_r = max(mean_sq_r - mean_r**2, 0.0)
                std_r = np.sqrt(var_r)
                sem_r = std_r / np.sqrt(corr_n[r_idx])

                avg_corr[r_idx] = mean_r
                sem_corr[r_idx] = sem_r

            # neighbors
            mean_n = neigh_sum[r_idx] / float(neigh_n[r_idx])
            mean_sq_n = neigh_sq_sum[r_idx] / float(neigh_n[r_idx])
            var_n = max(mean_sq_n - mean_n**2, 0.0)
            std_n = np.sqrt(var_n)

            avg_neigh[r_idx] = mean_n
            std_neigh[r_idx] = std_n

        avg_corr_per_fish[fish_id] = {
            stim_key: {
                "radii": radii.copy(),
                "avg": avg_corr,
                "SEM": sem_corr,
            }
        }
        neigh_per_fish[fish_id] = {
            "radii": radii.copy(),
            "avg": avg_neigh,
            "std": std_neigh,
        }

    # ------------------------------------------------------------------
    # Pool across fish ("fish_concat" = average over fish)
    # ------------------------------------------------------------------
    fish_ids_valid = list(avg_corr_per_fish.keys())
    if len(fish_ids_valid) == 0:
        raise ValueError("No valid fish found in compute_neighbor_correlation_vs_radius.")

    # correlations: mean & SEM across fish
    all_means_corr = np.stack(
        [avg_corr_per_fish[f][stim_key]["avg"] for f in fish_ids_valid],
        axis=0,
    )  # (F, R)

    pooled_avg_corr = np.nanmean(all_means_corr, axis=0)
    with np.errstate(invalid="ignore"):
        pooled_sem_corr = np.nanstd(all_means_corr, axis=0, ddof=1) / np.sqrt(
            np.sum(~np.isnan(all_means_corr), axis=0).clip(min=1)
        )

    # neighbors: mean & std across fish means
    all_means_neigh = np.stack(
        [neigh_per_fish[f]["avg"] for f in fish_ids_valid],
        axis=0,
    )  # (F, R)

    pooled_avg_neigh = np.nanmean(all_means_neigh, axis=0)
    pooled_std_neigh = np.nanstd(all_means_neigh, axis=0, ddof=1)

    average_correlations_radii = {
        "per_fish": avg_corr_per_fish,
        "fish_concat": {
            stim_key: {
                "radii": radii.copy(),
                "avg": pooled_avg_corr,
                "SEM": pooled_sem_corr,
            }
        },
    }

    neighbors_radii = {
        "per_fish": neigh_per_fish,
        "fish_concat": {
            "radii": radii.copy(),
            "avg": pooled_avg_neigh,
            "std": pooled_std_neigh,
        },
    }

    return average_correlations_radii, neighbors_radii
