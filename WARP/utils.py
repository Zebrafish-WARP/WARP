import numpy as np
from scipy.spatial import KDTree

def calc_neighbor_list(cell_centers, dist_thr_inner: float = 0., dist_thr_outer: float = 20., spacing=None):
    """
    Calculates a list of neighbors for each cell center based on a threshold distance.

    Args:
        cell_centers (np.ndarray): Array of shape (n_cells, n_dimensions)
            contains the coordinates of each cell center.
            
        dist_thr (float):
            Threshold distance in physical units (generally µm) for determining closeness between cells.
            
        spacing (np.ndarray, default: None): Array of shape (n_dimensions,) 
            containing the scaling factors in physical units (generally µm) for each dimension.

    Returns:
        neighbor_list (list): 
            List containing the indices of neighbor cells for each cell in the dataset.
    """

    if dist_thr_outer <= 0:
        raise ValueError(f"Provided threshold distance ({dist_thr}) cannot be negative.")

    if spacing is not None:
        spacing = np.array(spacing)
        cell_centers = cell_centers * spacing

    # Create KDTree
    kdtree = KDTree(cell_centers)

    # Query neighbors within the threshold distance
    neighbor_list_outer = kdtree.query_ball_tree(kdtree, dist_thr_outer)
    # Remove self-references from the neighbor list
    neighbor_list_outer = [np.delete(neighbors, np.where(np.array(neighbors) == i)[0]) for i, neighbors in enumerate(neighbor_list_outer)]
    
    if dist_thr_inner > 0:
        neighbor_list_inner = kdtree.query_ball_tree(kdtree, dist_thr_inner)
        neighbor_list_inner = [np.delete(neighbors, np.where(np.array(neighbors) == i)[0]) for i, neighbors in enumerate(neighbor_list_inner)]
        
        neighbor_list = [np.setdiff1d(neighbors_outer, neighbors_inner) for neighbors_inner, neighbors_outer in zip(neighbor_list_inner, neighbor_list_outer)]
    else:
        neighbor_list = neighbor_list_outer
        
    return neighbor_list


def symmetrical_project(coords, coords_all=None, proj_axis=1):
    """
    Reflects points across the mean along a given axis to enforce symmetry.

    Parameters:
    - coords_proj: (N, D) array of coordinates to reflect (input is not modified).
    - coords_all:  Optional (N, D) array used to compute the reflection mean. 
                   If None, defaults to coords_proj.
    - proj_axis:   Axis (int) along which to compute the reflection mean.

    Returns:
    - coords_sym: New array with symmetrical projection applied.
    """
    if coords_all is None:
        coords_all = coords

    coords_sym = coords.copy()
    proj_mean = np.nanmean(coords_all[:, proj_axis])
    coords_sym[:, proj_axis] = np.abs(coords_sym[:, proj_axis] - proj_mean)
    return coords_sym


def KLDiv_discrete(P: np.ndarray, Q: np.ndarray, binsize: float = 50.0) -> float:
    """
    KL(P || Q) for 3D point clouds P and Q using a simple 3D histogram discretization.

    P, Q : (n_points, 3) arrays of coordinates (after any projection).
           Q is the 'reference' distribution (e.g., all cells).
    binsize : bin width in each dimension (same for x, y, z).

    Returns
    -------
    KL divergence (float). np.nan if P or Q is too small or degenerate.
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)

    if P.ndim != 2 or P.shape[1] != 3 or Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError("P and Q must be (n_points, 3) arrays.")

    # Need at least a few points to define a meaningful distribution
    if P.shape[0] < 5 or Q.shape[0] < 5:
        return np.nan

    # Define bins on the support of P (cluster)
    maxP = np.max(P, axis=0)
    x_bins = np.append(np.arange(0, maxP[0], binsize), maxP[0])
    y_bins = np.append(np.arange(0, maxP[1], binsize), maxP[1])
    z_bins = np.append(np.arange(0, maxP[2], binsize), maxP[2])

    if len(x_bins) < 2 or len(y_bins) < 2 or len(z_bins) < 2:
        return np.nan

    KL = 0.0
    nP = len(P)
    nQ = len(Q)

    for i in range(len(x_bins) - 1):
        x_lo, x_hi = x_bins[i], x_bins[i + 1]
        for j in range(len(y_bins) - 1):
            y_lo, y_hi = y_bins[j], y_bins[j + 1]
            for k in range(len(z_bins) - 1):
                z_lo, z_hi = z_bins[k], z_bins[k + 1]

                mask_Q = (
                    (Q[:, 0] >= x_lo) & (Q[:, 0] < x_hi) &
                    (Q[:, 1] >= y_lo) & (Q[:, 1] < y_hi) &
                    (Q[:, 2] >= z_lo) & (Q[:, 2] < z_hi)
                )
                mask_P = (
                    (P[:, 0] >= x_lo) & (P[:, 0] < x_hi) &
                    (P[:, 1] >= y_lo) & (P[:, 1] < y_hi) &
                    (P[:, 2] >= z_lo) & (P[:, 2] < z_hi)
                )

                Qx = np.sum(mask_Q) / nQ
                Px = np.sum(mask_P) / nP

                # Avoid log(0) and division by zero
                if Px == 0 or Qx == 0:
                    continue

                KL += Px * np.log(Px / Qx)

    return KL