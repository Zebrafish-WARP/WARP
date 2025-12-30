
import numpy as np
from pathlib import Path
import os
import glob
import nrrd

from collections import OrderedDict


good_genes = [
    'cart2', 'glyt2', 'tac1', 'pvalb7', 'npb', 'grm1b', 'irx1b', 'dat', 'net', 'calb1', 
    'penka', 'penkb', 'eomesa', 'emx3', 'cfos', 'gad1b', 'cx43', 'vglut2a', 'sst', 'uts1', 
    'pou4f2', 'cort', 'nr4a2a', 'cckb', 'tph2', 'chata', 'calb2a', 'npy', 'gfra1a', 'dmbx1a', 
    'gbx2', 'crhb', 'nefma', 'chodl', 'pyya', 'zic2a', 'th', 'pdyn', 'tbr1b', 'otpa', 'esrrb'
    ]

beh_dict = OrderedDict([('left_swims', 0),
                        ('right_swims', 1),
                        ('forward_swims', 2),
                        ('left_channel', 3),
                        ('right_channel', 4)])

stim_dict_visrap = OrderedDict([('black', 0),
                        ('omr_backward', 1),
                        ('omr_forward', 2),
                        ('omr_right', 3),
                        ('omr_left', 4),
                        ('looming_left', 5),
                        ('looming_right', 6),
                        ('dark_flash', 7),
                        ('light_flash', 8),
                        ('undefined', 9)])


def load_gene_counts(directory_path, filter_genes=[]):
    """
    Loads gene count data from .npy files in a specified directory.

    Args:
        directory_path (str): The path to the directory containing gene count files.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Stacked gene counts for all files.
            - np.ndarray: Array of gene names extracted from filenames.
    """
    
    # Find all gene count files recursively
    count_files = glob.glob(directory_path + '/assign_spots_em/*_counts.npy', recursive=True)

    # Initialize lists to store counts and names
    gene_counts_list = []
    gene_names_list = []

    # Load each file and extract gene name from the filename
    for count_file in count_files:
        gene_name = count_file.split('/')[-1].split('_')[0]
        if gene_name not in filter_genes:
            gene_counts_list.append(np.load(count_file))
            # Extract gene name (assuming format is 'genename_counts.npy')
            gene_names_list.append(gene_name)

    # Stack gene counts into a single array
    gene_counts_stacked = np.stack(gene_counts_list, axis=1)
    gene_names_array = np.array(gene_names_list)

    return gene_counts_stacked, gene_names_array


def threshold_gene_counts(gene_counts, gene_names, fish_n: str, sigma: float = 8.0):
    """
    Computes gene expression thresholds and binarizes the gene expression matrix,
    using custom per-animal logic when needed.

    Args:
        gene_counts (np.ndarray): shape (n_neurons, n_genes).
        gene_names (np.ndarray): shape (n_genes,). Names of the genes.
        fish_n (str): Fish identifier (e.g., '59', '63', '71').
        sigma (float): Sigma for max-deviation thresholding (default: 8.0).

    Returns:
        gene_neuron_list (np.ndarray): List of arrays of neuron indices per gene.
        gene_neuron_thr (np.ndarray): Thresholds per gene.
        gene_counts_binary (np.ndarray): Binarized matrix (neurons × genes).
    """
    from fishspot.filter import maximum_deviation_threshold
    import numpy as np

    n_genes = gene_counts.shape[1]
    gene_neuron_thr = np.empty(n_genes)
    gene_neuron_list = np.empty(n_genes, dtype=object)
    gene_counts_binary = np.zeros_like(gene_counts, dtype=bool)

    # Define gene-specific overrides
    thr1_genes = ['chata', 'glyt2', 'penka', 'cx43', 'cort', 'tph2', 'npy', 'gbx2', 'chodl']
    thr2_genes = ['dat', 'npb', 'emx3', 'th', 'otpa']

    for g_i in range(n_genes):
        gene_expr = gene_counts[:, g_i]
        expressed = gene_expr > 0
        expr_vals = gene_expr[expressed]

        if len(expr_vals) == 0:
            # No expression at all, skip
            gene_neuron_thr[g_i] = np.nan
            gene_neuron_list[g_i] = np.array([], dtype=int)
            continue

        gene_name = gene_names[g_i]

        if fish_n == '59':
            if gene_name in thr1_genes:
                thr = maximum_deviation_threshold(expr_vals)
            elif gene_name in thr2_genes:
                thr = maximum_deviation_threshold(expr_vals)
            elif gene_name == 'net':
                thr = 60
            elif gene_name == 'cfos':
                thr = 5
            else:
                thr = 25

        elif fish_n == '63':
            if gene_name in thr1_genes:
                thr = 2*maximum_deviation_threshold(expr_vals)
            elif gene_name in thr2_genes:
                thr = maximum_deviation_threshold(expr_vals)
            elif gene_name == 'net':
                thr = 60
            elif gene_name == 'cfos':
                thr = 5
            else:
                thr = 25

        elif fish_n == '71':
            if gene_name in thr1_genes:
                thr = 2*maximum_deviation_threshold(expr_vals)
            elif gene_name in thr2_genes:
                thr = maximum_deviation_threshold(expr_vals)
            elif gene_name == 'net':
                thr = 80
            elif gene_name == 'cfos':
                thr = 5
            else:
                thr = 25

        else:
            thr = maximum_deviation_threshold(expr_vals, sigma=sigma)

        gene_neuron_thr[g_i] = thr
        gene_neuron_list[g_i] = np.where(gene_expr > thr)[0]
        gene_counts_binary[gene_neuron_list[g_i], g_i] = True

    return gene_neuron_list, gene_neuron_thr, gene_counts_binary


def load_functional_data(directory_path):
    
    path = Path(directory_path)
    dirs = [d.name for d in path.iterdir() if d.is_dir() and d.name.startswith('EMLmultiFISH')]

    functional_files = {}
    for dir in dirs:
        experiment_name = dir.split('_')[-3].lower()
        functional_files[experiment_name] = np.load(os.path.join(directory_path, dir, 'dff_percentile.npy'))

    return functional_files


def load_and_preprocess_ephys(directory_path):
    from fish.ephys.ephys import load, windowed_variance, estimate_swims
    import glob
    import numpy as np
    from scipy.signal import find_peaks

    ephys_files = glob.glob(directory_path + '/ephys/*11chFlt')

    ephys_data = {}
    for f in ephys_files:
        experiment_name = f.split('/')[-1].split('_')[2].lower()

        ephys_data_experiment = load(f, num_channels=11)

        channel1 = ephys_data_experiment[0]
        channel2 = ephys_data_experiment[1]
        
        swim_signal1 = windowed_variance(channel1)[1]
        swim_signal2 = windowed_variance(channel2)[1]
        
        beh = np.concatenate((swim_signal1[:, None],
                              swim_signal2[:, None]),
                             axis=1).T

        channel_epoch = ephys_data_experiment[6]

        # Find stimulus epochs and downsample
        acquisition_times = ephys_data_experiment[2]
        peaks, _ = find_peaks(acquisition_times, height=3.8, distance=10)
        ephys_times = np.arange(0, acquisition_times.shape[0]) * 1/6000

        channel_epoch_downsampled = np.zeros(len(peaks))
        beh_downsampled = np.zeros((5, len(peaks)))
        time_stamps = np.zeros(len(peaks))
        for i in range(len(peaks)):
            temp = max(channel_epoch[peaks[i]:peaks[i]+(peaks[1]-peaks[0])])
            channel_epoch_downsampled[i] = temp
            beh_downsampled[3:, i] = np.max(beh[:, peaks[i]:peaks[i]+(peaks[1]-peaks[0])], axis=1)
            time_stamps[i] = np.mean(ephys_times[peaks[i]:peaks[i]+(peaks[1]-peaks[0])])

        # Map stimulus indices to full dataset stimulus indices
        stim = np.zeros_like(channel_epoch_downsampled, dtype='int')
        for e_i, e_s in enumerate(channel_epoch_downsampled):
            if experiment_name == 'gainhnl':
                stim[e_i] = 9

            elif experiment_name == 'givingup':
                stim[e_i] = 9

            elif experiment_name == 'random':
                stim[e_i] = 9

            elif experiment_name == 'spahom':
                stim[e_i] = 9

            elif experiment_name == 'visrap':
                if e_s in [0, 1, 2, 3, 4]:
                    stim[e_i] = e_s
                elif e_s in [9, 10, 11, 12]:
                    stim[e_i] = e_s-4

        # Delete first 120 time points to match dff
        beh_downsampled = np.delete(beh_downsampled, slice(0, 120), axis=1)
        stim = np.delete(stim, slice(0,120), axis=0)
        time_stamps = np.delete(time_stamps, slice(0,120), axis=0)
        
        swim_signal, swim_signal_ds, swim_signal_amps, ephys_times, swim_signal_counts_ds, swim_signal_amps_ds, ephys_times_ds = get_swim_events(ephys_data_experiment, start_T=120)

        ephys_data[experiment_name] = {
            'stimulus': stim,
            'behavior': beh_downsampled,
            'swim_signal': swim_signal,
            'swim_signal_downsampled': swim_signal_ds,
            'swim_signal_amps': swim_signal_amps,
            'swim_signal_amps_downsampled': swim_signal_amps_ds,
            'time_stamps': ephys_times, 
            'time_stamps_downsampled': ephys_times_ds}
        
    return ephys_data


def get_swim_events(ephys_data, start_T: int = 120):
    import numpy as np
    from scipy.signal import find_peaks
    from fish.ephys.ephys import load, windowed_variance, estimate_swims
    
    channel1 = ephys_data[0]
    channel2 = ephys_data[1]

    acquisition_times = ephys_data[2]
    peaks, _ = find_peaks(acquisition_times, height=3.8, distance=10)
    ephys_times = np.arange(0, acquisition_times.shape[0]) * 1/6000

    channel_mean = np.mean(np.stack([channel1, channel2]), axis=0)

    fltch1, var_estimate, mean_estimate = windowed_variance(channel1)
    fltch2, var_estimate, mean_estimate = windowed_variance(channel2)

    start1, stop1, thr1 = estimate_swims(fltch1)
    start2, stop2, thr2 = estimate_swims(fltch2)

    fltch, var_estimate, mean_estimate = windowed_variance(channel_mean)
    starts, stops, thr = estimate_swims(fltch)
    start_inds, stop_inds = np.where(starts)[0], np.where(stops)[0]
    starts, stops = ephys_times[start_inds], ephys_times[stop_inds]

    # Create a boolean swim signal trace (1 when the fish is swimming)
    swim_signal = np.zeros(shape=(len(ephys_times)), dtype=bool)
    swim_signal_amps = np.zeros(shape=(len(ephys_times)), dtype=float)
    amps = np.zeros(shape=(len(starts)), dtype=float)
    for i, [start, stop] in enumerate(zip(start_inds, stop_inds)):
        swim_signal[start:stop] = 1
        swim_signal_amps[start:stop] = np.max(fltch[start:stop])
        amps[i] = np.max(fltch[start:stop])
    
    # downsample behavior
    channel_epoch = ephys_data[6]
    channel_epoch_downsampled = np.zeros(len(peaks))

    fltch_ds = np.zeros(len(peaks)-start_T, dtype=float)
    fltch1_ds = np.zeros(len(peaks)-start_T, dtype=float)
    fltch2_ds = np.zeros(len(peaks)-start_T, dtype=float)
    swim_signal_ds = np.zeros(len(peaks)-start_T, dtype=bool)
    swim_signal_counts_ds = np.zeros(len(peaks)-start_T, dtype=int)
    swim_signal_amps_ds = np.zeros(len(peaks)-start_T, dtype=float)
    time_stamps = np.zeros(len(peaks)-start_T)
    for i, p in enumerate(range(start_T, len(peaks))):
        temp = max(channel_epoch[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])
        channel_epoch_downsampled[i] = temp
        fltch_ds[i] = np.mean(fltch[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])
        swim_signal_ds[i] = np.max(swim_signal[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])
        swim_signal_counts_ds[i] = np.sum(np.diff(swim_signal[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])==1)
        swim_signal_amps_ds[i] = np.max(swim_signal_amps[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])
        time_stamps[i] = np.mean(ephys_times[peaks[p]:peaks[p]+(peaks[1]-peaks[0])])
        
    return swim_signal, swim_signal_ds, swim_signal_amps, ephys_times, swim_signal_counts_ds, swim_signal_amps_ds, time_stamps


def get_filter_minimum_one_gene_inds(gene_counts_binary):
    filter_inds = np.where(~np.max(gene_counts_binary, axis=1))[0]
    return filter_inds


def get_filter_in_out_plane_inds(masks_path: str, total_cells: int):

    masks, _ = nrrd.read(masks_path)
    masks = masks.transpose(2,1,0)

    # separate in plane cells from purely projected cells (every 7th plane is functionally imaged)
    in_plane_indices = np.unique(masks[..., ::7])
    if in_plane_indices[..., 0] == 0: in_plane_indices = in_plane_indices[1:]
    out_plane_indices = np.setdiff1d(np.arange(1, total_cells+1), in_plane_indices)

    return in_plane_indices, out_plane_indices


def get_filter_nan_trace_inds(dff_data):
    nan_inds = [np.where(np.isnan(dff).any(axis=1))[0] for exp_name, dff in dff_data.items()]
    filter_inds = np.concatenate(nan_inds)
    return filter_inds


def get_filter_zero_trace_inds(dff_data):
    nan_inds = [np.where(np.all(dff == 0, axis=1))[0] for exp_name, dff in dff_data.items()]
    filter_inds = np.concatenate(nan_inds)
    return filter_inds


def filter_fish_data(fish_data, filter_minimum_one_gene=False, filter_in_out_of_plane=False, filter_nan_traces=False, filter_zero_traces=False, verbose=True):
    """
    Filters fish data based on provided indices.
    """

    fish_data_filtered = {}
    for fish_n in fish_data.keys():

        filter_inds = np.concatenate([fish_data[fish_n]['filter_inds_data'][f_key] for f_key, f_flag in [['minimum_one_gene', filter_minimum_one_gene],
                                                                                                ['out_of_plane', filter_in_out_of_plane],
                                                                                                ['nan_traces', filter_nan_traces], 
                                                                                                ['zero_traces', filter_zero_traces]]
                                                                                                if f_flag ])

        filter_inds = np.unique(filter_inds)
        keep_inds = np.setdiff1d(np.arange(fish_data[fish_n]['gene_data']['gene_counts'].shape[0]), filter_inds)

        fish_data_filtered[fish_n] = {}
        fish_data_filtered[fish_n]['gene_data'] = {
            'gene_counts': fish_data[fish_n]['gene_data']['gene_counts'][keep_inds], 
            'gene_counts_binary': fish_data[fish_n]['gene_data']['gene_counts_binary'][keep_inds],
            'gene_names': fish_data[fish_n]['gene_data']['gene_names']
        }
        fish_data_filtered[fish_n]['cell_centers_data'] = {
            'cell_centers_func': fish_data[fish_n]['cell_centers_data']['cell_centers_func'][keep_inds] if fish_data[fish_n]['cell_centers_data']['cell_centers_func'] is not None else None,
            'cell_centers_exm': fish_data[fish_n]['cell_centers_data']['cell_centers_exm'][keep_inds] if fish_data[fish_n]['cell_centers_data']['cell_centers_exm'] is not None else None,
            'cell_centers_zb': fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][keep_inds]
        }
        fish_data_filtered[fish_n]['region_data'] = {
            'region_labels': fish_data[fish_n]['region_data']['region_labels'][keep_inds],
            'region_names': fish_data[fish_n]['region_data']['region_names']
        }
        fish_data_filtered[fish_n]['functional_data'] = {
            exp_name: dff[keep_inds] for exp_name, dff in fish_data[fish_n]['functional_data'].items()
        }
        fish_data_filtered[fish_n]['ephys_data'] = fish_data[fish_n]['ephys_data']
        fish_data_filtered[fish_n]['stim_response_data'] = {
            exp_name: {k: dff[keep_inds] for k, dff in dictionary.items()} for exp_name, dictionary in fish_data[fish_n]['stim_response_data'].items()
        }

        if verbose: print(f'Fish {fish_n}: Filtered out {len(filter_inds)} neurons; {len(keep_inds)} neurons remain.')

    return fish_data_filtered

def compute_stim_responses_visrap(functional_data, ephys_data):

    from WARP.stimulus_response_utils import calc_stim_responses
    from WARP.stimulus_response_utils import calc_avg_stim_responses
    
    dff = functional_data['visrap']
    stim = ephys_data['visrap']['stimulus']

    dff_z = (dff - np.mean(dff, axis=1, keepdims=True))/np.std(dff, axis=1, keepdims=True)
    stim_responses = calc_stim_responses(dff_z, stim, pre_duration=15, stim_duration=7, post_duration=16)
    avg_stim_responses, sem_stim_responses = calc_avg_stim_responses(stim_responses)

    stim_response_data = {'visrap': {}}
    
    stim_response_data['visrap']['avg_stim_responses'] = avg_stim_responses
    stim_response_data['visrap']['sem_stim_responses'] = sem_stim_responses
    
    stim_response_data['visrap']['avg_stim_responses_cat'] = avg_stim_responses.reshape(avg_stim_responses.shape[0],
                                                avg_stim_responses.shape[1]*avg_stim_responses.shape[2])
    stim_response_data['visrap']['sem_stim_responses_cat'] = sem_stim_responses.reshape(sem_stim_responses.shape[0],
                                                sem_stim_responses.shape[1]*sem_stim_responses.shape[2])

    return stim_response_data


def load_fish_data(fish_data_path: str, fish_n: str, filter_genes=[], verbose=True):

    if verbose: print(f'Loading and preprocessing data for fish F{fish_n}.')

    # Load cell centers
    cell_centers_exm = None # TODO: Load other cell centers
    cell_centers_func = None
    cell_centers_zb = np.load(os.path.join(fish_data_path, 'centroids_aligned_MF' + fish_n + '_masks1st.npy'))

    # Load region labels and names
    region_labels = np.load(os.path.join(fish_data_path, 'region_labels_MF' + fish_n + '_masks1st.npy'))
    region_names = np.load(os.path.join(fish_data_path, 'region_names.npy'), allow_pickle=True)

    # Load and process gene data
    gene_counts, gene_names = load_gene_counts(fish_data_path, filter_genes=filter_genes)
    gene_neuron_list, gene_neuron_thr, gene_counts_binary = threshold_gene_counts(gene_counts, gene_names, fish_n=fish_n, sigma=8)

    # Load functional data for all experiments (Light Sheet)
    functional_data = load_functional_data(fish_data_path)

    # Load behavioral data for all experiments (ephys)
    ephys_data = load_and_preprocess_ephys(fish_data_path)

    stim_responses_visrap = compute_stim_responses_visrap(functional_data, ephys_data)

    # Apply filtering if requested
    filter_inds_data = {'minimum_one_gene': get_filter_minimum_one_gene_inds(gene_counts_binary), 
                        'out_of_plane': get_filter_in_out_plane_inds(os.path.join(fish_data_path, 'masks.nrrd'), gene_counts.shape[0])[1], 
                        'nan_traces': get_filter_nan_trace_inds(functional_data),
                        'zero_traces': get_filter_zero_trace_inds(functional_data)}

    fish_data = {'gene_data': {'gene_counts': gene_counts,
                               'gene_counts_binary': gene_counts_binary,
                               'gene_names': gene_names},
                'cell_centers_data': {'cell_centers_func': cell_centers_func,
                                'cell_centers_exm': cell_centers_exm,
                                'cell_centers_zb': cell_centers_zb},
                'region_data': {'region_labels': region_labels, 
                                'region_names': region_names}, 
                'functional_data': functional_data,
                'ephys_data': ephys_data, 
                'filter_inds_data': filter_inds_data, 
                 'stim_response_data': stim_responses_visrap}

    if verbose: print(f'Successfully loaded and preprocessed data for fish F{fish_n}.')

    return fish_data


def load_WARP_data(data_path: str, fish_list=[59, 63, 71], filter_genes=[]):

    # For each fish, load data
    fish_data = {}
    for fish_n in fish_list:
        fish_data_path = os.path.join(data_path, f'Fish{fish_n}')
        fish_data[fish_n] = load_fish_data(fish_data_path, str(fish_n), filter_genes=filter_genes)

    return fish_data



def combine_fish_data(fish_data, fish_list=[59, 63, 71]):
    """
    Combines per-fish data dictionaries into a single dataset,
    concatenating across the neuron axis where applicable.

    Args:
        fish_data (dict): Dictionary {fish_n: fish_data_dict}.
        fish_list (list): List of fish IDs to include.

    Returns:
        dict: Combined data dictionary matching the same structure as single-fish data.
    """
    combined_fish_data = {}

    # ---------- Combine gene data ----------
    combined_fish_data['gene_data'] = {
        'gene_counts': np.concatenate([fish_data[f]['gene_data']['gene_counts'] for f in fish_list], axis=0),
        'gene_counts_binary': np.concatenate([fish_data[f]['gene_data']['gene_counts_binary'] for f in fish_list], axis=0),
        'gene_names': fish_data[fish_list[0]]['gene_data']['gene_names']  # same across fish
    }

    # ---------- Combine cell centers ----------
    combined_fish_data['cell_centers_data'] = {}
    for key in ['cell_centers_func', 'cell_centers_exm', 'cell_centers_zb']:
        arrs = [fish_data[f]['cell_centers_data'][key] for f in fish_list if fish_data[f]['cell_centers_data'][key] is not None]
        combined_fish_data['cell_centers_data'][key] = np.concatenate(arrs, axis=0) if len(arrs) > 0 else None

    # ---------- Combine region data ----------
    combined_fish_data['region_data'] = {
        'region_labels': np.concatenate([fish_data[f]['region_data']['region_labels'] for f in fish_list], axis=0),
        'region_names': fish_data[fish_list[0]]['region_data']['region_names']
    }

    # ---------- Combine functional data ----------
    exp_names = fish_data[fish_list[0]]['functional_data'].keys()
    combined_fish_data['functional_data'] = {
        exp_name: np.concatenate(
            [fish_data[f]['functional_data'][exp_name] for f in fish_list],
            axis=0
        ) for exp_name in exp_names
    }

    # ---------- Combine ephys data ----------
    # Instead of concatenating across time, we concatenate across experiments (neurons stay separate)
    combined_fish_data['ephys_data'] = {}
    all_ephys = [fish_data[f]['ephys_data'] for f in fish_list]
    for ephys_dict in all_ephys:
        for exp_name, exp_data in ephys_dict.items():
            if exp_name not in combined_fish_data['ephys_data']:
                combined_fish_data['ephys_data'][exp_name] = exp_data
            else:
                # Append new experiment as a new entry with suffix to keep unique
                new_name = f"{exp_name}_F{len(combined_fish_data['ephys_data'])}"
                combined_fish_data['ephys_data'][new_name] = exp_data

    # ---------- Build fish_index_map ----------
    fish_index_map = {}
    start_idx = 0
    for f in fish_list:
        n_cells = fish_data[f]['gene_data']['gene_counts'].shape[0]
        fish_index_map[f] = np.arange(start_idx, start_idx + n_cells)
        start_idx += n_cells

    combined_fish_data['fish_index_map'] = fish_index_map

    fish_data['combined'] = combined_fish_data

    return fish_data




def filter_data_by_indices(indices, dff_files, dff_files_naive, stim_average_files, gene_counts, cell_centers, cell_centers_zb, cell_centers_f, region_labels, remap_inds):
    """
    Filters various data arrays using a boolean index array.
    Internal helper function.
    """
    dff_files_filtered = {k: dff[indices] for k, dff in dff_files.items()}
    dff_files_naive_filtered = {k: dff[indices] for k, dff in dff_files_naive.items()}
    stim_average_files_filtered = {k: dff[indices] for k, dff in stim_average_files.items()}
    gene_counts_filtered = gene_counts[indices]
    cell_centers_filtered = cell_centers[indices]
    cell_centers_zb_filtered = cell_centers_zb[indices]
    cell_centers_f_filtered = cell_centers_f[indices] if cell_centers_f is not None else None
    region_labels_filtered = region_labels[indices]
    remap_inds_filtered = remap_inds[indices]
    

    return (dff_files_filtered, dff_files_naive_filtered, stim_average_files_filtered, gene_counts_filtered,
            cell_centers_filtered, cell_centers_zb_filtered, cell_centers_f_filtered,
            region_labels_filtered, remap_inds_filtered)



def identify_gene_types(gene_counts_binary, use_all_neurons: bool = True, neuron_thr: int = 0):
    """
    From a binarized gene count matrix, identify distinct gene types and return their indices. 
    Additionally generate and return gene type names associated with each gene type.

    Args:
        gene_counts (np.ndarray):
            Array of shape (n_neurons, n_genes) containing .
        
        use_all_neurons (bool, default True):
            Boolean flag indicating whether to use all neurons for each gene type (neurons can then be part
            of multiple distinct gene types)
            
        neuron_thr (int, default 0):
            Threshold on the number of neurons that need to be part of a distinct gene type

    Returns:
        cluster_neuron_inds (np.ndarray): 
    """
    
    if use_all_neurons:
        unique_clusters = np.unique(gene_counts_binary[neuron_inds], axis=0)[1:]

        cluster_neuron_inds = np.empty(len(unique_clusters), dtype=object)
        counts = np.zeros(len(unique_clusters), dtype=int)
        for i, u in enumerate(unique_clusters):
            gene_inds = np.where(u)[0]
            neuron_inds_clust = np.where(np.sum(gene_counts_binary[neuron_inds][:, gene_inds], axis=1) == len(gene_inds))[0]
            cluster_neuron_inds[i] = neuron_inds_clust
            counts[i] = len(neuron_inds_clust)

    else:
        unique_clusters, counts = np.unique(gene_counts_binary, axis=0, return_counts=True)
        unique_clusters = unique_clusters[1:]
        counts = counts[1:]

        cluster_neuron_inds = np.array([np.where(np.all(gene_counts_binary == u, axis=1))[0] for u in unique_clusters], dtype=object)

    print('Number of unique gene types: {}'.format(len(unique_clusters)))

    # Remove any clusters that have a lower neuron count than the user-defined threshold
    cluster_remove_inds = np.where(counts < neuron_thr)
    unique_clusters = np.delete(unique_clusters, cluster_remove_inds, axis=0)
    counts = np.delete(counts, cluster_remove_inds)
    cluster_neuron_inds = np.delete(cluster_neuron_inds, cluster_remove_inds)

    print('Number of unique gene types after thresholding: {}'.format(len(unique_clusters)))

    # Give each cluster a name
    cluster_names = ['c_' + '_'.join(gene_names[np.where(c)[0]]) for c in unique_clusters]
    
    return cluster_neuron_inds, cluster_names, counts


def get_unique_types(fish_data, fish_list, min_neighbor_thr: int = 2, count_thr: int = 5, dist_thr: float = None, ignore_genes_list: list = None):

    import numpy as np
    from scipy.spatial import KDTree
    import itertools
    import copy
    
    # Remove neurons expressing a specific gene (if applicable)
    gene_counts_binary_dict = {fish_n: copy.deepcopy(fish_data[fish_n]['gene_data']['gene_counts_binary']) for fish_n in fish_list}   
    for fish_n in fish_list:
        gene_names = fish_data[fish_n]['gene_data']['gene_names']
        for gene_ignore in ignore_genes_list:
            gene_counts_binary_dict[fish_n] = np.delete(gene_counts_binary_dict[fish_n], np.where(gene_names == gene_ignore)[0][0], axis=1)

    # Find unique gene combinations and filter for counts based on threshold
    # Here, count thresholds are applied per fish individually
    # Additionally throw out completely empty set
    unique_types_data = [np.unique(gene_counts_binary_dict[fish_n], axis=0, return_counts=True) for fish_n in fish_list]

    unique_types = [unique_types_data[fish_i][0][unique_types_data[fish_i][1] > count_thr] for fish_i in range(len(fish_list))]
    unique_types = [u[~(u == [False]*u.shape[1]).all(axis=1)] for u in unique_types]

    # find intersect between all animals
    # We do this by finding unique entries in a concatenated unique types array and finding all entries who have as many counts
    # as there are separate datasets (numbers of fish)
    unique_types_conc = np.concatenate(unique_types, axis=0)
    unique_types_full, counts_full = np.unique(unique_types_conc, axis=0, return_counts=True)
    unique_types_full = unique_types_full[counts_full==len(fish_list)]

    # If a distance threshold is given, filter types based on distances between animals
    # We require each type to be present across all animals within a given radius
    # To this end, we search for all neuron-neuron combinations of each unique type and filter
    # For each type to at least be present as many times as the count threshold
    combs = list(itertools.combinations(np.arange(len(fish_list)), r=2))
    if dist_thr is not None:
        unique_type_threshold_pass = [True] * len(unique_types_full)
        midpoints = [[0, np.nanmean(fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][:, 1]), 0] for fish_n in fish_list]
        for i, unique_type in enumerate(unique_types_full):
            neuron_inds = [np.where((gene_counts_binary_dict[fish_n] == unique_type).all(axis=1))[0]
                           for fish_n in fish_list]
            kdtrees = [KDTree(np.abs(fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][inds][~np.isnan(fish_data[fish_n]['cell_centers_data']['cell_centers_zb'][inds][:, 0])]))
                       for fish_n, inds, midpoint in zip(fish_list, neuron_inds, midpoints)]

            for comb in combs:
                inds_c0 = kdtrees[comb[0]].query_ball_tree(kdtrees[comb[1]], r=dist_thr)
                inds_c1 = kdtrees[comb[1]].query_ball_tree(kdtrees[comb[0]], r=dist_thr)
                
                lengths_set_c0 = np.array([len(n) for n in inds_c0])
                lengths_set_c1 = np.array([len(n) for n in inds_c1])
                
                number_set_pass_c0 = np.where(lengths_set_c0 >= min_neighbor_thr)[0]
                number_set_pass_c1 = np.where(lengths_set_c1 >= min_neighbor_thr)[0]
                
                if len(number_set_pass_c0) < count_thr or len(number_set_pass_c1) < count_thr:
                    unique_type_threshold_pass[i] = False

        unique_types_full = unique_types_full[unique_type_threshold_pass]
    return unique_types_full


def assign_subtype_names(unique_types, gene_names, ignore_genes_list=[]):
    import numpy as np

    gene_names = np.array([g for g in gene_names if g not in ignore_genes_list])
    
    cluster_names = ['_'.join(gene_names[np.where(c)]) for c in unique_types]
    return cluster_names


def update_fish_data_with_subtypes(fish_data, unique_types, subtype_names, ignore_genes_list: list = None):

    for fish_n in fish_data.keys():

        # Remove neurons expressing a specific gene (if applicable)
        gene_counts_binary = np.copy(fish_data[fish_n]['gene_data']['gene_counts_binary'])
        gene_names = fish_data[fish_n]['gene_data']['gene_names']
        gene_counts_binary = np.delete(gene_counts_binary, [np.where(gene_names == gene_ignore)[0][0] for gene_ignore in ignore_genes_list], axis=1)

        fish_data[fish_n]['subtype_data'] = {}
        fish_data[fish_n]['subtype_data']['subtype_binary'] = np.stack([(gene_counts_binary == unique_type).all(axis=1) for unique_type in unique_types]).T
        fish_data[fish_n]['subtype_data']['subtype_names'] = np.array(subtype_names)

    return fish_data

def add_subtypes_to_fish_data(fish_data, ignore_genes_list=['cfos']):

    fish_inspect = list(fish_data.keys())
    if 'combined' in fish_inspect:
        fish_inspect.remove('combined')

    # Assign unique types without cfos
    unique_types_full = get_unique_types(fish_data, fish_inspect, min_neighbor_thr = 2, count_thr = 5, dist_thr= 40, ignore_genes_list=ignore_genes_list)

    # Sort gene names based on number of neurons across all fish
    sort_inds = np.argsort(fish_data[fish_inspect[0]]['gene_data']['gene_counts_binary'].sum(0))
    gene_names_sort = fish_data[fish_inspect[0]]['gene_data']['gene_names'][sort_inds]

    # Assign subtype names
    subtype_names = assign_subtype_names(unique_types_full, gene_names_sort, ignore_genes_list=ignore_genes_list)

    # Update fish data dictionary with subtype information
    fish_data = update_fish_data_with_subtypes(fish_data, unique_types_full, subtype_names, ignore_genes_list=ignore_genes_list)
    
    return fish_data
    