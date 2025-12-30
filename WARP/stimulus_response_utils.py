"""
This file contains functions related to stimulus response analyses for the RAVERFISH project

Author:
Luuk W. Hesselink (luuk.hesselink@donders.ru.nl)
"""

def get_stim_times(stim, stim_i, pre_duration: int = 5, stim_duration: int = 7, post_duration: int = 10):
    """
    Identifies the time points where a specific stimulus is presented in a stimulus trace.

    Args:
        stim (np.ndarray): Array of shape (n_timepoints,) 
            containing the stimulus identity at each time point.
            
        stim_i (int): 
            The specific stimulus ID to be identified in the time series.
        
        pre_duration (int, default: 5):
            Duration of the pre-stimulus window (in time points).
            
        stim_duration (int, default: 7):
            Duration of the stimulus ON period (in time points).
            
        post_duration (int, default: 10):
            Duration of the post-stimulus window (in time points).

    Returns:
        np.ndarray: Array containing the indices of time points where the specified stimulus is presented.
    """
    
    import numpy as np

    # Identify time steps where the stimulus is active
    stim_times = stim == stim_i

    # Find transitions from inactive to active state
    stim_time_onsets = np.where(np.diff(stim_times.astype(int)) > 0)[0]
    
    # Apply time window constraints to remove invalid presentations
    stim_time_onsets = stim_time_onsets[(stim_time_onsets - pre_duration >= 0) & (stim_time_onsets + stim_duration + post_duration < stim.shape[0])]   
 
    # Get stim times array
    stim_times = np.concatenate([np.arange(stim_time-pre_duration, stim_time+stim_duration+post_duration) for stim_time in stim_time_onsets])
   

    return stim_times, stim_time_onsets


def calc_stim_responses(dff, stim, pre_duration: int = 5, stim_duration: int = 7, post_duration: int = 10):
    """
    Calculates the stimulus response for each unique stimulus presentation in a neural population.

    Args:
        dff (np.ndarray): Array of shape (n_neurons, n_timepoints)
            containing the dF/F signal for each neuron over time.
            
        stim (np.ndarray): Array of shape (n_timepoints,) 
            containing the stimulus identity at each time point.
            
        pre_duration (int, default: 5):
            Duration of the pre-stimulus window (in time points).
            
        stim_duration (int, default: 7): 
            Duration of the stimulus ON period (in time points). Defaults to 7.
            
        post_duration (int, default: 10): 
            Duration of the post-stimulus window (in time points).

    Returns:
        np.ndarray: Array of shape (n_neurons, n_unique_stimuli, n_stim_presentations, total_window_duration)
            contains the stimulus responses for each neuron, for each unique stimulus, for each presentation.
    """
    
    import numpy as np

    # Identify unique stimuli and their corresponding time points
    unique_stimuli = np.unique(stim)[1:]
    stim_time_onsets = {s: get_stim_times(stim, s, pre_duration, stim_duration, post_duration)[1] for s in unique_stimuli}

    # Find the maximum number of stimulus presentations across all stimuli
    max_stim_times = max([len(s) for s in stim_time_onsets.values()])

    # Initialize array to store the stimulus responses
    stim_responses = np.full(shape=(dff.shape[0],
                                    len(unique_stimuli),
                                    max_stim_times,
                                    pre_duration + stim_duration + post_duration),
                             fill_value=np.nan)

    # Extract stimulus responses for each unique stimulus and each presentation
    for i, (stim_i, stim_times_i) in enumerate(stim_time_onsets.items()):
        for j, stim_time_j in enumerate(stim_times_i):
            stim_responses[:, i, j, :] = dff[:, stim_time_j - pre_duration:stim_time_j + stim_duration + post_duration]

    return stim_responses


def calc_avg_stim_responses(stim_responses):
    """
    Calculates the average stimulus response and standard error of the mean (SEM) for each unique stimulus for all neurons.

    Args:
        stim_responses (np.ndarray): Array of shape (n_neurons, n_unique_stimuli, n_stim_presentations, total_window_duration)
            contains the stimulus responses for each neuron, for each unique stimulus, for each presentation.

    Returns:
        avg_stim_responses (np.ndarray): Array of shape (n_neurons, n_unique_stimuli, total_window_duration)
            contains the average stimulus response for each neuron, for each unique stimulus.
            
        sem_stim_responses (np.ndarray): Array of shape (n_neurons, n_unique_stimuli, total_window_duration)
            contains the standard error of the mean for each neuron, for each unique stimulus.
    """
    
    import numpy as np

    avg_stim_responses = np.nanmean(stim_responses, axis=2)
    sem_stim_responses = np.nanstd(stim_responses, axis=2) / np.sqrt(np.sum(~np.isnan(stim_responses), axis=2))

    return avg_stim_responses, sem_stim_responses


def get_gene_stim_responses(gene_counts_binary, stim_responses_avg, min_neurons: int = 0):
    """
    Extracts the stimulus responses for neurons expressing a specific gene.

    Args:
        gene_counts_binary (np.ndarray): Array of shape (n_neurons, n_genes)
            containing binary values indicating whether a neuron expresses a specific gene (1) or not (0).
        stim_responses (np.ndarray): Array of shape (n_neurons, n_stims, total_window_duration)
            containing the stimulus responses for each neuron and stimulus.
        min_neurons (int, optional): Minimum number of neurons required to be expressing a gene
            for its stimulus responses to be included. Defaults to 0.

    Returns:
        gene_neuron_responses (dict): Dictionary containing the average stimulus responses
            for each gene that has at least `min_neurons` expressing it. Keys are gene IDs and values are
            arrays of shape (n_stims, total_window_duration).
    """
    
    import numpy as np

    # Check for matching dimensions between gene counts and stim responses
    n_neurons, n_genes = gene_counts_binary.shape
    n_stims = stim_responses_avg.shape[1]
    
    if n_neurons != stim_responses_avg.shape[0]:
        raise ValueError('Number of neurons in gene counts matrix does not correspond to number of neurons in stim responses')

    # Initialize dictionary to store gene-specific neuron responses
    gene_neuron_responses = {}

    # Iterate through each gene
    for g_i in range(n_genes):

        # Get indices of neurons expressing the current gene
        neuron_inds_gene = np.where(gene_counts_binary[:, g_i])[0]

        # Check if enough neurons express the gene to proceed
        if len(neuron_inds_gene) > min_neurons:

            # Extract stimulus responses for neurons expressing the gene
            gene_neuron_responses[g_i] = stim_responses_avg[neuron_inds_gene]
            
        else:
            gene_neuron_responses[g_i] = np.full_like(stim_responses_avg[neuron_inds_gene], fill_value=np.nan)

    return gene_neuron_responses


def calc_avg_gene_neuron_responses(gene_neuron_responses):
    """
    Calculates the average stimulus response and standard error of the mean (SEM)
    for each gene with at least one responding neuron.

    Args:
        gene_neuron_responses (dict): Dictionary containing the stimulus responses
            for each gene that has at least one responding neuron. Keys are gene IDs and values are
            arrays of shape (n_stims, total_window_duration).

    Returns:
        gene_neuron_responses_avg (np.ndarray): Array of shape (n_genes, n_stims, total_window_duration)
            contains the average stimulus response for each gene.
        gene_neuron_responses_SEM (np.ndarray): Array of shape (n_genes, n_stims, total_window_duration)
            contains the standard error of the mean for each gene.
    """
    
    import numpy as np

    # Get dimensions of response data
    _, n_stims, total_window_duration = list(gene_neuron_responses.values())[0].shape
    n_genes = len(gene_neuron_responses)

    # Initialize arrays to store average and SEM responses
    gene_neuron_responses_avg = np.empty((n_genes, n_stims, total_window_duration))
    gene_neuron_responses_SEM = np.empty((n_genes, n_stims, total_window_duration))

    # Iterate through each gene and calculate average and SEM
    for i, (gene_id, responses) in enumerate(gene_neuron_responses.items()):
        gene_neuron_responses_avg[i] = np.nanmean(responses, axis=0)
        gene_neuron_responses_SEM[i] = np.nanstd(responses, axis=0) / np.sqrt(np.sum(~np.isnan(responses), axis=0))

    return gene_neuron_responses_avg, gene_neuron_responses_SEM

