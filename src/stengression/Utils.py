import numpy as np 
import pandas as pd 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Optional
import torch.optim as optim
import typing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import genpareto
import time
import warnings
warnings.filterwarnings('ignore')

# 1. Spatiotemporal dataset class.
class SpatioTemporalDataset(Dataset):
    """Spatiotemporal dataset for sliding window sequence preparation.

    Prepares lagged input sequences and forecast targets from multivariate 
    time series data. It supports both multi-horizon (sequence-to-sequence) 
    and single-step (sequence-to-point) targets.

    Parameters:
        data (torch.Tensor): Time series data of shape 
            :math:`(T_{observed}, N, D)`, where :math:`N` is the number 
            of nodes and :math:`D` is the feature dimension.
        input_seq_len (int): Number of lagged time steps :math:`(p)` 
            used as model input.
        output_seq_len (int): Number of future time steps :math:`(q)` 
            to predict.
        multi_horizon (bool): If True, returns a sequence of length :math:`(q)`. 
            If False, returns a single step at the :math:`q`-th offset.
    """
    def __init__(self, data, input_seq_len, output_seq_len, multi_horizon=True):
        
        self.data = data
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.multi_horizon = multi_horizon
        
        self.num_samples = data.shape[0] - (input_seq_len + output_seq_len) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx is the index of the input sequence start
        # Input window: idx -> idx + input_seq_len, shape (input_seq_len, num_nodes, num_features)
        x = self.data[idx : idx + self.input_seq_len]
        
        # Output: sequence from end of input sequence to next output_seq_len steps
        if self.multi_horizon:
            y = self.data[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]
        else:
            # If not multi_horizon, implement single step prediction
            y = self.data[idx + self.input_seq_len + self.output_seq_len]  # single step
        
        return x.float(), y.float()



# 3. Compute adjacency matrix from distances matrix.
def compute_adjacency_matrix(route_distances : np.ndarray, sigma2: float, epsilon: float, n=10000):
    """Computes the adjacency matrix from a distance matrix using a Gaussian kernel.

    Following the data preprocessing method from the STGCN (Spatio-Temporal Graph 
    Convolutional Networks) paper, this function applies a thresholded Gaussian 
    kernel to normalize distances and enforce sparsity in the resulting graph.

    Args:
        route_distances (np.ndarray): Distance matrix of shape :math:`(N, N)`, 
            where :math:`N` is the number of nodes (states/stations).
        sigma2 (float): The variance parameter :math:`\sigma^2` for the Gaussian 
            kernel, controlling the width of the neighborhood.
        epsilon (float): The sparsity threshold :math:`\epsilon`. Values below 
            this threshold are set to zero.
        n (int, optional): Scaling factor used to normalize the distances 
            (e.g., maximum distance in the dataset). Defaults to 10000.

    Returns:
        np.ndarray: Adjacency matrix of shape :math:`(N, N)` with zeros on the diagonal (no self-loops).
    """
    n_states = route_distances.shape[0]
    states_distances_norm = route_distances / n
    w2, w_mask = (states_distances_norm * states_distances_norm, np.ones([n_states, n_states]) - np.identity(n_states))

    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


# 4. Prepare spatial weights for SEN.
def prepare_spatial_weights(W, max_lag):
    """Pre-computes the powers of the spatial weight matrix for the STEN model.

    This function generates a sequence of spatial weights matrices 
    :math:`\{W^0, W^1, \dots, W^L\}`. These matrices are used by the 
    STAR-Layer to aggregate information from increasingly distant spatial 
    neighborhoods (lags).

    Args:
        W (torch.Tensor): The base spatial weights matrix 
            of shape :math:`(N, N)`.
        max_lag (int): The maximum spatial lag :math:`(L)` to consider.

    Returns:
        list[torch.Tensor]: A list of :math:`L+1` tensors, where the element at index :math:`l` is the matrix :math:`W` raised to the power of :math:`l`.

    Note:
        The first element (index 0) is always the Identity matrix :math:`I`, 
        representing the "self-loop" or zero-lag spatial component.
    """
    W_list =[torch.eye(W.shape[0])] # W^0 is the identity matrix
    W_pow = W
    for _ in range(max_lag):
        W_list.append(W_pow)
        W_pow = torch.matmul(W_pow, W)
    return W_list


# 5. Plotting helper function. Takes in an ensemble of forecasts and plots the original values and forecasted values along with CIs.
def plot_forecasts(num_nodes, plots_per_row, t_pred, forecast_ensemble, ground_truth, confidence_level=0.95, node_names=None,
                   title='Forecast vs Actual with Confidence Interval', savefig=False, filename=None):
    """Plots forecasted time series against ground truth with prediction intervals.

    This function visualizes the probabilistic performance of the model by plotting
    the median forecast and a shaded confidence area (based on ensemble quantiles)
    against the observed ground truth for multiple nodes in a grid layout.

    Args:
        num_nodes (int): Number of nodes (time series) to plot.
        plots_per_row (int): Number of subplots to display per row in the figure.
        t_pred (int): Number of timesteps in the forecast horizon.
        forecast_ensemble (torch.Tensor): Ensemble of forecast samples of shape 
            :math:`(M, t_{pred}, N)` or :math:`(M, t_{pred}, N, 1)`.
        ground_truth (torch.Tensor): Actual observed values of shape 
            :math:`(t_{pred}, N)`.
        confidence_level (float, optional): Confidence level for the prediction 
            intervals (e.g., 0.95 for 95%). Defaults to 0.95.
        node_names (list of str, optional): Names for each node. If None, 
            nodes are labeled by their index. Defaults to None.
        title (str, optional): Overall title for the figure. 
            Defaults to 'Forecast vs Actual with Confidence Interval'.
        savefig (bool, optional): Whether to save the figure to a file. 
            Defaults to False.
        filename (str, optional): The path/filename to save the figure if 
            `savefig` is True. Defaults to None.

    Note:
        The prediction interval is calculated using the :math:`\\alpha` and 
        :math:`1-\\alpha` quantiles, where 
        :math:`\\alpha = (1 - confidence\_level) / 2`.

    Returns:
        None: Displays a matplotlib figure or saves it to disk.
    """
    
    num_rows = int(np.ceil(num_nodes / plots_per_row))
    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 4 * num_rows), 
                            sharex=True, sharey=False)
    
    if num_nodes == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    timesteps = np.arange(1, t_pred + 1)
    if forecast_ensemble.ndim == 4:
        forecast_ensemble = forecast_ensemble.squeeze(-1)
    
    forecast_median = torch.median(forecast_ensemble, dim=0).values.view((t_pred, num_nodes))
    forecast_median = torch.clamp(torch.round(forecast_median), min=0)
    alpha = (1.0 - confidence_level) / 2.0
    quantiles = torch.tensor([alpha, 1.0 - alpha], device=forecast_ensemble.device)
    bounds = torch.quantile(forecast_ensemble, quantiles, dim=0)

    # bounds will have shape (2, t_pred, num_nodes)
    lower_bound = torch.clamp(bounds[0], min=0)
    upper_bound = torch.clamp(bounds[1], min=0) 
    
    for node in range(num_nodes):
        ax = axs[node]
        # Assigning lines to variables to extract handles later
        line1, = ax.plot(timesteps, ground_truth[:, node].cpu(), marker='o', 
                         label='Actual', markersize=4, color='black')
        line2, = ax.plot(timesteps, forecast_median[:, node].cpu(), marker='+', 
                         label='Forecasted', color='blue')
        fill = ax.fill_between(timesteps,
                               lower_bound[:, node].cpu(),
                               upper_bound[:, node].cpu(),
                               color='gray', alpha=0.3, label='Prediction Interval')
        
        ax.set_title(node_names[node] if node_names else f'Node {node+1}')
        ax.grid(alpha=0.5)

    # Remove unused subplots
    for i in range(num_nodes, len(axs)):
        fig.delaxes(axs[i])
    
    # Global labels
    fig.suptitle(title, fontsize=16)

    # Create a single legend at the bottom center
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
    
    # Adjust layout: we increase the bottom margin (0.07) to fit the new legend
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    if savefig and (filename is not None):
        plt.savefig(filename, format='pdf')
    plt.show()




