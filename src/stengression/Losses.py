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
import warnings
warnings.filterwarnings('ignore')


# 1. Energy score loss.

def energy_score_loss(y_true, y_pred_samples):
    """Calculates the Energy Score loss for multivariate probabilistic forecasting.

    The Energy Score is a proper scoring rule used to evaluate the quality of 
    probabilistic forecasts.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape 
            :math:`(B, t_{pred}, N, D)`.
        y_pred_samples (torch.Tensor): Predicted samples or trajectories of shape 
            :math:`(M, B, t_{pred}, N, D)`, where :math:`M` is the number 
            of stochastic samples.

    Returns:
        torch.Tensor: A scalar representing the mean Energy Score loss across the batch.

    Note:
        The loss is calculated as:
        
        .. math::
            ES(F, y) = E_F[\|X - y\|] - \\frac{1}{2}E_F[\|X - X'\|]

        where :math:`X` and :math:`X'` are independent samples from the 
        forecast distribution :math:`F`.
    """
    M, B, T, N, D = y_pred_samples.shape
    y_pred_samples = y_pred_samples.permute(1,0,2,3,4) # (B, M, T, N, D)
    y_true_expanded = y_true.unsqueeze(1) # (B, 1, T, N, D)

    # First term: E||Y_hat - y||
    # First, the difference (y_pred_samples - y_true_expanded) computes element-wise differences between each predicted sample and the true value.
    # torch.norm(..., p=2, dim=(-3, -2, -1)): Computes the Euclidean (L2) norm of the difference over the last three dimensions (T, N, D). 
    # This gives a norm per batch and sample, measuring how far each predicted sample is from the true sequence in Euclidean distance.
    # .mean(dim=1): Averages this norm over the M samples dimension, resulting in one average distance per batch.
    term1 = torch.norm(y_pred_samples - y_true_expanded, p=2, dim=(-3, -2, -1)).mean(dim=1)

    if M > 1:
        # Second term: 0.5 * E||Y_hat - Y_hat'||
        # Reshape for pairwise distance calculation
        y_pred_flat = y_pred_samples.view(B, M, -1) # -> (B, M, T*N*D)
        
        # Calculate pairwise distances
        dist_matrix = torch.cdist(y_pred_flat, y_pred_flat, p=2)
        term2 = 0.5 * dist_matrix.mean(dim=(1, 2))
        return (term1 - term2).mean()

    return term1.mean()

# 2. Energy score + MSE
def EnergyMSELoss(y_true, y_pred_samples, lambda_mse=0.5):
    """Combines Energy Score and Mean Squared Error (MSE) loss.

    This hybrid loss function regularizes the probabilistic Energy Score with a 
    deterministic MSE term calculated from the sample mean. This is often used 
    to stabilize training and improve the accuracy of the point-estimate.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape 
            :math:`(B, t_{pred}, N, D)`.
        y_pred_samples (torch.Tensor): Predicted samples of shape 
            :math:`(M, B, t_{pred}, N, D)`.
        lambda_mse (float, optional): Weighting factor for the MSE component. 
            Defaults to 0.5.

    Returns:
        torch.Tensor: Combined loss value calculated as:
            :math:`L = |ES(y, \hat{y}) + \lambda_{mse} \cdot MSE(y, \mathbb{E}[\hat{y}])|`.

    Note:
        The MSE component is computed by first averaging the :math:`M` samples 
        to obtain a point forecast.
    """
    eng_loss = energy_score_loss(y_true, y_pred_samples) 
    y_pred_mean = y_pred_samples.mean(dim=0) # (B, T, N, D)
    mse_loss = nn.MSELoss()(y_pred_mean, y_true)
    
    return torch.abs(eng_loss + lambda_mse * mse_loss)
