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
import warnings
warnings.filterwarnings('ignore')

# 1. Graph Convolution. PyTorch implementation of the GC used in E-STGCN and STGCN.
# For details, refer to https://github.com/mad-stat/E_STGCN/blob/main/Codes/E_STGCN_Code_Share.ipynb

class GraphInfo:
    """A class to hold graph structure information."""
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes

class GraphConv(nn.Module):
    """
    Graph convolution layer for learning node representations in a graph.

    Args:
        in_feat (int): Input feature dimension per node.
        out_feat (int): Output feature dimension per node after convolution.
        graph_info (:class:`~stengression.Models.GraphInfo`): Graph structure 
            metadata containing edges and node counts.
        gcn_seed (int, optional): Random seed for weight initialization. 
            Defaults to 21.
        aggregation_type (str, optional): How to aggregate neighbor messages. 
            Options: 'mean', 'sum', 'max'. Defaults to 'mean'.
        combination_type (str, optional): How to combine node features and 
            aggregated messages. Options: 'concat', 'add'. Defaults to 'concat'.
        activation (str, optional): Name of the activation function from 
            ``torch.nn.functional`` (e.g., 'relu'). If None, no activation 
            is applied. Defaults to None.

    Note:
        This layer computes spatial graph convolutions by aggregating neighbor 
        features, combining with transformed node features, and applying 
        a non-linear activation. The layer performs a message-passing operation:
        
        1. **Aggregate**: Collects features from neighbors defined in `graph_info`.
        2. **Combine**: Merges aggregated features with the node's own features.
        3. **Activate**: Applies the specified non-linear function.

    Shapes:
        - **Input**: :math:`(N, B, T, D_{in})` where :math:`N` is num_nodes, 
          :math:`B` is batch_size, :math:`T` is sequence_length, and 
          :math:`D_{in}` is in_feat.
        - **Output**: :math:`(N, B, T, D_{out})` where :math:`D_{out}` is out_feat.
    """
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        graph_info,
        gcn_seed: int = 21,
        aggregation_type: str = "mean",
        combination_type: str = "concat",
        activation: Optional[str] = None,
    ):
        super(GraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.gcn_seed = gcn_seed
        
        # weight parameter
        self.weight = nn.Parameter(
            torch.empty(in_feat, out_feat)
        )
        torch.manual_seed(self.gcn_seed)
        nn.init.xavier_uniform_(self.weight)  # Xavier Glorot initialization

        # Activation
        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = lambda x: x

    def aggregate(self, neighbour_representations: torch.Tensor):
        src_nodes = torch.tensor(self.graph_info.edges[0], device=neighbour_representations.device)  # shape: [num_edges]
        dst_nodes = torch.tensor(self.graph_info.edges[1], device=neighbour_representations.device)  # shape: [num_edges]

        num_nodes = self.graph_info.num_nodes

        if self.aggregation_type == "sum":
            aggregated = torch.zeros(
                (num_nodes,) + neighbour_representations.shape[1:], 
                device=neighbour_representations.device
            )
            aggregated.index_add_(0, src_nodes, neighbour_representations)

        elif self.aggregation_type == "mean":
            aggregated = torch.zeros(
                (num_nodes,) + neighbour_representations.shape[1:], 
                device=neighbour_representations.device
            )
            counts = torch.zeros(num_nodes, device=neighbour_representations.device).float()
            aggregated.index_add_(0, src_nodes, neighbour_representations)
            counts.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float))
            counts = counts.clamp(min=1).view(-1, *([1]*(neighbour_representations.dim()-1)))
            aggregated = aggregated / counts

        elif self.aggregation_type == "max":
            # Group-wise max per node
            aggregated = torch.full(
                (num_nodes,) + neighbour_representations.shape[1:], 
                float('-inf'), device=neighbour_representations.device
            )
            for i in range(src_nodes.shape[0]):
                aggregated[src_nodes[i]] = torch.max(
                    aggregated[src_nodes[i]], neighbour_representations[i]
                )

        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

        return aggregated

    def compute_nodes_representation(self, features: torch.Tensor):
        """
        Compute node representations via a linear projection on the last dimension.

        This method performs a matrix multiplication between the input features 
        and the layer's internal weight matrix.

        Args:
            features (torch.Tensor): Input feature tensor of shape 
                :math:`(N, B, T, D_{in})`, where :math:`N` is the number 
                of nodes, :math:`B` is batch size, :math:`T` is sequence 
                length, and :math:`D_{in}` is the input feature dimension.

        Returns:
            torch.Tensor: The projected representations of shape 
                :math:`(N, B, T, D_{out})`, where :math:`D_{out}` is the 
                output feature dimension.
        """
        return torch.matmul(features, self.weight)  # last-dim matmul

    def compute_aggregated_messages(self, features: torch.Tensor):
        dst_nodes = self.graph_info.edges[1]
        # gather neighbors
        neighbour_representations = features[dst_nodes]  
        aggregated_messages = self.aggregate(neighbour_representations)
        return torch.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: torch.Tensor, aggregated_messages: torch.Tensor):
        if self.combination_type == "concat":
            h = torch.cat([nodes_representation, aggregated_messages], dim=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")
        return self.activation(h)

    def forward(self, features: torch.Tensor):
        """
        Perform the graph convolution forward pass.

        This method implements the three-step message passing process:
        1. **Projection**: Computes node-wise representations.
        2. **Aggregation**: Gathers messages from neighboring nodes.
        3. **Update**: Combines local and neighborhood information.

        Args:
            features (torch.Tensor): Input spatiotemporal features of shape
                :math:`(N, B, T, D_{in})`, where :math:`N` is number of nodes,
                :math:`B` is batch size, :math:`T` is sequence length, and
                :math:`D_{in}` is input feature dimension.

        Returns:
            torch.Tensor: The updated node representations of shape :math:`(N, B, T, D_{out})`, where :math:`D_{out}` is the output feature dimension.
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


# 2. Graph Engression Network (GEN) Model with LSTM as temporal module.
class GCEN(nn.Module):
    """
    Graph Convolutional Engression Network (GCEN) for probabilistic spatiotemporal forecasting.

    This model integrates a :class:`GraphConv` spatial module with an LSTM-based 
    temporal module to capture both spatial dependencies and temporal dynamics. 
    It follows the engression principle, using noise injection to generate 
    stochastic forecast samples.

    Args:
        in_feat_dim (int): Input feature dimension per node.
        gcn_out_feat (int): Output feature dimension from the GCN spatial module.
        lstm_hidden_dim (int): Hidden dimension size of the LSTM temporal module.
        lstm_num_layers (int): Number of layers in the LSTM.
        lstm_dropout (float): Dropout probability in the LSTM.
        p_lag (int): Number of past timesteps used as input (lookback window).
        t_pred (int): Number of future timesteps predicted (forecast horizon).
        graph_info (GraphInfo): An instance of :class:`~stengression.Models.GraphInfo` 
            containing graph structure.
        noise_encode (str, optional): Method for noise injection: ``'add'`` or 
            ``'concat'``. Defaults to ``'add'``.
        noise_dist (str, optional): Distribution for noise: ``'gaussian'`` or 
            ``'uniform'``. Defaults to ``'gaussian'``.
        noise_dim (int, optional): Dimension of noise features if concatenated. 
            Defaults to 2.
        noise_std (float, optional): Scaling factor for the noise. Defaults to 1.
        graph_conv_params (dict, optional): Dictionary of params for 
            :class:`GraphConv`. Defaults to None.
        gcn_seed (int, optional): Seed for GCN weight initialization. Defaults to 21.
        temporal_seed (int, optional): Seed for LSTM initialization. Defaults to 9.

    Attributes:
        gcn (GraphConv): The spatial processing module.
        lstm (nn.LSTM): The temporal processing module.
        output_layer (nn.Linear): Fully connected layer mapping hidden state to 
            forecast horizon.

    Example:
        >>> # Initialize the model with graph metadata
        >>> model = GCEN(
        ...     in_feat_dim=1, 
        ...     gcn_out_feat=16, 
        ...     lstm_hidden_dim=32, 
        ...     lstm_num_layers=2, 
        ...     lstm_dropout=0.1, 
        ...     p_lag=12, 
        ...     t_pred=4, 
        ...     graph_info=graph_data
        ... )
        >>> # Train using a probabilistic loss function
        >>> model.fit(data_loader, optimizer, loss_fn=energy_score_loss,
        ...           num_epochs=100, m_samples=2, device=device, visualize=True)
        >>> # Generate a single stochastic forward pass
        >>> output = model(history_tensor)
        >>> # Generate an ensemble of 100 out-of-sample forecast samples
        >>> forecast_samples = model.predict(history_tensor, m_samples=100, device=device)
        
        >>> # In-sample analysis and residual diagnostics
        >>> in_sample_preds = model.predict_in_sample(train_data, m_samples=100, method="q_step")
        >>> residuals = model.get_residuals(in_sample_preds, train_data, point_method="median")
        >>> model.plot_residuals(residuals, plots_per_row=4)
        
        >>> # Out-of-sample performance evaluation
        >>> metrics_df = model.evaluate_forecasts(history_tensor, y_test, y_train, point_method="median")
        >>> print(metrics_df)
    """
    def __init__(self, in_feat_dim, gcn_out_feat, lstm_hidden_dim, lstm_num_layers, lstm_dropout, 
                 p_lag, t_pred, graph_info, noise_encode="add", noise_dist="gaussian", noise_dim=2, noise_std=1, 
                 graph_conv_params: typing.Optional[dict] = None, gcn_seed=21, temporal_seed=9):
        super().__init__()
        self.p_lag = p_lag # input_seq_len or past lagged values to use
        self.t_pred = t_pred # output_seq_len or forecast_horizon
        self.in_feat_dim = in_feat_dim
        self.gcn_out_feat = gcn_out_feat
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.noise_dist = noise_dist # gaussian or uniform
        self.noise_encode = noise_encode # add or concat
        self.noise_dim = noise_dim # only required when noise_encode = concat
        self.noise_std = noise_std 
        self.num_nodes = graph_info.num_nodes
        self.temporal_seed = temporal_seed

        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat", 
                "activation": None,
            }

        # Spatial Module: Graph Convolution
        self.gcn = GraphConv(
            in_feat=in_feat_dim,
            out_feat=gcn_out_feat,
            graph_info=graph_info,
            gcn_seed=gcn_seed,
            **graph_conv_params
        )
        # gcn_effective_out = 2 * gcn_out_feat if combination_type = "concat"
        if graph_conv_params["combination_type"] == "concat":
            gcn_out_feat *= 2

        if self.noise_encode=="add":
            lstm_input_dim = gcn_out_feat
        elif self.noise_encode=="concat":
            lstm_input_dim = gcn_out_feat + noise_dim
        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")
        
        # Temporal Module: LSTM
        torch.manual_seed(self.temporal_seed)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=False, # Input shape will be (seq_len, batch, input_size)
            num_layers = lstm_num_layers,
            dropout=lstm_dropout
        )

        # Output Module: Fully connected layer to generate forecasts
        self.output_layer = nn.Linear(lstm_hidden_dim, t_pred * in_feat_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for generating a single stochastic forecast.

        Args:
            x (torch.Tensor): Input tensor of shape :math:`(B, T_{in}, N, D_{in})`, 
                where :math:`B` is batch size, :math:`T_{in}` is ``p_lag``, 
                :math:`N` is ``num_nodes``, and :math:`D_{in}` is ``in_feat_dim``.

        Returns:
            torch.Tensor: Forecasted tensor of shape :math:`(B, T_{out}, N, D_{in})`, where :math:`T_{out}` is ``t_pred``.
        """
        # Input x shape: (batch_size, p_lag, num_nodes, in_feat_dim)
        # Convert shape to  (num_nodes, batch_size, p_lag, in_feat_dim)
        x = x.permute(2,0,1,3)
        z = self.gcn(x) # z has shape: (num_nodes, batch_size, p_lag, gcn_out_feat)
        
        # Permute back to a batch-first-like format: (batch_size, p_lag, num_nodes, gcn_out_feat)
        z = z.permute(1, 2, 0, 3)
        batch_size, seq_len = z.shape[0], z.shape[1]

        # Noise Injection for Sample Generation
        if self.noise_encode == "add":
            if self.noise_dist == "gaussian":
                noise = torch.randn_like(z) * self.noise_std # Sample from N(0,1)
            elif self.noise_dist == "uniform":
                noise = torch.rand_like(z) * self.noise_std # Sample from U(0,1)
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            z_noisy = z + noise

        elif self.noise_encode == "concat":
            noise_shape = (batch_size, seq_len, self.num_nodes, self.noise_dim)
            if self.noise_dist == "gaussian":
                noise = torch.randn(noise_shape) * self.noise_std # Sample from N(0,1)
                noise = noise.to(x.device) 
            elif self.noise_dist == "uniform":
                noise = torch.rand(noise_shape) * self.noise_std # Sample from U(0,1)
                noise = noise.to(x.device)
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            z_noisy = torch.cat([z, noise], dim=-1)

        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")

        # Sequential Processing with LSTM        
        lstm_input = z_noisy.permute(1, 0, 2, 3) # (p_lag, batch_size, num_nodes, gcn_out_feat)
        batch_size, gcn_out_feat = lstm_input.shape[1], lstm_input.shape[3]
        
        # Reshape for LSTM: (p_lag, batch_size * num_nodes, gcn_out_feat)
        lstm_input = lstm_input.reshape(self.p_lag, batch_size * self.num_nodes, gcn_out_feat)
        
        # LSTM processes the sequence of p lagged values
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Get the last hidden state: (batch_size * num_nodes, lstm_hidden_dim)
        last_hidden_state = h_n[-1]

        # Forecasting with Output Layer
        # Output: (batch_size * num_nodes, t_pred * in_feat_dim)
        predictions = self.output_layer(last_hidden_state)
        
        # Reshape to final forecast shape: (batch_size, num_nodes, t_pred, in_feat_dim)
        predictions = predictions.view(batch_size, self.num_nodes, self.t_pred, self.in_feat_dim)
        
        # Permute to match target Y shape: (batch_size, t_pred, num_nodes, in_feat_dim)
        predictions = predictions.permute(0, 2, 1, 3)

        return predictions

    def fit(self, data_loader, optimizer, loss_fn, num_epochs=100, m_samples=2, device="cpu", 
            monitor=True, visualize=True, verbose=False):
        """
        Trains the GCEN model using a probabilistic loss function.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch DataLoader yielding ``(x_batch, y_batch)``.
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
            loss_fn (callable): Loss function accepting ``(target, pred_samples)``.
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            m_samples (int, optional): Number of stochastic samples per batch 
                to estimate the probabilistic loss. Defaults to 2.
            device (str, optional): Device to run training on (``'cpu'`` or ``'cuda'``).
            monitor (bool, optional): If True, shows a progress bar.
            visualize (bool, optional): If True, plots the loss curve after training.
            verbose (bool, optional): If True, prints periodic loss updates.
        """
        self.train()
        if visualize:
            losses = []
        epoch_iter = (tqdm(range(num_epochs), desc="Training", unit="epoch", leave=True) 
                      if monitor else range(num_epochs))
        for epoch in epoch_iter:
            total_loss = 0
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = []
                for _ in range(m_samples):
                    pred = self(x_batch)
                    predictions.append(pred)
                predictions_tensor = torch.stack(predictions, dim=0)
                loss = loss_fn(y_batch, predictions_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            if visualize:
                losses.append(avg_loss)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        if visualize:
            epochs = np.arange(1, num_epochs+1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, losses)
            plt.title("Training loss")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()
        if verbose:
            print("Training finished.")

    def predict(self, history: torch.Tensor, m_samples: int = 100, 
                unstandardize: typing.Optional[list] = None, device="cpu") -> torch.Tensor:
        """
        Generates an ensemble of forecasts for a single historical observation.

        Args:
            history (torch.Tensor): Past observations of shape :math:`(T_{in}, N, D_{in})`.
            m_samples (int, optional): Number of ensemble members to generate. 
                Defaults to 100.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Computation device. Defaults to ``'cpu'``.

        Returns:
            torch.Tensor: Ensemble of forecasts of shape :math:`(M, T_{out}, N, D_{in})`, where :math:`M` is ``m_samples``.
        """
        self.eval()
        forecast_ensemble = []
        history = history.to(device)
        with torch.no_grad():
            for _ in range(m_samples):
                # Add batch dimension: (1, p_lag, num_nodes, in_feat_dim)
                history_batch = history.unsqueeze(0)
                # Generate forecast sample
                prediction = self(history_batch)  # shape: (batch_size, t_pred, num_nodes, in_feat_dim)
                prediction = prediction.squeeze(0).to(device) # remove batch dimension 
                if unstandardize is not None:
                    mean, std = unstandardize[0].to(device), unstandardize[1].to(device)
                    prediction = prediction * std + mean
                forecast_ensemble.append(prediction)
        return torch.stack(forecast_ensemble, dim=0)

    def evaluate_forecasts(self, history: torch.Tensor, y_true: torch.Tensor, y_train: torch.Tensor,
                           m_samples=100, n_repeats=50, point_method: typing.Union[str, float] = "median",
                             unstandardize=None, device=None):
        """
        Repeatedly generate probabilistic forecasts from a single trained model 
        and return summary metrics.

        This method performs Monte Carlo style evaluation by generating multiple 
        ensembles to account for the stochastic nature of the engression model.

        Args:
            history (torch.Tensor): Last ``p_lag`` observations of shape 
                :math:`(T_{in}, N, D_{in})`.
            y_true (torch.Tensor): Ground truth for the forecast horizon of shape 
                :math:`(T_{out}, N, D_{in})`.
            y_train (torch.Tensor): In-sample training data of shape 
                :math:`(T_{train}, N, D_{in})`. Required for scaling MASE and RMSSE.
            m_samples (int, optional): Number of samples per forecast ensemble. 
                Defaults to 100.
            n_repeats (int, optional): Number of times to repeat the ensemble 
                generation to calculate metric stability. Defaults to 50.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (torch.device or str, optional): Device for evaluation. 
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the mean and standard deviation 
            across repeats for the following metrics:
            
            * **Point**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration**: Empirical Coverage, Winkler Score.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            # CRPS approx as mean absolute error between samples and true values minus half the mean pairwise absolute differences
            # CRPS = E|X - y| - 0.5 E|X - X'|
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            # Calculate MAE for forecast period across all elements
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))

            # Calculate scaling factor: mean absolute difference of y_train across all elements
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            mase_score = mae_forecast / scale

            return mase_score.item()

        def rmsse(y_true, y_pred, y_train):
            # Calculate MSE for forecast period across all elements
            mse_forecast = torch.mean((y_true - y_pred)**2)

            # Calculate scaling factor: mean squared difference of y_train across all elements
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            rmsse_score = torch.sqrt(mse_forecast / scale)

            return rmsse_score.item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            """
            Calculates the rho-risk (normalized quantile loss) as defined in the DeepAR paper.
            
            Args:
                ground_truth (np.array): Shape (H, N)
                                         H = Forecast Horizon
                                         N = Number of Nodes
                forecast_ensemble (np.array): Shape (M, H, N, D)
                                              M = Ensemble samples
                                              D = Feature dimension (expected to be 1)
                rho (float): The quantile to evaluate (e.g., 0.5 or 0.9).
                
            Returns:
                float: The calculated rho-risk value.
            """

            # Squeeze the feature dimension D=1 to get shape (M, H, N)
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
            else:
                # Fallback if D > 1
                forecasts = forecast_ensemble
                
            # Calculate the Empirical Quantile (Z_hat)
            # We collapse the ensemble dimension (M) to get the rho-quantile prediction
            # resulting shape: (H, N)
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            
            # Calculate the Quantile Loss (L_rho) element-wise
            # The DeepAR paper uses a specific formula with a scaling factor of 2.
            # We use boolean masking to handle the asymmetric penalty correctly.
            
            # Mask for Underestimation: True where Ground Truth > Prediction
            under_bias = ground_truth > forecast_quantiles
            
            # Mask for Overestimation: True where Ground Truth <= Prediction
            over_bias = ~under_bias
            
            # Initialize loss array
            losses = np.zeros_like(ground_truth)
            
            # Apply penalty for Underestimation: 2 * rho * (Z - Z_hat)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            
            # Apply penalty for Overestimation: 2 * (1 - rho) * (Z_hat - Z)
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            # Aggregation and Normalization
            # The paper defines rho-risk as the sum of quantile losses divided by the sum of target values.
            # Summation occurs over the entire horizon (H) and all nodes (N).
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            
            rho_risk = total_loss / total_target
            
            return rho_risk

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            """
            Calculate empirical coverage probability that true values lie within predicted CIs.
        
            Args:
                true_values (array-like): True target values.
                lower_bounds (array-like): Lower bounds of predicted confidence intervals.
                upper_bounds (array-like): Upper bounds of predicted confidence intervals.
        
            Returns:
                float: Fraction of true values inside the confidence intervals.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            coverage = np.mean(inside)
            return coverage


        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            """
            Calculate average Winkler score for prediction intervals.
        
            Parameters:
            true_values (array-like): Ground truth values.
            lower_bounds (array-like): CI lower bounds
            upper_bounds (array-like): CI upper bounds.
            alpha (float): Significance level (e.g. 0.05 for 95% CI).
        
            Returns:
            float: Average Winkler score.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
        
            below = true_values < lower_bounds
            above = true_values > upper_bounds
        
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
        
            return np.mean(scores)

        all_metrics = []
        y_true = y_true.to(device)
        history = history.to(device)
        y_train = y_train.to(device)
        self.to(device)
        self.eval()

        for _ in range(n_repeats):
            with torch.no_grad():
                forecast_ensemble = self.predict(
                    history=history,
                    m_samples=m_samples,
                    unstandardize=unstandardize,
                    device=device
                )
                forecast_ensemble = torch.round(forecast_ensemble) # Since epidemic incidence cases can be integers only
                y_preds_ensemble = forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float quantile (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train)
                rmsse_val = rmsse(y_true, y_pred_point, y_train)

                rho_50 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5).item()
                rho_90 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9).item()

                lower_bound = torch.quantile(forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(forecast_ensemble, 0.975, dim=0)
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                    
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    # The following methods will be useful for diagnosis of in-sample predictions.
    
    def predict_in_sample(self, data: torch.Tensor, m_samples: int = 100, method: str = "1_step", batch_size: int = 64, 
                          unstandardize: typing.Optional[list] = None, device: str = "cpu") -> torch.Tensor:
        """
        Generates in-sample predictions for the entire training dataset.

        This method applies a sliding window across the provided historical data 
        to produce stochastic forecasts for every possible time step.

        Args:
            data (torch.Tensor): The original dataset of shape :math:`(T, N, D)`, 
                where :math:`T` is the total time steps.
            m_samples (int, optional): Number of stochastic forecast samples 
                to generate per window. Defaults to 100.
            method (str, optional): Strategy for in-sample forecasting. 
                Options include:
                
                * ``"1_step"``: Slides the input window by 1 step, recording 
                  only the 1-step ahead forecast for each position.
                * ``"q_step"``: Slides by ``t_pred`` (q) steps, recording 
                  full non-overlapping forecast horizons.
                  
                Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows to process 
                simultaneously to optimize memory. Defaults to 64.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Device to perform computations on. 
                Defaults to ``"cpu"``.

        Returns:
            torch.Tensor: In-sample forecast ensemble of shape 
            :math:`(M, T, N, D)`, where :math:`M` is ``m_samples``.

        Note:
            The first ``p_lag`` time steps in the output tensor will contain 
            ``NaN`` values because there is insufficient historical context 
            to generate a forecast for the beginning of the sequence.
        """
        self.eval()
        self.to(device)
        data = data.to(device)
        if data.ndim < 3:
            data = data.unsqueeze(-1)
        T, N, D = data.shape
        p = self.p_lag
        q = self.t_pred
        
        # Pre-allocate the full prediction tensor with NaNs to match (T, N, D)
        # Shape: (m_samples, T, N, D)
        full_preds = torch.full((m_samples, T, N, D), float('nan'), device=device)
        
        # Determine the starting indices for our sliding windows based on the chosen method
        if method == "1_step":
            start_indices = list(range(0, T - p))
        elif method == "q_step":
            start_indices = list(range(0, T - p, q))
        else:
            raise ValueError("Method must be either '1_step' or 'q_step'")

        with torch.no_grad():
            for batch_start in range(0, len(start_indices), batch_size):
                batch_indices = start_indices[batch_start : batch_start + batch_size]
                
                # Shape: (B, p, N, D)
                batch_in = torch.stack([data[i : i + p] for i in batch_indices])
                
                # Generate m_samples for the current batch
                batch_samples = []
                for _ in range(m_samples):
                    # self(x) returns predictions of shape (B, q, N, D)
                    batch_samples.append(self(batch_in))
                
                # Shape: (m_samples, B, q, N, D)
                batch_samples = torch.stack(batch_samples, dim=0)
                
                # Place the predictions into the correct temporal alignment in the full tensor
                for b_idx, i in enumerate(batch_indices):
                    if method == "1_step":
                        # Record only the 1-step ahead forecast (index 0 of the q dimension)
                        # Target time index is exactly i + p
                        full_preds[:, i + p, :, :] = batch_samples[:, b_idx, 0, :, :]
                        
                    elif method == "q_step":
                        # Record the full q-step window
                        # Target time indices are i + p to i + p + q
                        end_idx = min(i + p + q, T)
                        valid_q = end_idx - (i + p) # Handle boundary condition if T isn't perfectly divisible
                        full_preds[:, i + p : end_idx, :, :] = batch_samples[:, b_idx, :valid_q, :, :]
                        
        # Apply unstandardization if parameters are provided
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            # Broadcasting matches the trailing (N, D) dimensions of full_preds
            # NaNs in the first p_lag steps will remain NaN
            full_preds = full_preds * std + mean
        return full_preds

    def get_residuals(self, in_sample_preds: torch.Tensor, original_data: torch.Tensor,
                     point_method: typing.Union[str, float] = "median") -> torch.Tensor:
        """
        Computes the residual matrix from in-sample predictions.

        This method reduces the stochastic ensemble into a single point forecast 
        using the specified ``point_method`` and calculates the error: 
        :math:`Residual = Actual - Predicted`.

        Args:
            in_sample_preds (torch.Tensor): Predictions from :meth:`predict_in_sample` 
                with shape :math:`(M, T, N, D)`, where :math:`M` is the number 
                of samples.
            original_data (torch.Tensor): The ground truth dataset of shape 
                :math:`(T, N, D)`.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float representing a quantile (e.g., ``0.75``). 
                Defaults to ``"median"``.

        Returns:
            torch.Tensor: Residual tensor of shape :math:`(T, N, D)`.

        Note:
            Following the structure of the in-sample predictions, the first 
            ``p_lag`` time steps will contain ``NaN`` values. These represent 
            the "warm-up" period where no forecasts were generated.
        """
        original_data = original_data.to(in_sample_preds.device)
        
        # Calculate the point predictions over the ensemble dimension (dim=0)
        # Shape: (T, N, D)
        if point_method == "mean":
            point_preds = torch.mean(in_sample_preds.float(), dim=0)
        elif point_method == "median":
            point_preds = torch.median(in_sample_preds, dim=0).values
        elif isinstance(point_method, (float, int)):
            # Extract a specific quantile
            point_preds = torch.quantile(in_sample_preds.float(), q=float(point_method), dim=0)
        else:
            raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")
        
        # Compute residuals (Actual - Predicted)
        # NaNs in median_preds will naturally propagate to the residuals
        residuals = original_data - point_preds
        
        return residuals

    def evaluate_in_sample_fit(
        self, 
        data: torch.Tensor, 
        m_samples: int = 100, 
        n_repeats: int = 10,
        method: str = "1_step",
        batch_size: int = 32,
        point_method: typing.Union[str, float] = "median",
        unstandardize: typing.Optional[list] = None, 
        device: typing.Optional[str] = None):
        """
        Repeatedly generates in-sample probabilistic forecasts and returns summary metrics.

        This method assesses the model's ability to reconstruct the historical training 
        sequence by performing multiple stochastic passes over the data. It accounts 
        for the "Engression" noise injection by repeating the process ``n_repeats`` 
        times and reporting the stability of the metrics.

        Args:
            data (torch.Tensor): The full historical dataset of shape 
                :math:`(T, N, D_{in})`, where :math:`T` is total time steps.
            m_samples (int, optional): Number of stochastic samples per forecast 
                ensemble window. Defaults to 100.
            n_repeats (int, optional): Number of full evaluation cycles to 
                perform to calculate mean/std of metrics. Defaults to 10.
            method (str, optional): The sliding window strategy: ``"1_step"`` 
                or ``"q_step"``. Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows processed in a single 
                forward pass. Defaults to 32.
            point_method (str or float, optional): Metric for extracting a single 
                forecast from the ensemble to compute point errors (SMAPE, MAE, 
                RMSE, MASE, RMSSE). Accepts ``"median"``, ``"mean"``, or a 
                float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` used to 
                rescale predictions and ground truth to original units. 
                Defaults to None.
            device (str or torch.device, optional): Computation device. 
                Defaults to None.

        Returns:
            pd.DataFrame: A summary table containing the mean and standard 
            deviation for the following metrics across all repeats:
            
            * **Point Metrics**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic Metrics**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration Metrics**: Empirical Coverage, Winkler Score.

        Note:
            Similar to :meth:`predict_in_sample`, metrics are calculated only 
            for the time steps after the initial ``p_lag`` warm-up period.
        """
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)
            scale = scale if scale != 0 else 1e-8
            return (mae_forecast / scale).item()

        def rmsse(y_true, y_pred, y_train):
            mse_forecast = torch.mean((y_true - y_pred)**2)
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)
            scale = scale if scale != 0 else 1e-8
            return torch.sqrt(mse_forecast / scale).item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
                ground_truth = ground_truth.squeeze(-1)
            else:
                forecasts = forecast_ensemble
                
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            under_bias = ground_truth > forecast_quantiles
            over_bias = ~under_bias
            
            losses = np.zeros_like(ground_truth)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            return total_loss / total_target if total_target != 0 else 0.0

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            return np.mean(inside)

        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
            
            below = true_values < lower_bounds
            above = true_values > upper_bounds
            
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
            return np.mean(scores)

        # Evaluation Logic
        all_metrics = []
        self.to(device)
        self.eval()
        data = data.to(device)
        
        # Prepare ground truth (unstandardize if needed to match predictions)
        y_train_full = data.clone()
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            y_train_full = y_train_full * std + mean
            
        # Ground truth for metrics ignores the first p_lag steps
        y_true = y_train_full[self.p_lag:]

        for _ in range(n_repeats):
            with torch.no_grad():
                # Generate predictions mapping to the full dataset
                full_in_sample_preds = self.predict_in_sample(
                    data=data,
                    m_samples=m_samples,
                    method=method,
                    batch_size=batch_size,
                    unstandardize=unstandardize,
                    device=device
                )
                
                # Slice out the initial p_lag steps (which are NaNs)
                valid_forecast_ensemble = full_in_sample_preds[:, self.p_lag:, :, :]
                
                # Round and clamp because epidemic cases are non-negative integers
                valid_forecast_ensemble = torch.clamp(torch.round(valid_forecast_ensemble), min=0)
                y_preds_ensemble = valid_forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train_full)
                rmsse_val = rmsse(y_true, y_pred_point, y_train_full)

                rho_50 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5)
                rho_90 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9)

                lower_bound = torch.quantile(valid_forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(valid_forecast_ensemble, 0.975, dim=0)
                
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    def plot_in_sample_fit(
        self,
        in_sample_preds: torch.Tensor,
        original_data: torch.Tensor,
        plots_per_row: int = 4,
        confidence_level: float = 0.95,
        node_names: typing.Optional[list] = None,
        title: str = 'GCEN: In-Sample Forecast Fit vs. Actual',
        savefig: bool = False,
        filename: typing.Optional[str] = None):
        """
        Plots in-sample forecasted time series against actual ground truth values.

        This method generates a grid of subplots (one per node) showing the median 
        forecast, the ground truth, and a shaded prediction interval based on the 
        stochastic ensemble.

        Args:
            in_sample_preds (torch.Tensor): Prediction ensemble from 
                :meth:`predict_in_sample` of shape :math:`(M, T, N, D)`.
            original_data (torch.Tensor): Ground truth dataset of shape 
                :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display per row 
                in the figure grid. Defaults to 4.
            confidence_level (float, optional): The width of the shaded prediction 
                interval (e.g., ``0.95`` for a 95% interval). Defaults to 0.95.
            node_names (list of str, optional): Custom labels for each node. If 
                None, nodes are labeled by index. Defaults to None.
            title (str, optional): The main title for the entire figure. 
                Defaults to 'GCEN: In-Sample Forecast Fit vs. Actual'.
            savefig (bool, optional): If True, the resulting figure is exported 
                to a file. Defaults to False.
            filename (str, optional): The file path/name for saving the figure 
                (e.g., 'fit_plot.png'). Required if ``savefig`` is True.

        Note:
            The shaded area represents the uncertainty captured by the Engression 
            noise injection. The bounds are calculated as the 
            :math:`(1 - confidence\_level)/2` and :math:`(1 + confidence\_level)/2` 
            quantiles of the ensemble.

        Returns:
            None: This method displays the plot using ``plt.show()`` or saves it to disk.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if in_sample_preds.dim() == 4:
            in_sample_preds = in_sample_preds.squeeze(-1)  # Shape: (m_samples, T, N)
        if original_data.dim() == 3:
            original_data = original_data.squeeze(-1)      # Shape: (T, N)
            
        m_samples, T, num_nodes = in_sample_preds.shape
        p = self.p_lag
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 4 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        # Calculate dynamic quantiles based on confidence_level
        alpha = 1.0 - confidence_level
        q_lower = alpha / 2.0
        q_upper = 1.0 - (alpha / 2.0)
        
        # Isolate the valid predictions (ignoring the first p_lag NaNs) to compute statistics safely
        valid_preds = in_sample_preds[:, p:, :]
        
        # Initialize full tensors with NaNs to maintain temporal alignment
        forecast_median = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        lower_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        upper_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        
        # Compute statistics only on the valid temporal segment and apply clamping/rounding
        forecast_median[p:] = torch.clamp(torch.round(torch.median(valid_preds, dim=0).values), min=0)
        lower_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_lower, dim=0), min=0)
        upper_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_upper, dim=0), min=0)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Ground truth line
            ax.plot(timesteps, original_data[:, node].cpu(), marker='o', 
                    label='Actual', markersize=4, color='black')
            
            # Median forecast line (NaNs will cause matplotlib to skip the first p_lag steps automatically)
            ax.plot(timesteps, forecast_median[:, node].cpu(), marker='+', 
                    label='Fitted', color='blue')
            
            # Shaded Confidence Interval
            ax.fill_between(timesteps,
                            lower_bound[:, node].cpu(),
                            upper_bound[:, node].cpu(),
                            color='gray', alpha=0.8, label=f'{int(confidence_level*100)}% CI')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            plt.savefig(filename, format='pdf')
            
        plt.show()

    def plot_residuals(
        self,
        residuals: torch.Tensor,
        plots_per_row: int = 4,
        node_names: typing.Optional[list] = None,
        title: str = 'GCEN: In-Sample Residuals per Node',
        savefig: bool = False,
        filename: typing.Optional[str] = None):
        """
        Plots the time series of residuals (errors) for each node in a grid.

        This visualization helps in identifying systematic biases or patterns in 
        the model's errors across different spatial nodes. A horizontal line at 
        zero is included for reference.

        Args:
            residuals (torch.Tensor): Residual matrix obtained from 
                :meth:`get_residuals`, with shape :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display in 
                each row of the grid. Defaults to 4.
            node_names (list of str, optional): Labels for each node subplot. 
                If None, node indices are used. Defaults to None.
            title (str, optional): The main title for the figure. 
                Defaults to 'GCEN: In-Sample Residuals per Node'.
            savefig (bool, optional): If True, the plot will be saved to the 
                specified ``filename``. Defaults to False.
            filename (str, optional): Path where the figure should be saved. 
                Required if ``savefig`` is True. Defaults to None.

        Note:
            Any ``NaN`` values present in the residuals (typically the first 
            ``p_lag`` steps) are automatically handled by the plotting 
            backend and will appear as gaps in the time series.

        Returns:
            None: Displays the plot using ``plt.show()`` or saves the file.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if residuals.dim() == 3:
            residuals = residuals.squeeze(-1)  # Shape: (T, N)
            
        T, num_nodes = residuals.shape
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        # Slightly shorter figure height per row since residuals usually need less vertical space
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 3 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Plot the residual line (NaNs from the first p_lag steps are safely ignored by matplotlib)
            ax.plot(timesteps, residuals[:, node].cpu(), marker='o', 
                    linestyle='-', markersize=3, color='purple', alpha=0.7, label='Residual')
            
            # Add a horizontal zero line for easy visual bias checking
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            # Saving as PDF is generally preferred for high-quality manuscript figures
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            
        plt.show()


# 2. Multivariate Engression Network: MVEN 
class MVEN(nn.Module):
    """Multivariate Engression Network: a multivariate engression-LSTM for probabilistic time-series forecasting.

    Args:
        p_lag (int): Number of past timesteps used as input (input sequence length).
        t_pred (int): Number of future timesteps predicted (forecast horizon).
        in_feat_dim (int): Input feature dimension per node.
        lstm_hidden_dim (int): Hidden dimension size of the LSTM temporal module.
        lstm_num_layers (int): Number of layers in the LSTM.
        lstm_dropout (float): Dropout probability in the LSTM.
        noise_dist (str): Distribution type for noise injection, either 'gaussian' or 'uniform'.
        noise_encode (str): Method for noise injection: 'add' (additive) or 'concat' (concatenation).
        noise_dim (int): Dimension of noise features if concatenated.
        noise_std (int): Standard deviation/scaling of the noise.
        num_nodes (int): Number of nodes in the data.
        temporal_seed (int): Seed for reproducibility in LSTM initialization.

    Example:
        >>> model = MVEN(
        ...     in_feat_dim=1, lstm_hidden_dim=32, lstm_num_layers=2, 
        ...     lstm_dropout=0.1, p_lag=12, t_pred=4
        ... )
        >>> model.fit(
        ...     data_loader, optimizer, loss_fn=energy_score_loss,
        ...     num_epochs=100, m_samples=2, device=device, visualize=True
        ... )
        >>> output = model(history_tensor)
        >>> forecast_samples = model.predict(history_tensor, m_samples=100, device=device)
        >>> # In-sample analysis
        >>> in_sample_preds = model.predict_in_sample(train_data, m_samples=100, method="q_step")
        >>> residuals = model.get_residuals(in_sample_preds, train_data, point_method="median")
        >>> model.plot_residuals(residuals, plots_per_row=4)
        >>> # Out-of-sample evaluation
        >>> metrics_df = model.evaluate_forecasts(
        ...     history_tensor, y_test, y_train, point_method="median"
        ... )
        >>> print(metrics_df)
    """
    def __init__(self, in_feat_dim, num_nodes, lstm_hidden_dim, lstm_num_layers, lstm_dropout, p_lag, t_pred,
                noise_encode="add", noise_dist="gaussian", noise_dim=2, noise_std=1, temporal_seed=9):
        super().__init__()
        self.p_lag = p_lag # input_seq_len or past lagged values to use
        self.t_pred = t_pred # output_seq_len or forecast_horizon
        self.in_feat_dim = in_feat_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.num_nodes = num_nodes
        self.noise_dist = noise_dist # gaussian or uniform
        self.noise_encode = noise_encode # add or concat
        self.noise_dim = noise_dim # only required when noise_encode = concat
        self.noise_std = noise_std
        self.temporal_seed = temporal_seed

        if self.noise_encode=="add":
            lstm_input_dim = in_feat_dim
        elif self.noise_encode=="concat":
            lstm_input_dim = in_feat_dim + self.noise_dim
        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")
        
        
        torch.manual_seed(self.temporal_seed)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=False, # Input shape will be (seq_len, batch, input_size)
            num_layers = lstm_num_layers,
            dropout=lstm_dropout
        )

        # Output Module: Fully connected layer to generate forecasts
        self.output_layer = nn.Linear(lstm_hidden_dim, t_pred * in_feat_dim)

    def forward(self, x: torch.Tensor):
        # Input x shape: (batch_size, input_seq_len, N, D)

        batch_size, seq_len = x.shape[0], x.shape[1]
        if self.noise_encode == "add":
            if self.noise_dist == "uniform":
                noise = torch.rand_like(x) * self.noise_std # Sample from U(0,1)
            elif self.noise_dist == "gaussian":
                noise = torch.randn_like(x) * self.noise_std # Sample from N(0,1)
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            x = x + noise

        elif self.noise_encode == "concat":
            noise_shape = (batch_size, seq_len, self.num_nodes, self.noise_dim)
            if self.noise_dist == "gaussian":
                noise = torch.randn(noise_shape) * self.noise_std # Sample from N(0,1)
                noise = noise.to(x.device) 
            elif self.noise_dist == "uniform":
                noise = torch.rand(noise_shape) * self.noise_std # Sample from U(0,1)
                noise = noise.to(x.device) 
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            x = torch.cat([x, noise], dim=-1)

        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")

            
        
        # Sequential Processing with LSTM        
        lstm_input = x.permute(1, 0, 2, 3) # (input_seq_len, batch_size, N, D)
        batch_size, in_feat_dim = lstm_input.shape[1], lstm_input.shape[3]
        
        # Reshape for LSTM: (input_seq_len, batch_size * N, gcn_out_feat)
        lstm_input = lstm_input.reshape(self.p_lag, batch_size * self.num_nodes, in_feat_dim)
        
        # LSTM processes the sequence of p lagged values
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Get the last hidden state: (batch_size * N, lstm_hidden_dim)
        last_hidden_state = h_n[-1]

        # Forecasting with Output Layer
        # Output: (batch_size * N, t_pred * D)
        predictions = self.output_layer(last_hidden_state)
        
        # Reshape to final forecast shape: (batch_size, N, t_pred, D)
        predictions = predictions.view(batch_size, self.num_nodes, self.t_pred, self.in_feat_dim)
        
        # Permute to match target Y shape: (batch_size, t_pred, N, D)
        predictions = predictions.permute(0, 2, 1, 3)

        return predictions

    def fit(self, data_loader, optimizer, loss_fn, num_epochs=100, m_samples=2, device="cpu", 
            monitor=True, visualize=True, verbose=False):
        """Trains the MVEN model using the provided data loader, optimizer, and loss function.
        
        Args:
            data_loader (torch.utils.data.DataLoader): Iterator that yields (x_batch, y_batch).
            optimizer (torch.optim.Optimizer): Optimizer instance used for training.
            loss_fn (Callable): Loss function taking arguments (target, pred_samples).
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            m_samples (int, optional): Number of stochastic samples per batch for probabilistic loss. Defaults to 2.
            device (str, optional): Computation device (``'cpu'`` or ``'cuda'``). Defaults to ``"cpu"``.
            monitor (bool, optional): If True, displays a tqdm progress bar. Defaults to True.
            visualize (bool, optional): If True, plots the training loss curve after training completes. Defaults to True.
            verbose (bool, optional): If True, prints periodic loss updates to the console. Defaults to False.
        """
        self.train()
        if visualize:
            losses = []
        epoch_iter = (tqdm(range(num_epochs), desc="Training", unit="epoch", leave=True) 
                      if monitor else range(num_epochs))
        for epoch in epoch_iter:
            total_loss = 0
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = []
                for _ in range(m_samples):
                    pred = self(x_batch)
                    predictions.append(pred)
                predictions_tensor = torch.stack(predictions, dim=0)
                loss = loss_fn(y_batch, predictions_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            if visualize:
                losses.append(avg_loss)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        if visualize:
            epochs = np.arange(1, num_epochs+1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, losses)
            plt.title("Training loss")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()
        if verbose:
            print("Training finished.")

    def predict(self, history: torch.Tensor, m_samples: int = 100, unstandardize=None, device="cpu") -> torch.Tensor:
        """
        Generates an ensemble of forecasts for a single historical observation.

        Args:
            history (torch.Tensor): Past observations of shape :math:`(T_{in}, N, D_{in})`.
            m_samples (int, optional): Number of ensemble members to generate. 
                Defaults to 100.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Computation device. Defaults to ``'cpu'``.

        Returns:
            torch.Tensor: Ensemble of forecasts of shape :math:`(M, T_{out}, N, D_{in})`, where :math:`M` is ``m_samples``.
        """
        self.eval()
        forecast_ensemble = []
        history = history.to(device)
        with torch.no_grad():
            for _ in range(m_samples):
                # Add batch dimension: (1, p_lag, N, D)
                history_batch = history.unsqueeze(0)
                # Generate forecast sample
                prediction = self(history_batch)  # shape: (batch_size, t_pred, N, D)
                prediction = prediction.squeeze(0).to(device) # remove batch dimension 
                if unstandardize is not None:
                    mean, std = unstandardize[0].to(device), unstandardize[1].to(device)
                    prediction = prediction * std + mean
                forecast_ensemble.append(prediction)
        return torch.stack(forecast_ensemble, dim=0)

    def evaluate_forecasts(self, history: torch.Tensor, y_true: torch.Tensor, y_train: torch.Tensor,
                           m_samples=100, n_repeats=50, point_method: typing.Union[str, float] = "median",
                             unstandardize=None, device=None):
        """
        Repeatedly generate probabilistic forecasts from a single trained model 
        and return summary metrics.

        This method performs Monte Carlo style evaluation by generating multiple 
        ensembles to account for the stochastic nature of the engression model.

        Args:
            history (torch.Tensor): Last ``p_lag`` observations of shape 
                :math:`(T_{in}, N, D_{in})`.
            y_true (torch.Tensor): Ground truth for the forecast horizon of shape 
                :math:`(T_{out}, N, D_{in})`.
            y_train (torch.Tensor): In-sample training data of shape 
                :math:`(T_{train}, N, D_{in})`. Required for scaling MASE and RMSSE.
            m_samples (int, optional): Number of samples per forecast ensemble. 
                Defaults to 100.
            n_repeats (int, optional): Number of times to repeat the ensemble 
                generation to calculate metric stability. Defaults to 50.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (torch.device or str, optional): Device for evaluation. 
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the mean and standard deviation 
            across repeats for the following metrics:
            
            * **Point**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration**: Empirical Coverage, Winkler Score.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            # CRPS approx as mean absolute error between samples and true values minus half the mean pairwise absolute differences
            # CRPS = E|X - y| - 0.5 E|X - X'|
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            # Calculate MAE for forecast period across all elements
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))

            # Calculate scaling factor: mean absolute difference of y_train across all elements
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            mase_score = mae_forecast / scale

            return mase_score.item()

        def rmsse(y_true, y_pred, y_train):
            # Calculate MSE for forecast period across all elements
            mse_forecast = torch.mean((y_true - y_pred)**2)

            # Calculate scaling factor: mean squared difference of y_train across all elements
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            rmsse_score = torch.sqrt(mse_forecast / scale)

            return rmsse_score.item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            """
            Calculates the rho-risk (normalized quantile loss) as defined in the DeepAR paper.
            
            Args:
                ground_truth (np.array): Shape (H, N)
                                         H = Forecast Horizon
                                         N = Number of Nodes
                forecast_ensemble (np.array): Shape (M, H, N, D)
                                              M = Ensemble samples
                                              D = Feature dimension (expected to be 1)
                rho (float): The quantile to evaluate (e.g., 0.5 or 0.9).
                
            Returns:
                float: The calculated rho-risk value.
            """

            # Squeeze the feature dimension D=1 to get shape (M, H, N)
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
            else:
                # Fallback if D > 1
                forecasts = forecast_ensemble
                
            # Calculate the Empirical Quantile (Z_hat)
            # We collapse the ensemble dimension (M) to get the rho-quantile prediction
            # resulting shape: (H, N)
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            
            # Calculate the Quantile Loss (L_rho) element-wise
            # The DeepAR paper uses a specific formula with a scaling factor of 2.
            # We use boolean masking to handle the asymmetric penalty correctly.
            
            # Mask for Underestimation: True where Ground Truth > Prediction
            under_bias = ground_truth > forecast_quantiles
            
            # Mask for Overestimation: True where Ground Truth <= Prediction
            over_bias = ~under_bias
            
            # Initialize loss array
            losses = np.zeros_like(ground_truth)
            
            # Apply penalty for Underestimation: 2 * rho * (Z - Z_hat)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            
            # Apply penalty for Overestimation: 2 * (1 - rho) * (Z_hat - Z)
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            # Aggregation and Normalization
            # The paper defines rho-risk as the sum of quantile losses divided by the sum of target values.
            # Summation occurs over the entire horizon (H) and all nodes (N).
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            
            rho_risk = total_loss / total_target
            
            return rho_risk

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            """
            Calculate empirical coverage probability that true values lie within predicted CIs.
        
            Args:
                true_values (array-like): True target values.
                lower_bounds (array-like): Lower bounds of predicted confidence intervals.
                upper_bounds (array-like): Upper bounds of predicted confidence intervals.
        
            Returns:
                float: Fraction of true values inside the confidence intervals.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            coverage = np.mean(inside)
            return coverage


        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            """
            Calculate average Winkler score for prediction intervals.
        
            Parameters:
            true_values (array-like): Ground truth values.
            lower_bounds (array-like): CI lower bounds
            upper_bounds (array-like): CI upper bounds.
            alpha (float): Significance level (e.g. 0.05 for 95% CI).
        
            Returns:
            float: Average Winkler score.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
        
            below = true_values < lower_bounds
            above = true_values > upper_bounds
        
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
        
            return np.mean(scores)

        all_metrics = []
        y_true = y_true.to(device)
        history = history.to(device)
        y_train = y_train.to(device)
        self.to(device)
        self.eval()

        for _ in range(n_repeats):
            with torch.no_grad():
                forecast_ensemble = self.predict(
                    history=history,
                    m_samples=m_samples,
                    unstandardize=unstandardize,
                    device=device
                )
                forecast_ensemble = torch.round(forecast_ensemble) # Since epidemic incidence cases can be integers only
                y_preds_ensemble = forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float quantile (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train)
                rmsse_val = rmsse(y_true, y_pred_point, y_train)

                rho_50 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5).item()
                rho_90 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9).item()

                lower_bound = torch.quantile(forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(forecast_ensemble, 0.975, dim=0)
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                    
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    # The following methods will be useful for diagnosis of in-sample predictions.
    
    def predict_in_sample(self, data: torch.Tensor, m_samples: int = 100, method: str = "1_step", batch_size: int = 64, 
                          unstandardize: typing.Optional[list] = None, device: str = "cpu") -> torch.Tensor:
        """
        Generates in-sample predictions for the entire training dataset.

        This method applies a sliding window across the provided historical data 
        to produce stochastic forecasts for every possible time step.

        Args:
            data (torch.Tensor): The original dataset of shape :math:`(T, N, D)`, 
                where :math:`T` is the total time steps.
            m_samples (int, optional): Number of stochastic forecast samples 
                to generate per window. Defaults to 100.
            method (str, optional): Strategy for in-sample forecasting. 
                Options include:
                
                * ``"1_step"``: Slides the input window by 1 step, recording 
                  only the 1-step ahead forecast for each position.
                * ``"q_step"``: Slides by ``t_pred`` (q) steps, recording 
                  full non-overlapping forecast horizons.
                  
                Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows to process 
                simultaneously to optimize memory. Defaults to 64.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Device to perform computations on. 
                Defaults to ``"cpu"``.

        Returns:
            torch.Tensor: In-sample forecast ensemble of shape :math:`(M, T, N, D)`, where :math:`M` is ``m_samples``.

        Note:
            The first ``p_lag`` time steps in the output tensor will contain 
            ``NaN`` values because there is insufficient historical context 
            to generate a forecast for the beginning of the sequence.
        """
        self.eval()
        self.to(device)
        data = data.to(device)
        if data.ndim < 3:
            data = data.unsqueeze(-1)
        T, N, D = data.shape
        p = self.p_lag
        q = self.t_pred
        
        # Pre-allocate the full prediction tensor with NaNs to match (T, N, D)
        # Shape: (m_samples, T, N, D)
        full_preds = torch.full((m_samples, T, N, D), float('nan'), device=device)
        
        # Determine the starting indices for our sliding windows based on the chosen method
        if method == "1_step":
            start_indices = list(range(0, T - p))
        elif method == "q_step":
            start_indices = list(range(0, T - p, q))
        else:
            raise ValueError("Method must be either '1_step' or 'q_step'")

        with torch.no_grad():
            for batch_start in range(0, len(start_indices), batch_size):
                batch_indices = start_indices[batch_start : batch_start + batch_size]
                
                # Shape: (B, p, N, D)
                batch_in = torch.stack([data[i : i + p] for i in batch_indices])
                
                # Generate m_samples for the current batch
                batch_samples = []
                for _ in range(m_samples):
                    # self(x) returns predictions of shape (B, q, N, D)
                    batch_samples.append(self(batch_in))
                
                # Shape: (m_samples, B, q, N, D)
                batch_samples = torch.stack(batch_samples, dim=0)
                
                # Place the predictions into the correct temporal alignment in the full tensor
                for b_idx, i in enumerate(batch_indices):
                    if method == "1_step":
                        # Record only the 1-step ahead forecast (index 0 of the q dimension)
                        # Target time index is exactly i + p
                        full_preds[:, i + p, :, :] = batch_samples[:, b_idx, 0, :, :]
                        
                    elif method == "q_step":
                        # Record the full q-step window
                        # Target time indices are i + p to i + p + q
                        end_idx = min(i + p + q, T)
                        valid_q = end_idx - (i + p) # Handle boundary condition if T isn't perfectly divisible
                        full_preds[:, i + p : end_idx, :, :] = batch_samples[:, b_idx, :valid_q, :, :]
                        
        # Apply unstandardization if parameters are provided
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            # Broadcasting matches the trailing (N, D) dimensions of full_preds
            # NaNs in the first p_lag steps will remain NaN
            full_preds = full_preds * std + mean
        return full_preds

    def get_residuals(self, in_sample_preds: torch.Tensor, original_data: torch.Tensor,
                     point_method: typing.Union[str, float] = "median") -> torch.Tensor:
        """
        Computes the residual matrix from in-sample predictions.

        This method reduces the stochastic ensemble into a single point forecast 
        using the specified ``point_method`` and calculates the error: 
        :math:`Residual = Actual - Predicted`.

        Args:
            in_sample_preds (torch.Tensor): Predictions from :meth:`predict_in_sample` 
                with shape :math:`(M, T, N, D)`, where :math:`M` is the number 
                of samples.
            original_data (torch.Tensor): The ground truth dataset of shape 
                :math:`(T, N, D)`.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float representing a quantile (e.g., ``0.75``). 
                Defaults to ``"median"``.

        Returns:
            torch.Tensor: Residual tensor of shape :math:`(T, N, D)`.

        Note:
            Following the structure of the in-sample predictions, the first 
            ``p_lag`` time steps will contain ``NaN`` values. These represent 
            the "warm-up" period where no forecasts were generated.
        """
        original_data = original_data.to(in_sample_preds.device)
        
        # Calculate the point predictions over the ensemble dimension (dim=0)
        # Shape: (T, N, D)
        if point_method == "mean":
            point_preds = torch.mean(in_sample_preds.float(), dim=0)
        elif point_method == "median":
            point_preds = torch.median(in_sample_preds, dim=0).values
        elif isinstance(point_method, (float, int)):
            # Extract a specific quantile
            point_preds = torch.quantile(in_sample_preds.float(), q=float(point_method), dim=0)
        else:
            raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")
        
        # Compute residuals (Actual - Predicted)
        # NaNs in median_preds will naturally propagate to the residuals
        residuals = original_data - point_preds
        
        return residuals

    def evaluate_in_sample_fit(
        self, 
        data: torch.Tensor, 
        m_samples: int = 100, 
        n_repeats: int = 10,
        method: str = "1_step",
        batch_size: int = 32,
        point_method: typing.Union[str, float] = "median",
        unstandardize: typing.Optional[list] = None, 
        device: typing.Optional[str] = None):
        """
        Repeatedly generates in-sample probabilistic forecasts and returns summary metrics.

        This method assesses the model's ability to reconstruct the historical training 
        sequence by performing multiple stochastic passes over the data. It accounts 
        for the "Engression" noise injection by repeating the process ``n_repeats`` 
        times and reporting the stability of the metrics.

        Args:
            data (torch.Tensor): The full historical dataset of shape 
                :math:`(T, N, D_{in})`, where :math:`T` is total time steps.
            m_samples (int, optional): Number of stochastic samples per forecast 
                ensemble. Defaults to 100.
            n_repeats (int, optional): Number of full evaluation cycles to 
                perform to calculate mean/std of metrics. Defaults to 10.
            method (str, optional): The sliding window strategy: ``"1_step"`` 
                or ``"q_step"``. Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows processed in a single 
                forward pass. Defaults to 32.
            point_method (str or float, optional): Metric for extracting a single 
                forecast from the ensemble to compute point errors (SMAPE, MAE, 
                RMSE, MASE, RMSSE). Accepts ``"median"``, ``"mean"``, or a 
                float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` used to 
                rescale predictions and ground truth to original units. 
                Defaults to None.
            device (str or torch.device, optional): Computation device. 
                Defaults to None.

        Returns:
            pd.DataFrame: A summary table containing the mean and standard 
            deviation for the following metrics across all repeats:
            
            * **Point Metrics**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic Metrics**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration Metrics**: Empirical Coverage, Winkler Score.

        Note:
            Similar to :meth:`predict_in_sample`, metrics are calculated only 
            for the time steps after the initial ``p_lag`` warm-up period.
        """
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)
            scale = scale if scale != 0 else 1e-8
            return (mae_forecast / scale).item()

        def rmsse(y_true, y_pred, y_train):
            mse_forecast = torch.mean((y_true - y_pred)**2)
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)
            scale = scale if scale != 0 else 1e-8
            return torch.sqrt(mse_forecast / scale).item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
                ground_truth = ground_truth.squeeze(-1)
            else:
                forecasts = forecast_ensemble
                
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            under_bias = ground_truth > forecast_quantiles
            over_bias = ~under_bias
            
            losses = np.zeros_like(ground_truth)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            return total_loss / total_target if total_target != 0 else 0.0

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            return np.mean(inside)

        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
            
            below = true_values < lower_bounds
            above = true_values > upper_bounds
            
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
            return np.mean(scores)

        # Evaluation Logic
        all_metrics = []
        self.to(device)
        self.eval()
        data = data.to(device)
        
        # Prepare ground truth (unstandardize if needed to match predictions)
        y_train_full = data.clone()
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            y_train_full = y_train_full * std + mean
            
        # Ground truth for metrics ignores the first p_lag steps
        y_true = y_train_full[self.p_lag:]

        for _ in range(n_repeats):
            with torch.no_grad():
                # Generate predictions mapping to the full dataset
                full_in_sample_preds = self.predict_in_sample(
                    data=data,
                    m_samples=m_samples,
                    method=method,
                    batch_size=batch_size,
                    unstandardize=unstandardize,
                    device=device
                )
                
                # Slice out the initial p_lag steps (which are NaNs)
                valid_forecast_ensemble = full_in_sample_preds[:, self.p_lag:, :, :]
                
                # Round and clamp because epidemic cases are non-negative integers
                valid_forecast_ensemble = torch.clamp(torch.round(valid_forecast_ensemble), min=0)
                y_preds_ensemble = valid_forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train_full)
                rmsse_val = rmsse(y_true, y_pred_point, y_train_full)

                rho_50 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5)
                rho_90 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9)

                lower_bound = torch.quantile(valid_forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(valid_forecast_ensemble, 0.975, dim=0)
                
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    def plot_in_sample_fit(self,
        in_sample_preds: torch.Tensor,
        original_data: torch.Tensor,
        plots_per_row: int = 4,
        confidence_level: float = 0.95,
        node_names: typing.Optional[list] = None,
        title: str = 'MVEN: In-Sample Forecast Fit vs. Actual',
        savefig: bool = False,
        filename: typing.Optional[str] = None):
        """
        Plots in-sample forecasted time series against actual ground truth values.

        This method generates a grid of subplots (one per node) showing the median 
        forecast, the ground truth, and a shaded prediction interval based on the 
        stochastic ensemble.

        Args:
            in_sample_preds (torch.Tensor): Prediction ensemble from 
                :meth:`predict_in_sample` of shape :math:`(M, T, N, D)`.
            original_data (torch.Tensor): Ground truth dataset of shape 
                :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display per row 
                in the figure grid. Defaults to 4.
            confidence_level (float, optional): The width of the shaded prediction 
                interval (e.g., ``0.95`` for a 95% interval). Defaults to 0.95.
            node_names (list of str, optional): Custom labels for each node. If 
                None, nodes are labeled by index. Defaults to None.
            title (str, optional): The main title for the entire figure.
            savefig (bool, optional): If True, the resulting figure is exported 
                to a file. Defaults to False.
            filename (str, optional): The file path/name for saving the figure 
                (e.g., 'fit_plot.png'). Required if ``savefig`` is True.

        Note:
            The shaded area represents the uncertainty captured by the Engression 
            noise injection. The bounds are calculated as the 
            :math:`(1 - confidence\_level)/2` and :math:`(1 + confidence\_level)/2` 
            quantiles of the ensemble.

        Returns:
            None: This method displays the plot using ``plt.show()`` or saves it to disk.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if in_sample_preds.dim() == 4:
            in_sample_preds = in_sample_preds.squeeze(-1)  # Shape: (m_samples, T, N)
        if original_data.dim() == 3:
            original_data = original_data.squeeze(-1)      # Shape: (T, N)
            
        m_samples, T, num_nodes = in_sample_preds.shape
        p = self.p_lag
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 4 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        # Calculate dynamic quantiles based on confidence_level
        alpha = 1.0 - confidence_level
        q_lower = alpha / 2.0
        q_upper = 1.0 - (alpha / 2.0)
        
        # Isolate the valid predictions (ignoring the first p_lag NaNs) to compute statistics safely
        valid_preds = in_sample_preds[:, p:, :]
        
        # Initialize full tensors with NaNs to maintain temporal alignment
        forecast_median = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        lower_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        upper_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        
        # Compute statistics only on the valid temporal segment and apply clamping/rounding
        forecast_median[p:] = torch.clamp(torch.round(torch.median(valid_preds, dim=0).values), min=0)
        lower_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_lower, dim=0), min=0)
        upper_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_upper, dim=0), min=0)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Ground truth line
            ax.plot(timesteps, original_data[:, node].cpu(), marker='o', 
                    label='Actual', markersize=4, color='black')
            
            # Median forecast line (NaNs will cause matplotlib to skip the first p_lag steps automatically)
            ax.plot(timesteps, forecast_median[:, node].cpu(), marker='+', 
                    label='Fitted', color='blue')
            
            # Shaded Confidence Interval
            ax.fill_between(timesteps,
                            lower_bound[:, node].cpu(),
                            upper_bound[:, node].cpu(),
                            color='gray', alpha=0.8, label=f'{int(confidence_level*100)}% CI')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            plt.savefig(filename, format='pdf')
            
        plt.show()

    def plot_residuals(self,
        residuals: torch.Tensor,
        plots_per_row: int = 4,
        node_names: typing.Optional[list] = None,
        title: str = 'MVEN: In-Sample Residuals per Node',
        savefig: bool = False,
        filename: typing.Optional[str] = None):
        """
        Plots the time series of residuals (errors) for each node in a grid.

        This visualization helps in identifying systematic biases or patterns in 
        the model's errors across different spatial nodes. A horizontal line at 
        zero is included for reference.

        Args:
            residuals (torch.Tensor): Residual matrix obtained from 
                :meth:`get_residuals`, with shape :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display in 
                each row of the grid. Defaults to 4.
            node_names (list of str, optional): Labels for each node subplot. 
                If None, node indices are used. Defaults to None.
            title (str, optional): The main title for the figure. 
                Defaults to 'MVEN: In-Sample Residuals per Node'.
            savefig (bool, optional): If True, the plot will be saved to the 
                specified ``filename``. Defaults to False.
            filename (str, optional): Path where the figure should be saved. 
                Required if ``savefig`` is True. Defaults to None.

        Note:
            Any ``NaN`` values present in the residuals (typically the first 
            ``p_lag`` steps) are automatically handled by the plotting 
            backend and will appear as gaps in the time series.

        Returns:
            None: Displays the plot using ``plt.show()`` or saves the file.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if residuals.dim() == 3:
            residuals = residuals.squeeze(-1)  # Shape: (T, N)
            
        T, num_nodes = residuals.shape
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        # Slightly shorter figure height per row since residuals usually need less vertical space
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 3 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Plot the residual line (NaNs from the first p_lag steps are safely ignored by matplotlib)
            ax.plot(timesteps, residuals[:, node].cpu(), marker='o', 
                    linestyle='-', markersize=3, color='purple', alpha=0.7, label='Residual')
            
            # Add a horizontal zero line for easy visual bias checking
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            # Saving as PDF is generally preferred for high-quality manuscript figures
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            
        plt.show()

# 3. Spatio-Temporal Engression Network: STEN
class STARLayer(nn.Module):
    """A differentiable neural layer inspired by the STARMA model's spatial component.

    This layer computes a spatial embedding by aggregating features from neighbors
    at multiple spatial lags (distances). It transforms node features through 
    a combination of learnable weights and fixed spatial weight matrices.

    Args:
        in_features (int): Number of input features for each node (D).
        out_features (int): Number of output features for each node (D').
        max_spatial_lag (int): The maximum spatial lag (L) to consider.

    Note:
        The forward pass expects a list of spatial weight matrices (W_list), 
        where each matrix represents a specific spatial lag.
    """
    def __init__(self, in_features, out_features, max_spatial_lag, spatial_seed=21):
        super(STARLayer, self).__init__()
        self.max_spatial_lag = max_spatial_lag
        self.spatial_seed = spatial_seed
        # Create a list of linear layers, one for each spatial lag (from 0 to L).
        # Each layer learns a transformation matrix Phi_l.
        torch.manual_seed(self.spatial_seed)
        self.spatial_lag_layers = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) for _ in range(max_spatial_lag + 1)
        ])
        self.activation = nn.ReLU()

    def forward(self, x, W_list):
        """
        Forward pass for the STAR-Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_nodes, in_features).
            W_list (list of torch.Tensor): A list of spatial weight matrices.

        Returns:
            torch.Tensor: The spatial embedding tensor of shape (batch, seq_len, num_nodes, out_features).

        Shapes:
            - **Input**: :math:`(B, T, N, D_{in})` where :math:`N` is num_nodes, :math:`B` is batch_size, :math:`T` is sequence_length, and :math:`D_{in}` is in_feat.
            - **Output**: :math:`(B, T, N, D_{out})` where :math:`D_{out}` is :math:`D'`.
        """
        # Ensure the number of weight matrices matches the expected number of lags.
        assert len(W_list) == self.max_spatial_lag + 1, "Number of weight matrices must match max_spatial_lag + 1"

        # List to store the output of each spatial lag's transformation.
        spatial_embeddings = list()

        # Loop through each spatial lag from 0 to L.
        for l in range(self.max_spatial_lag + 1):
            # Get the spatial weight matrix for the current lag.
            Wl = W_list[l].to(x.device).float()
            
            # Apply the spatial aggregation: W^l * X_t
            # Reshape x for matrix multiplication: (batch * seq_len, num_nodes, in_features)
            batch_size, seq_len, num_nodes, _ = x.shape
            x_reshaped = x.reshape(batch_size * seq_len, num_nodes, -1).to(x.device).float()
            spatially_lagged_x = torch.bmm(Wl.unsqueeze(0).expand(batch_size * seq_len, -1, -1), x_reshaped)
            
            # Reshape back to original batch and sequence dimensions.
            spatially_lagged_x = spatially_lagged_x.reshape(batch_size, seq_len, num_nodes, -1)

            # Apply the learnable linear transformation: (W^l * X_t) * Phi_l
            transformed_x = self.spatial_lag_layers[l](spatially_lagged_x)
            spatial_embeddings.append(transformed_x)

        # Sum the embeddings from all spatial lags.
        aggregated_embedding = torch.stack(spatial_embeddings, dim=0).sum(dim=0)

        # Apply the final activation function.
        output = self.activation(aggregated_embedding)
        return output


class _EngressionLSTM(nn.Module):
    """
    Engression-LSTM module for STEN.
    """
    def __init__(self, in_feat_dim, lstm_input_dim, num_nodes, lstm_hidden_dim, lstm_num_layers, lstm_dropout, p_lag, t_pred,
                noise_encode="add", noise_dist="gaussian", noise_dim=2, noise_std=1, temporal_seed=9):
        super().__init__()
        self.p_lag = p_lag # input_seq_len or past lagged values to use
        self.t_pred = t_pred # output_seq_len or forecast_horizon
        self.in_feat_dim = in_feat_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.num_nodes = num_nodes
        self.noise_dist = noise_dist # gaussian or uniform
        self.noise_encode = noise_encode # add or concat
        self.noise_dim = noise_dim # only required when noise_encode = concat
        self.noise_std = noise_std
        self.temporal_seed = temporal_seed

        if self.noise_encode=="add":
            pass
        elif self.noise_encode=="concat":
            lstm_input_dim += self.noise_dim
        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")
        
        
        torch.manual_seed(self.temporal_seed)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=False, # Input shape will be (seq_len, batch, input_size)
            num_layers = lstm_num_layers,
            dropout=lstm_dropout
        )

        # Output Module: Fully connected layer to generate forecasts
        self.output_layer = nn.Linear(lstm_hidden_dim, t_pred * in_feat_dim)

    def forward(self, x: torch.Tensor):
        # Input x shape: (batch_size, input_seq_len, N, D)

        batch_size, seq_len = x.shape[0], x.shape[1]
        if self.noise_encode == "add":
            if self.noise_dist == "uniform":
                noise = torch.rand_like(x) * self.noise_std # Sample from U(0,1)
            elif self.noise_dist == "gaussian":
                noise = torch.randn_like(x) * self.noise_std # Sample from N(0,1)
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            x = x + noise

        elif self.noise_encode == "concat":
            noise_shape = (batch_size, seq_len, self.num_nodes, self.noise_dim)
            if self.noise_dist == "gaussian":
                noise = torch.randn(noise_shape) * self.noise_std
                noise = noise.to(x.device)
            elif self.noise_dist == "uniform":
                noise = torch.rand(noise_shape) * self.noise_std
                noise = noise.to(x.device) 
            else:
                raise ValueError(f"Unexpected value for noise_dist = {self.noise_dist}, only `gaussian` and `uniform` are allowed.")
            x = torch.cat([x, noise], dim=-1)

        else:
            raise ValueError(f"Unexpected value for noise_encode = {self.noise_encode}, only `add` and `concat` are allowed.")

            
        
        # Sequential Processing with LSTM        
        lstm_input = x.permute(1, 0, 2, 3) # (input_seq_len, batch_size, N, D)
        batch_size, in_feat_dim = lstm_input.shape[1], lstm_input.shape[3]
        
        # Reshape for LSTM: (input_seq_len, batch_size * N, gcn_out_feat)
        lstm_input = lstm_input.reshape(self.p_lag, batch_size * self.num_nodes, in_feat_dim)
        
        # LSTM processes the sequence of p lagged values
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Get the last hidden state: (batch_size * N, lstm_hidden_dim)
        last_hidden_state = h_n[-1]

        # Forecasting with Output Layer
        # Output: (batch_size * N, t_pred * D)
        predictions = self.output_layer(last_hidden_state)
        
        # Reshape to final forecast shape: (batch_size, N, t_pred, D)
        predictions = predictions.view(batch_size, self.num_nodes, self.t_pred, self.in_feat_dim)
        
        # Permute to match target Y shape: (batch_size, t_pred, N, D)
        predictions = predictions.permute(0, 2, 1, 3)

        return predictions

class STEN(nn.Module):
    """Spatio-Temporal Engression Network.

    Combines the STAR-Layer and Engression-LSTM for probabilistic 
    spatiotemporal forecasting.

    Args:
        in_feat_dim (int): Number of input features for each node (D).
        embedding_dim (int): The dimension of the spatial embedding (D').
        max_spatial_lag (int): The maximum spatial lag (L) for the STAR-Layer.
        p_lag (int): Number of past timesteps used as input (input sequence length).
        t_pred (int): Number of future timesteps predicted (forecast horizon).
        lstm_hidden_dim (int): Hidden dimension size of the LSTM temporal module.
        lstm_num_layers (int): Number of layers in the LSTM.
        lstm_dropout (float): Dropout probability in the LSTM.
        noise_dist (str): Distribution type for noise injection ('gaussian' or 'uniform').
        noise_encode (str): Method for noise injection: 'add' (additive) or 'concat' (concatenation).
        noise_dim (int): Dimension of noise features if concatenated.
        noise_std (int): Standard deviation/scaling of the noise.
        num_nodes (int): Number of nodes in the graph.
        temporal_seed (int): Seed for reproducibility in LSTM initialization.

    Example:
        >>> model = STEN(
        ...     in_feat_dim=1, embedding_dim=8, lstm_hidden_dim=32, 
        ...     lstm_num_layers=2, lstm_dropout=0.1, p_lag=12, t_pred=4
        ... )
        >>> model.fit(
        ...     data_loader, optimizer, loss_fn=energy_score_loss,
        ...     num_epochs=100, m_samples=2, device=device, 
        ...     visualize=True, W_list=W_list
        ... )
        >>> output = model(history_tensor, W_list=W_list)
        >>> forecast_samples = model.predict(
        ...     history_tensor, m_samples=100, device=device, W_list=W_list
        ... )
        >>> # In-sample analysis
        >>> in_sample_preds = model.predict_in_sample(
        ...     train_data, m_samples=100, method="q_step", W_list=W_list
        ... )
        >>> residuals = model.get_residuals(
        ...     in_sample_preds, train_data, point_method="median"
        ... )
        >>> model.plot_residuals(residuals, plots_per_row=4)
        >>> # Out-of-sample evaluation
        >>> metrics_df = model.evaluate_forecasts(
        ...     history_tensor, y_test, y_train, 
        ...     point_method="median", W_list=W_list
        ... )
        >>> print(metrics_df)
    """
    def __init__(self, in_feat_dim, num_nodes, embedding_dim, max_spatial_lag, lstm_hidden_dim, lstm_num_layers, lstm_dropout, p_lag, t_pred,
                 noise_encode="add", noise_dist="gaussian", noise_dim=2, noise_std=1, temporal_seed=9):
        super(STEN, self).__init__()
        self.p_lag = p_lag
        self.in_feat_dim = in_feat_dim
        self.t_pred = t_pred
        self.num_nodes = num_nodes
        self.star_layer = STARLayer(in_feat_dim, embedding_dim, max_spatial_lag)
        self.engression_lstm = _EngressionLSTM(in_feat_dim, embedding_dim, num_nodes, lstm_hidden_dim, lstm_num_layers, lstm_dropout, p_lag, t_pred,
                noise_encode, noise_dist, noise_dim, noise_std, temporal_seed)

    def forward(self, x, W_list):
        """
        Forward pass of STEN for generating a single stochastic forecast.

        Args:
            x (torch.Tensor): Input tensor of shape :math:`(B, T_{in}, N, D_{in})`, 
                where :math:`B` is batch size, :math:`T_{in}` is ``p_lag``, 
                :math:`N` is ``num_nodes``, and :math:`D_{in}` is ``in_feat_dim``.
            W_list (list): List of spatial weights matrices.

        Returns:
            torch.Tensor: Forecasted tensor of shape :math:`(B, T_{out}, N, D_{in})`, where :math:`T_{out}` is ``t_pred``.
        """
        # 1. Get spatial embeddings from the STAR-Layer.
        spatial_embeddings = self.star_layer(x, W_list)
        
        # 2. Get probabilistic forecast from the Engression-LSTM.
        forecast = self.engression_lstm(spatial_embeddings)
        
        return forecast

    def fit(self, data_loader, optimizer, loss_fn, W_list, num_epochs=100, m_samples=2, device="cpu", 
            monitor=True, visualize=True, verbose=False):
        """
        Trains the STEN model using a dataloader, optimizer, and probabilistic loss function.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch DataLoader yielding ``(x_batch, y_batch)``.
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
            loss_fn (callable): Loss function accepting ``(target, pred_samples)``.
            W_list (list): List of spatial weights matrices.
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            m_samples (int, optional): Number of stochastic samples per batch 
                to estimate the probabilistic loss. Defaults to 2.
            device (str, optional): Device to run training on (``'cpu'`` or ``'cuda'``).
            monitor (bool, optional): If True, shows a progress bar.
            visualize (bool, optional): If True, plots the loss curve after training.
            verbose (bool, optional): If True, prints periodic loss updates.
        """
        self.train()
        if visualize:
            losses = []
        epoch_iter = (tqdm(range(num_epochs), desc="Training", unit="epoch", leave=True) 
                      if monitor else range(num_epochs))
        for epoch in epoch_iter:
            total_loss = 0
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = []
                for _ in range(m_samples):
                    pred = self(x_batch, W_list)
                    predictions.append(pred)
                predictions_tensor = torch.stack(predictions, dim=0)
                loss = loss_fn(y_batch, predictions_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            if visualize:
                losses.append(avg_loss)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        if visualize:
            epochs = np.arange(1, num_epochs+1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, losses)
            plt.title("Training loss")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()
        if verbose:
            print("Training finished.")

    def predict(self, history: torch.Tensor, W_list, m_samples: int = 100, unstandardize=None, device=None) -> torch.Tensor:
        """
        Generates an ensemble of forecasts for a single historical observation.

        Args:
            history (torch.Tensor): Past observations of shape :math:`(T_{in}, N, D_{in})`.
            W_list (list): List of spatial weights matrices.
            m_samples (int, optional): Number of ensemble members to generate. 
                Defaults to 100.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Computation device. Defaults to ``'cpu'``.

        Returns:
            torch.Tensor: Ensemble of forecasts of shape :math:`(M, T_{out}, N, D_{in})`,  where :math:`M` is ``m_samples``.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        forecast_ensemble = []
        history = history.to(device)
        with torch.no_grad():
            for _ in range(m_samples):
                # Add batch dimension: (1, p_lag, N, D)
                history_batch = history.unsqueeze(0)
                # Generate forecast sample
                prediction = self(history_batch, W_list)  # shape: (batch_size, t_pred, N, D)
                prediction = prediction.squeeze(0).to(device) # remove batch dimension 
                if unstandardize is not None:
                    mean, std = unstandardize[0].to(device), unstandardize[1].to(device)
                    prediction = prediction * std + mean
                forecast_ensemble.append(prediction)
        return torch.stack(forecast_ensemble, dim=0)

    def evaluate_forecasts(self, history: torch.Tensor, y_true: torch.Tensor, y_train: torch.Tensor, W_list,
                           m_samples=100, n_repeats=50, point_method: typing.Union[str, float] = "median",
                             unstandardize=None, device=None):
        """
        Repeatedly generate probabilistic forecasts from a single trained model 
        and return summary metrics.

        This method performs Monte Carlo style evaluation by generating multiple 
        ensembles to account for the stochastic nature of the engression model.

        Args:
            history (torch.Tensor): Last ``p_lag`` observations of shape 
                :math:`(T_{in}, N, D_{in})`.
            y_true (torch.Tensor): Ground truth for the forecast horizon of shape 
                :math:`(T_{out}, N, D_{in})`.
            y_train (torch.Tensor): In-sample training data of shape 
                :math:`(T_{train}, N, D_{in})`. Required for scaling MASE and RMSSE.
            W_list (list): List of spatial weights matrices.
            m_samples (int, optional): Number of samples per forecast ensemble. 
                Defaults to 100.
            n_repeats (int, optional): Number of times to repeat the ensemble 
                generation to calculate metric stability. Defaults to 50.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (torch.device or str, optional): Device for evaluation. 
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the mean and standard deviation 
            across repeats for the following metrics:
            
            * **Point**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration**: Empirical Coverage, Winkler Score.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            # CRPS approx as mean absolute error between samples and true values minus half the mean pairwise absolute differences
            # CRPS = E|X - y| - 0.5 E|X - X'|
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            # Calculate MAE for forecast period across all elements
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))

            # Calculate scaling factor: mean absolute difference of y_train across all elements
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            mase_score = mae_forecast / scale

            return mase_score.item()

        def rmsse(y_true, y_pred, y_train):
            # Calculate MSE for forecast period across all elements
            mse_forecast = torch.mean((y_true - y_pred)**2)

            # Calculate scaling factor: mean squared difference of y_train across all elements
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)

            # Avoid division by zero
            scale = scale if scale != 0 else 1e-8

            rmsse_score = torch.sqrt(mse_forecast / scale)

            return rmsse_score.item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            """
            Calculates the rho-risk (normalized quantile loss) as defined in the DeepAR paper.
            
            Args:
                ground_truth (np.array): Shape (H, N)
                                         H = Forecast Horizon
                                         N = Number of Nodes
                forecast_ensemble (np.array): Shape (M, H, N, D)
                                              M = Ensemble samples
                                              D = Feature dimension (expected to be 1)
                rho (float): The quantile to evaluate (e.g., 0.5 or 0.9).
                
            Returns:
                float: The calculated rho-risk value.
            """

            # Squeeze the feature dimension D=1 to get shape (M, H, N)
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
            else:
                # Fallback if D > 1
                forecasts = forecast_ensemble
                
            # Calculate the Empirical Quantile (Z_hat)
            # We collapse the ensemble dimension (M) to get the rho-quantile prediction
            # resulting shape: (H, N)
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            
            # Calculate the Quantile Loss (L_rho) element-wise
            # The DeepAR paper uses a specific formula with a scaling factor of 2.
            # We use boolean masking to handle the asymmetric penalty correctly.
            
            # Mask for Underestimation: True where Ground Truth > Prediction
            under_bias = ground_truth > forecast_quantiles
            
            # Mask for Overestimation: True where Ground Truth <= Prediction
            over_bias = ~under_bias
            
            # Initialize loss array
            losses = np.zeros_like(ground_truth)
            
            # Apply penalty for Underestimation: 2 * rho * (Z - Z_hat)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            
            # Apply penalty for Overestimation: 2 * (1 - rho) * (Z_hat - Z)
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            # Aggregation and Normalization
            # The paper defines rho-risk as the sum of quantile losses divided by the sum of target values.
            # Summation occurs over the entire horizon (H) and all nodes (N).
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            
            rho_risk = total_loss / total_target
            
            return rho_risk

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            """
            Calculate empirical coverage probability that true values lie within predicted CIs.
        
            Args:
                true_values (array-like): True target values.
                lower_bounds (array-like): Lower bounds of predicted confidence intervals.
                upper_bounds (array-like): Upper bounds of predicted confidence intervals.
        
            Returns:
                float: Fraction of true values inside the confidence intervals.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            coverage = np.mean(inside)
            return coverage


        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            """
            Calculate average Winkler score for prediction intervals.
        
            Parameters:
            true_values (array-like): Ground truth values.
            lower_bounds (array-like): CI lower bounds
            upper_bounds (array-like): CI upper bounds.
            alpha (float): Significance level (e.g. 0.05 for 95% CI).
        
            Returns:
            float: Average Winkler score.
            """
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
        
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
        
            below = true_values < lower_bounds
            above = true_values > upper_bounds
        
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
        
            return np.mean(scores)

        all_metrics = []
        y_true = y_true.to(device)
        history = history.to(device)
        y_train = y_train.to(device)
        self.to(device)
        self.eval()

        for _ in range(n_repeats):
            with torch.no_grad():
                forecast_ensemble = self.predict(
                    history=history,
                    m_samples=m_samples,
                    unstandardize=unstandardize,
                    W_list=W_list,
                    device=device
                )
                forecast_ensemble = torch.round(forecast_ensemble) # Since epidemic incidence cases can be integers only
                y_preds_ensemble = forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float quantile (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train)
                rmsse_val = rmsse(y_true, y_pred_point, y_train)

                rho_50 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5).item()
                rho_90 = calculate_rho_risk(y_true.squeeze(-1).cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9).item()

                lower_bound = torch.quantile(forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(forecast_ensemble, 0.975, dim=0)
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                    
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    # The following methods will be useful for diagnosis of in-sample predictions.
    
    def predict_in_sample(self, data: torch.Tensor, W_list, m_samples: int = 100, method: str = "1_step", batch_size: int = 64, 
                          unstandardize: typing.Optional[list] = None, device: str = "cpu") -> torch.Tensor:
        """
        Generates in-sample predictions for the entire training dataset.

        This method applies a sliding window across the provided historical data 
        to produce stochastic forecasts for every possible time step.

        Args:
            data (torch.Tensor): The original dataset of shape :math:`(T, N, D)`, 
                where :math:`T` is the total time steps.
            W_list (list): List of spatial weights matrices.
            m_samples (int, optional): Number of stochastic forecast samples 
                to generate per window. Defaults to 100.
            method (str, optional): Strategy for in-sample forecasting. 
                Options include:
                
                * ``"1_step"``: Slides the input window by 1 step, recording 
                  only the 1-step ahead forecast for each position.
                * ``"q_step"``: Slides by ``t_pred`` (q) steps, recording 
                  full non-overlapping forecast horizons.
                  
                Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows to process 
                simultaneously to optimize memory. Defaults to 64.
            unstandardize (list, optional): A list ``[mean, std]`` to reverse 
                data normalization. Defaults to None.
            device (str, optional): Device to perform computations on. 
                Defaults to ``"cpu"``.

        Returns:
            torch.Tensor: In-sample forecast ensemble of shape :math:`(M, T, N, D)`, where :math:`M` is ``m_samples``.

        Note:
            The first ``p_lag`` time steps in the output tensor will contain 
            ``NaN`` values because there is insufficient historical context 
            to generate a forecast for the beginning of the sequence.
        """
        self.eval()
        self.to(device)
        data = data.to(device)
        if data.ndim < 3:
            data = data.unsqueeze(-1)
        T, N, D = data.shape
        p = self.p_lag
        q = self.t_pred
        
        # Pre-allocate the full prediction tensor with NaNs to match (T, N, D)
        # Shape: (m_samples, T, N, D)
        full_preds = torch.full((m_samples, T, N, D), float('nan'), device=device)
        
        # Determine the starting indices for our sliding windows based on the chosen method
        if method == "1_step":
            start_indices = list(range(0, T - p))
        elif method == "q_step":
            start_indices = list(range(0, T - p, q))
        else:
            raise ValueError("Method must be either '1_step' or 'q_step'")

        with torch.no_grad():
            for batch_start in range(0, len(start_indices), batch_size):
                batch_indices = start_indices[batch_start : batch_start + batch_size]
                
                # Shape: (B, p, N, D)
                batch_in = torch.stack([data[i : i + p] for i in batch_indices])
                
                # Generate m_samples for the current batch
                batch_samples = []
                for _ in range(m_samples):
                    # self(x) returns predictions of shape (B, q, N, D)
                    batch_samples.append(self(batch_in, W_list))
                
                # Shape: (m_samples, B, q, N, D)
                batch_samples = torch.stack(batch_samples, dim=0)
                
                # Place the predictions into the correct temporal alignment in the full tensor
                for b_idx, i in enumerate(batch_indices):
                    if method == "1_step":
                        # Record only the 1-step ahead forecast (index 0 of the q dimension)
                        # Target time index is exactly i + p
                        full_preds[:, i + p, :, :] = batch_samples[:, b_idx, 0, :, :]
                        
                    elif method == "q_step":
                        # Record the full q-step window
                        # Target time indices are i + p to i + p + q
                        end_idx = min(i + p + q, T)
                        valid_q = end_idx - (i + p) # Handle boundary condition if T isn't perfectly divisible
                        full_preds[:, i + p : end_idx, :, :] = batch_samples[:, b_idx, :valid_q, :, :]
                        
        # Apply unstandardization if parameters are provided
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            # Broadcasting matches the trailing (N, D) dimensions of full_preds
            # NaNs in the first p_lag steps will remain NaN
            full_preds = full_preds * std + mean
        return full_preds

    def get_residuals(self, in_sample_preds: torch.Tensor, original_data: torch.Tensor,
                     point_method: typing.Union[str, float] = "median") -> torch.Tensor:
        """
        Computes the residual matrix from in-sample predictions.

        This method reduces the stochastic ensemble into a single point forecast 
        using the specified ``point_method`` and calculates the error: 
        :math:`Residual = Actual - Predicted`.

        Args:
            in_sample_preds (torch.Tensor): Predictions from :meth:`predict_in_sample` 
                with shape :math:`(M, T, N, D)`, where :math:`M` is the number 
                of samples.
            original_data (torch.Tensor): The ground truth dataset of shape 
                :math:`(T, N, D)`.
            point_method (str or float, optional): Method to extract a point 
                forecast from the ensemble. Options: ``"median"``, ``"mean"``, 
                or a float representing a quantile (e.g., ``0.75``). 
                Defaults to ``"median"``.

        Returns:
            torch.Tensor: Residual tensor of shape :math:`(T, N, D)`.

        Note:
            Following the structure of the in-sample predictions, the first 
            ``p_lag`` time steps will contain ``NaN`` values. These represent 
            the "warm-up" period where no forecasts were generated.
        """
        original_data = original_data.to(in_sample_preds.device)
        
        # Calculate the point predictions over the ensemble dimension (dim=0)
        # Shape: (T, N, D)
        if point_method == "mean":
            point_preds = torch.mean(in_sample_preds.float(), dim=0)
        elif point_method == "median":
            point_preds = torch.median(in_sample_preds, dim=0).values
        elif isinstance(point_method, (float, int)):
            # Extract a specific quantile
            point_preds = torch.quantile(in_sample_preds.float(), q=float(point_method), dim=0)
        else:
            raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")
        
        # Compute residuals (Actual - Predicted)
        # NaNs in median_preds will naturally propagate to the residuals
        residuals = original_data - point_preds
        
        return residuals

    def evaluate_in_sample_fit(
        self, 
        data: torch.Tensor, 
        W_list,
        m_samples: int = 100, 
        n_repeats: int = 10,
        method: str = "1_step",
        batch_size: int = 32,
        point_method: typing.Union[str, float] = "median",
        unstandardize: typing.Optional[list] = None, 
        device: typing.Optional[str] = None):
        """
        Repeatedly generates in-sample probabilistic forecasts and returns summary metrics.

        This method assesses the model's ability to reconstruct the historical training 
        sequence by performing multiple stochastic passes over the data. It accounts 
        for the "Engression" noise injection by repeating the process ``n_repeats`` 
        times and reporting the stability of the metrics.

        Args:
            data (torch.Tensor): The full historical dataset of shape 
                :math:`(T, N, D_{in})`, where :math:`T` is total time steps.
            W_list (list): List of spatial weights matrices.
            m_samples (int, optional): Number of stochastic samples per forecast 
                ensemble window. Defaults to 100.
            n_repeats (int, optional): Number of full evaluation cycles to 
                perform to calculate mean/std of metrics. Defaults to 10.
            method (str, optional): The sliding window strategy: ``"1_step"`` 
                or ``"q_step"``. Defaults to ``"1_step"``.
            batch_size (int, optional): Number of windows processed in a single 
                forward pass. Defaults to 32.
            point_method (str or float, optional): Metric for extracting a single 
                forecast from the ensemble to compute point errors (SMAPE, MAE, 
                RMSE, MASE, RMSSE). Accepts ``"median"``, ``"mean"``, or a 
                float quantile (e.g., ``0.75``). Defaults to ``"median"``.
            unstandardize (list, optional): A list ``[mean, std]`` used to 
                rescale predictions and ground truth to original units. 
                Defaults to None.
            device (str or torch.device, optional): Computation device. 
                Defaults to None.

        Returns:
            pd.DataFrame: A summary table containing the mean and standard 
            deviation for the following metrics across all repeats:
            
            * **Point Metrics**: SMAPE, MAE, RMSE, MASE, RMSSE.
            * **Probabilistic Metrics**: Pinball (80%, 95%), Rho-risk (0.5, 0.9), CRPS.
            * **Calibration Metrics**: Empirical Coverage, Winkler Score.

        Note:
            Similar to :meth:`predict_in_sample`, metrics are calculated only 
            for the time steps after the initial ``p_lag`` warm-up period.
        """
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def pinball_loss(y_true, y_pred_quantile, quantile):
            error = y_true - y_pred_quantile
            return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

        def crps_approximation(y_true, y_preds_ensemble):
            abs_diff_true = torch.mean(torch.abs(y_preds_ensemble - y_true), dim=0)
            abs_diff_samples = torch.mean(torch.abs(y_preds_ensemble.unsqueeze(0) - y_preds_ensemble.unsqueeze(1)), dim=(0,1))
            crps = torch.mean(abs_diff_true - 0.5 * abs_diff_samples)
            return crps.item()

        def mase(y_true, y_pred, y_train):
            mae_forecast = torch.mean(torch.abs(y_true - y_pred))
            diff = torch.abs(y_train[1:] - y_train[:-1])
            scale = torch.mean(diff)
            scale = scale if scale != 0 else 1e-8
            return (mae_forecast / scale).item()

        def rmsse(y_true, y_pred, y_train):
            mse_forecast = torch.mean((y_true - y_pred)**2)
            diff = y_train[1:] - y_train[:-1]
            scale = torch.mean(diff ** 2)
            scale = scale if scale != 0 else 1e-8
            return torch.sqrt(mse_forecast / scale).item()

        def calculate_rho_risk(ground_truth, forecast_ensemble, rho):
            if forecast_ensemble.shape[-1] == 1:
                forecasts = forecast_ensemble.squeeze(-1)
                ground_truth = ground_truth.squeeze(-1)
            else:
                forecasts = forecast_ensemble
                
            forecast_quantiles = np.quantile(forecasts, rho, axis=0)
            under_bias = ground_truth > forecast_quantiles
            over_bias = ~under_bias
            
            losses = np.zeros_like(ground_truth)
            losses[under_bias] = 2 * rho * (ground_truth[under_bias] - forecast_quantiles[under_bias])
            losses[over_bias] = 2 * (1 - rho) * (forecast_quantiles[over_bias] - ground_truth[over_bias])
            
            total_loss = np.sum(losses)
            total_target = np.sum(np.abs(ground_truth)) 
            return total_loss / total_target if total_target != 0 else 0.0

        def empirical_coverage(true_values, lower_bounds, upper_bounds):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            inside = (true_values >= lower_bounds) & (true_values <= upper_bounds)
            return np.mean(inside)

        def winkler_score(true_values, lower_bounds, upper_bounds, alpha):
            true_values = np.array(true_values)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            widths = upper_bounds - lower_bounds
            scores = widths.copy()
            
            below = true_values < lower_bounds
            above = true_values > upper_bounds
            
            scores[below] += (2 / alpha) * (lower_bounds[below] - true_values[below])
            scores[above] += (2 / alpha) * (true_values[above] - upper_bounds[above])
            return np.mean(scores)

        # Evaluation Logic
        all_metrics = []
        self.to(device)
        self.eval()
        data = data.to(device)
        
        # Prepare ground truth (unstandardize if needed to match predictions)
        y_train_full = data.clone()
        if unstandardize is not None:
            mean = unstandardize[0].to(device)
            std = unstandardize[1].to(device)
            y_train_full = y_train_full * std + mean
            
        # Ground truth for metrics ignores the first p_lag steps
        y_true = y_train_full[self.p_lag:]

        for _ in range(n_repeats):
            with torch.no_grad():
                # Generate predictions mapping to the full dataset
                full_in_sample_preds = self.predict_in_sample(
                    data=data,
                    m_samples=m_samples,
                    method=method,
                    batch_size=batch_size,
                    unstandardize=unstandardize,
                    device=device, W_list=W_list
                )
                
                # Slice out the initial p_lag steps (which are NaNs)
                valid_forecast_ensemble = full_in_sample_preds[:, self.p_lag:, :, :]
                
                # Round and clamp because epidemic cases are non-negative integers
                valid_forecast_ensemble = torch.clamp(torch.round(valid_forecast_ensemble), min=0)
                y_preds_ensemble = valid_forecast_ensemble.to(device)

                if point_method == "mean":
                    y_pred_point = torch.mean(y_preds_ensemble.float(), dim=0)
                elif point_method == "median":
                    y_pred_point = torch.median(y_preds_ensemble, dim=0).values
                elif isinstance(point_method, (float, int)):
                    # Extract a specific quantile
                    y_pred_point = torch.quantile(y_preds_ensemble.float(), q=float(point_method), dim=0)
                else:
                    raise ValueError("point_method must be 'mean', 'median', or a float (e.g., 0.75).")

                mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
                rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
                numerator = torch.abs(y_pred_point - y_true)
                denominator = (torch.abs(y_pred_point) + torch.abs(y_true)) / 2
                smape = torch.mean(numerator / denominator.clamp(min=1e-8)) * 100

                q80_forecast = torch.quantile(y_preds_ensemble, 0.80, dim=0)
                pinball_80 = pinball_loss(y_true, q80_forecast, 0.80).item()

                q95_forecast = torch.quantile(y_preds_ensemble, 0.95, dim=0)
                pinball_95 = pinball_loss(y_true, q95_forecast, 0.95).item()

                crps = crps_approximation(y_true, y_preds_ensemble)

                mase_val = mase(y_true, y_pred_point, y_train_full)
                rmsse_val = rmsse(y_true, y_pred_point, y_train_full)

                rho_50 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.5)
                rho_90 = calculate_rho_risk(y_true.cpu().numpy(), y_preds_ensemble.cpu().numpy(), rho=0.9)

                lower_bound = torch.quantile(valid_forecast_ensemble, 0.025, dim=0)
                upper_bound = torch.quantile(valid_forecast_ensemble, 0.975, dim=0)
                
                ec = empirical_coverage(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu())
                ws = winkler_score(y_true.cpu(), lower_bound.cpu(), upper_bound.cpu(), alpha=0.05)
                
                all_metrics.append({
                    "SMAPE": smape.item(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MASE": mase_val,
                    "RMSSE": rmsse_val,
                    "Pinball_80": pinball_80,
                    "Pinball_95": pinball_95,
                    "Rho-0.5": rho_50,
                    "Rho-0.9": rho_90,
                    "CRPS": crps,
                    "EC": ec,
                    "Winkler": ws
                })

        metrics_df = pd.DataFrame(all_metrics)
        metrics_summary = metrics_df.agg(["mean", "std"])
        return metrics_summary.round(2)

    def plot_in_sample_fit(self, in_sample_preds: torch.Tensor,
        original_data: torch.Tensor, plots_per_row: int = 4,
        confidence_level: float = 0.95, node_names: typing.Optional[list] = None,
        title: str = 'STEN: In-Sample Forecast Fit vs. Actual',
        savefig: bool = False,
        filename: typing.Optional[str] = None):
        """
        Plots in-sample forecasted time series against actual ground truth values.

        This method generates a grid of subplots (one per node) showing the median 
        forecast, the ground truth, and a shaded prediction interval based on the 
        stochastic ensemble.

        Args:
            in_sample_preds (torch.Tensor): Prediction ensemble from 
                :meth:`predict_in_sample` of shape :math:`(M, T, N, D)`.
            original_data (torch.Tensor): Ground truth dataset of shape 
                :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display per row 
                in the figure grid. Defaults to 4.
            confidence_level (float, optional): The width of the shaded prediction 
                interval (e.g., ``0.95`` for a 95% interval). Defaults to 0.95.
            node_names (list of str, optional): Custom labels for each node. If 
                None, nodes are labeled by index. Defaults to None.
            title (str, optional): The main title for the entire figure. 
            savefig (bool, optional): If True, the resulting figure is exported 
                to a file. Defaults to False.
            filename (str, optional): The file path/name for saving the figure 
                (e.g., 'fit_plot.png'). Required if ``savefig`` is True.

        Note:
            The shaded area represents the uncertainty captured by the Engression 
            noise injection. The bounds are calculated as the 
            :math:`(1 - confidence\_level)/2` and :math:`(1 + confidence\_level)/2` 
            quantiles of the ensemble.

        Returns:
            None: This method displays the plot using ``plt.show()`` or saves it to disk.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if in_sample_preds.dim() == 4:
            in_sample_preds = in_sample_preds.squeeze(-1)  # Shape: (m_samples, T, N)
        if original_data.dim() == 3:
            original_data = original_data.squeeze(-1)      # Shape: (T, N)
            
        m_samples, T, num_nodes = in_sample_preds.shape
        p = self.p_lag
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 4 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        # Calculate dynamic quantiles based on confidence_level
        alpha = 1.0 - confidence_level
        q_lower = alpha / 2.0
        q_upper = 1.0 - (alpha / 2.0)
        
        # Isolate the valid predictions (ignoring the first p_lag NaNs) to compute statistics safely
        valid_preds = in_sample_preds[:, p:, :]
        
        # Initialize full tensors with NaNs to maintain temporal alignment
        forecast_median = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        lower_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        upper_bound = torch.full((T, num_nodes), float('nan'), device=in_sample_preds.device)
        
        # Compute statistics only on the valid temporal segment and apply clamping/rounding
        forecast_median[p:] = torch.clamp(torch.round(torch.median(valid_preds, dim=0).values), min=0)
        lower_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_lower, dim=0), min=0)
        upper_bound[p:] = torch.clamp(torch.quantile(valid_preds, q_upper, dim=0), min=0)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Ground truth line
            ax.plot(timesteps, original_data[:, node].cpu(), marker='o', 
                    label='Actual', markersize=4, color='black')
            
            # Median forecast line (NaNs will cause matplotlib to skip the first p_lag steps automatically)
            ax.plot(timesteps, forecast_median[:, node].cpu(), marker='+', 
                    label='Fitted', color='blue')
            
            # Shaded Confidence Interval
            ax.fill_between(timesteps,
                            lower_bound[:, node].cpu(),
                            upper_bound[:, node].cpu(),
                            color='gray', alpha=0.8, label=f'{int(confidence_level*100)}% CI')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            plt.savefig(filename, format='pdf')
            
        plt.show()

    def plot_residuals(self, residuals: torch.Tensor,
        plots_per_row: int = 4, node_names: typing.Optional[list] = None,
        title: str = 'STEN: In-Sample Residuals per Node',
        savefig: bool = False, filename: typing.Optional[str] = None):
        """
        Plots the time series of residuals (errors) for each node in a grid.

        This visualization helps in identifying systematic biases or patterns in 
        the model's errors across different spatial nodes. A horizontal line at 
        zero is included for reference.

        Args:
            residuals (torch.Tensor): Residual matrix obtained from 
                :meth:`get_residuals`, with shape :math:`(T, N, D)`.
            plots_per_row (int, optional): Number of subplots to display in 
                each row of the grid. Defaults to 4.
            node_names (list of str, optional): Labels for each node subplot. 
                If None, node indices are used. Defaults to None.
            title (str, optional): The main title for the figure. 
                Defaults to 'STEN: In-Sample Residuals per Node'.
            savefig (bool, optional): If True, the plot will be saved to the 
                specified ``filename``. Defaults to False.
            filename (str, optional): Path where the figure should be saved. 
                Required if ``savefig`` is True. Defaults to None.

        Note:
            Any ``NaN`` values present in the residuals (typically the first 
            ``p_lag`` steps) are automatically handled by the plotting 
            backend and will appear as gaps in the time series.

        Returns:
            None: Displays the plot using ``plt.show()`` or saves the file.
        """
        
        # Squeeze the feature dimension D (assuming D=1 for plotting)
        if residuals.dim() == 3:
            residuals = residuals.squeeze(-1)  # Shape: (T, N)
            
        T, num_nodes = residuals.shape
        
        num_rows = int(np.ceil(num_nodes / plots_per_row))
        # Slightly shorter figure height per row since residuals usually need less vertical space
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 3 * num_rows), 
                                sharex=True, sharey=False)
        
        if num_rows == 1 and plots_per_row == 1:
            axs = [axs]
        else:
            axs = np.array(axs).flatten()
            
        timesteps = np.arange(1, T + 1)
        
        for node in range(num_nodes):
            ax = axs[node]
            
            # Plot the residual line (NaNs from the first p_lag steps are safely ignored by matplotlib)
            ax.plot(timesteps, residuals[:, node].cpu(), marker='o', 
                    linestyle='-', markersize=3, color='purple', alpha=0.7, label='Residual')
            
            # Add a horizontal zero line for easy visual bias checking
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
            
            ax.set_title(node_names[node] if node_names else f'Node {node+1}')
            ax.grid(alpha=0.5)

        # Remove unused subplots if num_nodes doesn't perfectly fill the grid
        for i in range(num_nodes, len(axs)):
            fig.delaxes(axs[i])
        
        fig.suptitle(title, fontsize=16)

        # Extract legend handles from the last active axes to build a global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, 
                   fontsize='large', frameon=True, bbox_to_anchor=(0.5, 0.04))
        
        # Adjust layout to accommodate the global legend
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
        if savefig and (filename is not None):
            # Saving as PDF is generally preferred for high-quality manuscript figures
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            
        plt.show()
