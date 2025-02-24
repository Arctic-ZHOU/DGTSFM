import torch
import torch.nn as nn
from GAT2 import GAT, LayerType

class GATModel(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_out_features, num_layers, num_heads=8, dropout=0.6, edge_feature_dim=144):
        super(GATModel, self).__init__()
        self.gat = GAT(
            num_of_layers=num_layers,
            num_heads_per_layer=[num_heads] * num_layers,
            num_features_per_layer=[num_in_features] + [num_hidden_features] * (num_layers - 1) + [num_out_features],
            add_skip_connection=True,
            bias=True,
            dropout=dropout,
            layer_type=LayerType.IMP3,
            log_attention_weights=False,
            edge_feature_dim=edge_feature_dim
        )

    def forward(self, data):
        return self.gat(data)
