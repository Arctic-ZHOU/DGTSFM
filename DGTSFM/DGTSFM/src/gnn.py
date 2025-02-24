import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import networkx as nx

from GAT2 import GAT, LayerType

def adj_to_edge_index(adj):
    edge_index = np.array(np.nonzero(adj))
    return edge_index

def get_node_features(median_forecasts_list, last_48_train_list):
    median_forecasts_array = np.array(median_forecasts_list)  
    last_48_train_array = np.array(last_48_train_list)  
    median_forecasts_first24_array = median_forecasts_array[:, :24] 
    node_features_array = np.concatenate([last_48_train_array, median_forecasts_first24_array], axis=1) 

    scaler = StandardScaler()
    node_features_array_normalized = scaler.fit_transform(node_features_array)
    node_features = torch.tensor(node_features_array_normalized, dtype=torch.float32)
    return node_features

def get_edge_features(edge_index, countries, transmission_folder='transmission'):
    edge_features_dict = {}

    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        src = countries[src_idx]
        dst = countries[dst_idx]
        
        file_name = f"{src}-{dst}.csv"
        alt_file_name = f"{dst}-{src}.csv"
        
        file_path = os.path.join(transmission_folder, file_name)
        alt_file_path = os.path.join(transmission_folder, alt_file_name)
        
        if os.path.exists(file_path):
            edge_df = pd.read_csv(file_path)
            flow_y_to_x = edge_df[f'{dst}>{src}'].values  
            flow_x_to_y = edge_df[f'{src}>{dst}'].values  
        elif os.path.exists(alt_file_path):
            edge_df = pd.read_csv(alt_file_path)
            flow_y_to_x = edge_df[f'{src}>{dst}'].values  
            flow_x_to_y = edge_df[f'{dst}>{src}'].values  
        else:
            raise FileNotFoundError(f"Edge data file not found for edge {src}-{dst}")

        flow_y_to_x = flow_y_to_x[-(48+24):]
        flow_x_to_y = flow_x_to_y[-(48+24):]
        
        edge_feature = np.concatenate([flow_y_to_x, flow_x_to_y])
        edge_features_dict[(src_idx, dst_idx)] = edge_feature

    edge_features_list = []
    for i in range(edge_index.shape[1]):
        key = (edge_index[0, i].item(), edge_index[1, i].item())
        edge_feature = edge_features_dict[key]
        edge_features_list.append(edge_feature)

    edge_features_array = np.array(edge_features_list)
    edge_features = torch.tensor(edge_features_array, dtype=torch.float32)

    edge_scaler = StandardScaler()
    edge_features_array_normalized = edge_scaler.fit_transform(edge_features.numpy())
    edge_features = torch.tensor(edge_features_array_normalized, dtype=torch.float32)

    return edge_features
