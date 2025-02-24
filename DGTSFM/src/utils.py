import pandas as pd
import numpy as np
import torch
import os

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to edge index for graph operations.
    """
    edge_index = np.array(np.nonzero(adj))
    return edge_index

def parse_mtu(mtu_str):
    """
    Parse the MTU string to datetime.
    """
    start_time_str = mtu_str.split(' - ')[0]
    start_time_str = start_time_str.split('(')[0].strip()
    return pd.to_datetime(start_time_str, format='%d.%m.%Y %H:%M')

def preprocess_data(file_url="https://raw.githubusercontent.com/Arctic-ZHOU/Data/refs/heads/main/country_data_new.csv"):
    """
    Preprocess data by reading CSV, converting columns, and filling missing values.
    """
    df = pd.read_csv(file_url)
    df['MTU'] = df['MTU'].apply(parse_mtu)
    df.set_index('MTU', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def get_transmission_data(src, dst, transmission_folder='transmission'):
    """
    Fetch transmission data for a pair of countries (src, dst) from files.
    """
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
    
    return flow_y_to_x, flow_x_to_y
