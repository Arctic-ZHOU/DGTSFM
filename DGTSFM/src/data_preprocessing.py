import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def parse_mtu(mtu_str):
    start_time_str = mtu_str.split(' - ')[0]
    start_time_str = start_time_str.split('(')[0].strip()
    return pd.to_datetime(start_time_str, format='%d.%m.%Y %H:%M')

def preprocess_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Arctic-ZHOU/Data/refs/heads/main/country_data_new.csv")
    df['MTU'] = df['MTU'].apply(parse_mtu)
    df.set_index('MTU', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
