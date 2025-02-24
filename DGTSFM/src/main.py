from data_preprocessing import preprocess_data
from forecasting import get_forecasts
from gnn import get_node_features, get_edge_features, adj_to_edge_index
from optuna_optimization import optimize

if __name__ == '__main__':
    df = preprocess_data()
    evaluation_df, median_forecasts_last24_list, test_series_last24_list, countries = get_forecasts(df)

    edge_index = adj_to_edge_index(df)
    node_features = get_node_features(median_forecasts_last24_list, df)
    edge_features = get_edge_features(edge_index, countries)
    
    best_params = optimize(node_features, edge_features)
    print(f"Best Hyperparameters: {best_params}")
