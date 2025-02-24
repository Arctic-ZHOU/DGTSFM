import numpy as np
import torch
from chronos import ChronosPipeline
from properscoring import crps_ensemble

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

pipeline.model.config.prediction_length = 72

def get_forecasts(df, prediction_length=24, num_samples=20):
    forecasts = {}
    evaluation_metrics = {}
    median_forecasts_list = []      
    median_forecasts_last24_list = [] 
    test_series_last24_list = []    
    last_48_train_list = []           
    countries = []   

    for country in df.columns:
        print(f"Processing country: {country}")
        series = df[country].values
        train_series = series[:-prediction_length]
        test_series = series[-prediction_length:]
        context = torch.tensor(train_series)

        all_forecast_values = []
        for run in range(10):
            forecast = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=False, 
            )
            forecast_values = forecast[0].numpy()  
            all_forecast_values.append(forecast_values)

        all_forecast_values = np.concatenate(all_forecast_values, axis=0) 

        median_forecast = np.median(all_forecast_values, axis=0)
        median_forecasts_list.append(median_forecast) 
        median_forecasts_last24_list.append(median_forecast[-24:])
        test_series_last24_list.append(test_series[-24:])
        last_48_train_list.append(train_series[-48:])
        countries.append(country)  

        mae = np.mean(np.abs(median_forecast - test_series))
        rmse = np.sqrt(np.mean((median_forecast - test_series) ** 2))
        epsilon = 1e-10
        mape = np.mean(np.abs((median_forecast - test_series) / (test_series + epsilon))) * 100
        crps = np.mean(crps_ensemble(test_series, all_forecast_values.T))

        evaluation_metrics[country] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'CRPS': crps}

        low, median, high = np.quantile(all_forecast_values, [0.1, 0.5, 0.9], axis=0)
        forecast_index = df.index[-prediction_length:]

        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Median': median_forecast,
            'Low_10%': low,
            'High_90%': high,
            'Actual': test_series
        })

        forecast_df.to_csv(f'{country}_forecast.csv', index=False)

    evaluation_df = pd.DataFrame(evaluation_metrics).T
    return evaluation_df, median_forecasts_last24_list, test_series_last24_list, countries
