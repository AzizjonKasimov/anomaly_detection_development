from matplotlib import pyplot as plt
import pywt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import adfuller
from scipy import stats

def plot_feature_with_anomalies(data, feature):
    """
    Plot the values of a given feature from a DataFrame and highlight anomalies.
    
    Parameters:
        data (DataFrame): The DataFrame containing the feature and label columns.
        feature (str): The name of the feature column to plot.
        label_column (str): The name of the column containing anomaly labels for the feature.
    """
    # Extract the feature values and labels
    values = data[feature]
    labels = data['Fault']

    # Create the plot
    plt.figure(figsize=(12, 4))  # Set the figure size as needed

    # Plot the normal data
    plt.plot(data.index, values, label=f'Normal {feature}', color='blue', alpha=0.7)

    # Highlight the anomalies
    # Find indices where labels indicate an anomaly and ensure they are aligned with the DataFrame's index
    anomaly_indices = labels[labels != 0].index
    anomaly_values = values.loc[anomaly_indices]  # Use .loc[] to ensure proper alignment with the DataFrame index

    plt.scatter(anomaly_indices, anomaly_values, color='red', label='Anomalies', s=10)  # 's' adjusts the marker size

    # Add title and labels
    plt.title(f'{feature} with Anomalies Highlighted')
    plt.xlabel('Index')
    plt.ylabel(feature)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def clip_values(data, lower_percentile=0.01, upper_percentile=0.99):
    """ Remove outliers from the data using the specified percentiles"""
    # Calculate the percentile values
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)

    data = data.clip(lower_bound, upper_bound)
    return data

# Resample data to have different time intervals
def resample_data(df, interval='10min'):
    df = df.resample(interval).agg({
    'global_tilted_irradiance': 'mean',
    'diffuse_tilted_irradiance': 'mean',
    'ambient_temperature': 'mean',
    'panel_temperature': 'mean',
    'energy_injected': 'mean',
    'power_injected': 'mean',
    'ambient_current': 'mean',
    'current_injected': 'mean',
    'pv_plant_voltage': 'mean',
    'grid_voltage': 'mean',
    'grid_frequency': 'mean',
    'Fault': 'min'
    })
    return df

def filter_active_hours(df, monthly_thresholds, irradiance_column='global_tilted_irradiance'):
    # Group by date
    grouped = df.groupby(df.index.date)
    
    filtered_data = []
    
    for date, day_data in grouped:
        # Get the month of the current day
        current_month = day_data.index[0].month
        
        # Use the threshold for the current month if available, otherwise use the mean of all thresholds
        if current_month in monthly_thresholds.index:
            irradiance_threshold = monthly_thresholds[current_month]
        else:
            irradiance_threshold = monthly_thresholds.mean()
        
        # Identify active hours based on irradiance threshold
        active_mask = day_data[irradiance_column] >= irradiance_threshold
        
        # Find the first and last active hour
        if active_mask.any():
            start_time = day_data[active_mask].index.min()
            end_time = day_data[active_mask].index.max()
            
            # Filter data for the active period
            active_data = day_data.loc[start_time:end_time]
            
            filtered_data.append(active_data)
    
    # Concatenate all filtered daily data
    return pd.concat(filtered_data)

def find_irradiance_threshold(df, irradiance_column='global_tilted_irradiance', power_column='ambient_current', percentile=1):
    # Sort data by irradiance
    sorted_data = df.sort_values(by=irradiance_column)
    
    # Find the irradiance value at which power output starts to increase significantly
    cumulative_power = sorted_data[power_column].cumsum()
    total_power = cumulative_power.iloc[-1]
    
    threshold_index = (cumulative_power >= total_power * (percentile/100)).idxmax()
    threshold_irradiance = sorted_data.loc[threshold_index, irradiance_column]
    
    return threshold_irradiance

def identify_static_features(df, significance_level=0.05):
    static_features = []
    
    for column in df.columns:
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(df[column])
        p_value = result[1]
        
        # Check for trend using linear regression
        x = np.arange(len(df))
        slope, _, _, _, _ = stats.linregress(x, df[column])
        
        # If the feature is stationary (no unit root) and has no significant trend, consider it static
        if p_value < significance_level and abs(slope) < 1e-5:
            static_features.append(column)
    
    return static_features


def wavelet_denoise(data, wavelet='sym8', level=3):
    data = data.values

    # Decompose the signal
    coeff = pywt.wavedec(data, wavelet, mode="per")
    
    # Calculate the threshold
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply thresholding
    coeff[1:] = [pywt.threshold(i, value=threshold, mode="soft") for i in coeff[1:]]
    
    # Reconstruct the signal
    return pywt.waverec(coeff, wavelet, mode="per")


def savitzky_golay(data, window=5, polyorder=2):
    return pd.Series(savgol_filter(data, window, polyorder), index=data.index)


def rolling_median_filter(data, window=5):
    return data.rolling(window=window, center=True, min_periods=1).median()