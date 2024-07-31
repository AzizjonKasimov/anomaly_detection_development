from matplotlib import pyplot as plt
import pywt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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


def remove_outliers(data, lower_percentile=0.02, upper_percentile=0.98):
    """ Remove outliers from the data using the specified percentiles"""

    fault_col = data['Fault'] # Extract the 'Fault' column to a separate variable

    # Calculate the percentile values
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)

    data = data.clip(lower_bound, upper_bound, axis=1)
    data['Fault'] = fault_col # Assign the 'Fault' column back to the capped data
    return data


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


def savitzky_golay(data, window_length=4, polyorder=1):
    return pd.Series(savgol_filter(data, window_length, polyorder), index=data.index)


def rolling_median_filter(data, window=5):
    return data.rolling(window=window, center=True, min_periods=1).median()