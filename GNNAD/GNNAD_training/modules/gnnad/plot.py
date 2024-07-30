# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_error_scores(error_scores, ground_truth_anomalies, predicted_anomalies, time_index=None):
    # Create time index if not provided
    if time_index is None:
        time_index = range(len(error_scores))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    
    # Plot error scores
    ax1.plot(time_index, error_scores, label='Error Score', color='blue')

    # Highlight predicted anomalies
    anomaly_indices = np.where(predicted_anomalies)[0]
    ax1.scatter(np.array(time_index)[anomaly_indices], 
                error_scores[anomaly_indices], 
                color='red', label='Predicted Anomaly', s=50)

    ax1.set_title('Error Scores Over Time')
    ax1.set_ylabel('Error Score')
    ax1.legend()

    # Plot ground truth vs predictions
    ax2.plot(time_index, ground_truth_anomalies, label='Ground Truth', color='green', linestyle='--')
    ax2.plot(time_index, predicted_anomalies, label='Predictions', color='red', linestyle='-')
    
    ax2.set_title('Ground Truth vs Predicted Anomalies')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Anomaly (1) / Normal (0)')
    ax2.legend()

    # Set y-axis limits for the second plot
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Anomaly'])

    plt.tight_layout()
    plt.show()

    # Print average error score
    mean_error = np.mean(error_scores)
    print(f"Average Error Score: {mean_error:.4f}")


def plot_time_series_with_anomalies(dataset, ground_labels, predictions):
    # Count the number of anomalies
    anom_number = predictions.value_counts().get(1, 0)

    # Create subplots: one for each column in the dataset and one for the ground truth predictions
    num_columns = len(dataset.columns)
    fig, axs = plt.subplots(num_columns + 1, 1, figsize=(18, 1.5 * (num_columns + 1)), dpi=300, gridspec_kw={'height_ratios': [3] * num_columns + [1]})

    # Loop through each column in the DataFrame and create a plot
    for i, column in enumerate(dataset.columns):
        axs[i].plot(dataset.index, dataset[column], linewidth=0.5)
        if anom_number > 0:
            # Identify the indices of the anomalies
            anomal_idx = predictions[predictions == 1].index
            # Highlight the anomalies on the plot
            axs[i].scatter(anomal_idx, dataset[column].loc[anomal_idx], color='red', label='Anomaly', marker='o', s=5)
        # Set plot title and predictions
        axs[i].set_title(f'Time Series for {column}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')

    # Plot the ground truth predictions in the last subplot
    axs[-1].plot(ground_labels.index, ground_labels, color='blue', linewidth=0.5)
    axs[-1].set_title('Ground Truth Labels')
    axs[-1].set_xlabel('Time')
    axs[-1].set_ylabel('Label')
    axs[-1].set_yticks([0, 1])
    axs[-1].set_yticklabels(['Normal', 'Anomaly'])

    plt.tight_layout()
    plt.show()
