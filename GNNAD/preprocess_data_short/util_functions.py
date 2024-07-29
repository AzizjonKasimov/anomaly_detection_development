from matplotlib import pyplot as plt

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

def remove_outliers_iqr(df):  
    """ Remove outliers from the data using the IQR method""" 

    fault_col = df['Fault'] # Extract the 'Fault' column to a separate variable

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)


    df['Fault'] = fault_col # Assign the 'Fault' column back to the capped data

    return df