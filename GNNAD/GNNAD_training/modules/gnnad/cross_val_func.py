import numpy as np

def drop_anomalous_points(df, pred_anom_list, test_set, window_size=None):
    # Get the indices of the anomalies in the test set
    test_set_anomalies_idx = [i for i, x in enumerate(pred_anom_list) if x == 1] 

    # Get the indices of the anomalies in the original dataframe. The test_set index is shifted by window_size, because the model predicts anomalies for the window_size first samples
    df_anomalies_to_remove = test_set[window_size:].iloc[test_set_anomalies_idx] 

    # Get the indices of the anomalies in the original dataframe
    df_anom_idx_list = df_anomalies_to_remove.index.to_list() 

    # Remove the anomalies from the dataframe to avoid training on them
    df_filtered = df.drop(df_anom_idx_list)

    return df_filtered

def test_set_windows(test_set, window_size):
    # Create a list to store the test sets
    windows_list = []

    # Iterate over the range of dates
    for i in range(len(test_set) - window_size):
        # Get the data for the window
        window = test_set.iloc[i:i+window_size+1]

        windows_list.append(window)

    return windows_list

def process_anomalies(model, predictions, threshold, anomaly_dict, time_index, exclude_set=None):
    anomaly_timestamps = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            timestamp = time_index[i]
            if exclude_set and timestamp in exclude_set:
                continue
            err_scores = model.test_err_scores[:, i]
            unusual_features = [
                model.input_column_names[j] 
                for j, score in enumerate(err_scores) 
                if score > threshold[j]
            ]
            for feature in unusual_features:
                anomaly_dict.setdefault(feature, []).append(timestamp)
            anomaly_timestamps.append(i)
    return anomaly_dict, anomaly_timestamps

def timestamp_to_string(timestamp):
    return timestamp.strftime("%Y%m%d%H%M%S")

def get_anomalies_by_feature(model, pred_labels_yellow, pred_labels_red, time_index):
    time_index = time_index[model.slide_win:]
    error_thresholds_yellow = np.percentile(model.validate_err_scores, 90, axis=1)
    error_thresholds_red = np.max(model.validate_err_scores, axis=1) * model.threshold_multiplier

    red_anomalies, red_indices = process_anomalies(model, pred_labels_red, error_thresholds_red, {}, time_index)
    red_timestamps = set(timestamp for timestamps in red_anomalies.values() for timestamp in timestamps)
    yellow_anomalies, yellow_indices = process_anomalies(model, pred_labels_yellow, error_thresholds_yellow, {}, time_index, red_timestamps)

    all_anomaly_indices = red_indices + yellow_indices

    if all_anomaly_indices:
        anomaly_scores = model.test_err_scores[:, all_anomaly_indices]
        anomaly_score = float(np.mean(anomaly_scores))
    else:
        anomaly_score = 0.0

    # Convert timestamps to string format
    for color in [red_anomalies, yellow_anomalies]:
        for feature, timestamps in color.items():
            color[feature] = [timestamp_to_string(ts) for ts in timestamps]

    output = {
        "result": bool(yellow_anomalies or red_anomalies),
        "anomaly_score": anomaly_score,
        "anomalies": {
            "yellow": bool(yellow_anomalies),
            "red": bool(red_anomalies),
            "yellow_anomalies": yellow_anomalies,
            "red_anomalies": red_anomalies
        }
    }

    return output