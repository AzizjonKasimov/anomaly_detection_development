
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


def calculate_weighted_anomaly_density(only_red, only_yellow, time_index):
    if (only_red == 1).sum() == 0 and (only_yellow == 1).sum() == 0:
        return 0.0

    total_points = len(time_index)
    
    # Count red and yellow anomalies
    red_count = (only_red == 1).sum()
    yellow_count = (only_yellow == 1).sum()
    
    # Calculate weighted sum (red has full weight, yellow has half weight)
    weighted_sum = red_count + 0.5 * yellow_count
    
    # Calculate weighted density
    weighted_density = weighted_sum / total_points
    
    return weighted_density

def process_anomalies(model, predictions, threshold, anomaly_dict, time_index):
    for i, pred in enumerate(predictions):
        if pred == 1:
            timestamp = time_index[i]
            err_scores = model.test_err_scores[:, i]
            unusual_features = [
                model.input_column_names[j] 
                for j, score in enumerate(err_scores) 
                if score > threshold[j]
            ]
            for feature in unusual_features:
                anomaly_dict.setdefault(feature, []).append(timestamp)
    return anomaly_dict

def timestamp_to_string(timestamp):
    return timestamp.strftime("%Y%m%d%H%M%S")

def get_anomalies_by_feature(model, pred_labels_yellow, pred_labels_red, time_index):
    time_index = time_index[model.slide_win:]

    error_thresholds_yellow = model.thresholds_yellow
    error_thresholds_red = model.thresholds_red

    only_red = pred_labels_red
    only_yellow = pred_labels_yellow - pred_labels_red # Since we calculate the red anomalies by increasing the threshold, that means that the red anomalies are a subset of the yellow anomalies

    yellow_anomalies = process_anomalies(model, only_yellow, error_thresholds_yellow, {}, time_index)
    red_anomalies = process_anomalies(model, only_red, error_thresholds_red, {}, time_index)

    anomaly_score = calculate_weighted_anomaly_density(only_red, only_yellow, time_index)

    # Convert timestamps to string format
    for color in [red_anomalies, yellow_anomalies]:
        for feature, timestamps in color.items():
            color[feature] = [timestamp_to_string(ts) for ts in timestamps]

    # Check if there are any anomalies
    has_yellow_anomalies = any(len(timestamps) > 0 for timestamps in yellow_anomalies.values())
    has_red_anomalies = any(len(timestamps) > 0 for timestamps in red_anomalies.values())

    output = {
        "result": has_yellow_anomalies or has_red_anomalies,
        "anomaly_score": anomaly_score,
        "anomalies": {
            "yellow": has_yellow_anomalies,
            "red": has_red_anomalies,
            "yellow_anomalies": yellow_anomalies,
            "red_anomalies": red_anomalies
        }
    }

    return output