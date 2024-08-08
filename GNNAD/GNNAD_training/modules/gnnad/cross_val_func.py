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

def get_anomalies_by_feature(model, pred_labels_yellow, pred_labels_red, time_index):
    time_index = time_index[model.slide_win:]
    yellow_anomalies = {}
    red_anomalies = {}

    # Calculate percentiles of error scores for each feature from validation set
    error_thresholds_yellow = np.percentile(model.validate_err_scores, 90, axis=1)
    error_thresholds_red = np.max(model.validate_err_scores, axis=1) * model.threshold_multiplier

    # Set to keep track of timestamps already classified as red anomalies
    red_timestamps = set()

    # First, process red anomalies
    for i, red in enumerate(pred_labels_red):
        if red == 1:
            timestamp = time_index[i]
            red_timestamps.add(timestamp)
            err_scores = model.test_err_scores[:, i]
            
            # Identify features with unusually high error scores
            unusual_features = [
                model.input_column_names[j] 
                for j, score in enumerate(err_scores) 
                if score > error_thresholds_red[j]
            ]

            for feature in unusual_features:
                if feature not in red_anomalies:
                    red_anomalies[feature] = []
                red_anomalies[feature].append(timestamp)

    # Then, process yellow anomalies, excluding those already classified as red
    for i, yellow in enumerate(pred_labels_yellow):
        if yellow == 1:
            timestamp = time_index[i]
            # Skip if this timestamp is already classified as a red anomaly
            if timestamp in red_timestamps:
                continue

            err_scores = model.test_err_scores[:, i]
            
            # Identify features with unusually high error scores
            unusual_features = [
                model.input_column_names[j] 
                for j, score in enumerate(err_scores) 
                if score > error_thresholds_yellow[j]
            ]

            for feature in unusual_features:
                if feature not in yellow_anomalies:
                    yellow_anomalies[feature] = []
                yellow_anomalies[feature].append(timestamp)

    # return the output in the right format
    if yellow_anomalies == {} and red_anomalies == {}:
        output = {"result": False,
                  "anomalies": {"yellow": bool(yellow_anomalies), 
                                "red": bool(red_anomalies),
                                "yellow_anomalies": yellow_anomalies,
                                "red_anomalies": red_anomalies}}
    else:
        output = {"result": True,
                  "anomalies": {"yellow": bool(yellow_anomalies), 
                                "red": bool(red_anomalies),
                                "yellow_anomalies": yellow_anomalies,
                                "red_anomalies": red_anomalies
                                    }
                    }

    return output