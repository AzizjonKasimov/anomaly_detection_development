# -*- coding: utf-8 -*-
"""
Graph Neural Network-Based Anomaly Detection.

References
----------
[1] Deng, Ailin, and Bryan Hooi. "Graph neural network-based anomaly detection in multivariate time series."
Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.
[2] Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems"
arXiv preprint arXiv:2304.09367 (2023).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


__author__ = ["KatieBuc"]


class TimeDataset(Dataset):
    """
    A PyTorch dataset class for time series data, to provide additional functionality for
    processing time series data.

    Attributes
    ----------
    raw_data : list
        A list of raw data
    config : dict
        A dictionary containing the configuration of dataset
    edge_index : np.ndarray
        Edge index of the dataset
    mode : str
        The mode of dataset, either 'train' or 'test'
    x : torch.Tensor
        Feature data
    y : torch.Tensor
        Target data
    labels : torch.Tensor
        Anomaly labels of the data
    """

    def __init__(self, raw_data, mode="train", config=None):
        self.raw_data = raw_data
        self.config = config
        self.mode = mode

        # to tensor
        data = torch.tensor(raw_data[:-1]).double()
        labels = torch.tensor(raw_data[-1]).double()
        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr, labels_arr = [], [], []
        slide_win, slide_stride = self.config["slide_win"], self.config["slide_stride"]
        is_train = self.mode == "train"
        total_time_len = data.shape[1]

        for i in (
            range(slide_win, total_time_len, slide_stride)
            if is_train
            else range(slide_win, total_time_len)
        ):
            ft = data[:, i - slide_win : i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        label = self.labels[idx].double()

        return feature, y, label


def loss_func(y_pred, y_true, loss="mse"):
    if loss == "mse":
        loss_func = F.mse_loss(y_pred, y_true, reduction="mean")
    elif loss == "mae":
        loss_func = F.l1_loss(y_pred, y_true, reduction="mean")
    return loss_func


def parse_data(data, feature_list, labels=None):
    """
    In the case of training data, fill the last column with zeros. This is an
    implicit assumption in the uhnsupervised training case - that the data is
    non-anomalous. For the test data, keep the labels.
    """
    labels = [0] * data.shape[0] if labels is None else labels
    res = data[feature_list].T.values.tolist()
    res.append(labels)
    return res


def get_full_err_scores(test_result, smoothen_error=True):
    """Get array of error scores for each feature by applying the
    `get_err_scores` function on every slice of the `test_result` tensor.
    """
    all_scores = [
        get_err_scores(test_result[:2, :, i], smoothen_error)
        for i in range(test_result.shape[-1])
    ]
    return np.vstack(all_scores)


def get_err_scores(test_result_list, smoothen_error):
    """
    Calculate the error scores

    Parameters
    ----------
    test_result_list (list):
        List containing two lists of predicted and ground truth values
    smoothen_error (bool):
        A boolean value indicating whether error smoothing should be applied or not

    Returns
    -------
    err_scores (np.ndarray):
        An array of error scores
    """
    test_predict, test_ground = test_result_list

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_ground).astype(np.float64),
        )
    )

    if smoothen_error:
        smoothed_err_scores = np.zeros(test_delta.shape)
        before_num = 3
        for i in range(before_num, len(test_delta)):
            smoothed_err_scores[i] = np.mean(test_delta[i - before_num : i + 1])

        return smoothed_err_scores
    return test_delta


def aggregate_error_scores(err_scores, topk=1):

    # finds topk features idxs of max scores for each time point
    topk_indices = np.argpartition(err_scores, -topk, axis=0)[-topk:]

    # for each time, sum the topk error scores
    topk_err_scores = np.sum(
        np.take_along_axis(err_scores, topk_indices, axis=0), axis=0
    )

    return topk_indices, topk_err_scores


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------------------------------------------------------------------------
def drop_anomalous_points(df, pred_anom_list, test_set, window_size=None):
    test_set_anomalies_idx = [i for i, x in enumerate(pred_anom_list) if x == 1] # Get the indices of the anomalies in the test set
    df_anomalies_to_remove = test_set[window_size:].iloc[test_set_anomalies_idx] # Get the indices of the anomalies in the original dataframe. The test_set index is shifted by window_size, because the model predicts anomalies for the window_size first samples
    df_anom_idx_list = df_anomalies_to_remove.index.to_list() # Get the indices of the anomalies in the original dataframe
    df_filtered = df.drop(df_anom_idx_list) # Remove the anomalies from the dataframe to avoid training on them
    return df_filtered