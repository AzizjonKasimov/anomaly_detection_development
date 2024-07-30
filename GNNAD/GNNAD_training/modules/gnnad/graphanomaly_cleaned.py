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

import os
import random
from pathlib import Path

from modules.gnnad.GDN import GDN
from modules.gnnad.utils import TimeDataset, parse_data, aggregate_error_scores, get_full_err_scores, loss_func, seed_worker
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

__author__ = ["KatieBuc"]


class GNNAD(GDN):
    """
    Graph Neural Network-based Anomaly Detection for Multivariate Timeseries.
    """

    def __init__(
        self,
        batch: int = 128,
        epoch: int = 100,
        slide_win: int = 15,
        embed_dim: int = 64,
        slide_stride: int = 5,
        random_seed: int = 0,
        out_layer_num: int = 1,
        out_layer_inter_dim: int = 256,
        decay: float = 0,
        topk: int = 11,
        device: str = "cpu",
        save_model_name: str = "",
        early_stop_win: int = 15,
        lr: float = 0.001,
        shuffle_train: bool = True,
        threshold_type: str = None,
        smoothen_error: bool = True,
        use_deterministic: bool = False,
        percentile: int = 99,
        errors_topk: int = 1,
        loss_func: str = "mse",
        validation_size: int = None,
        input_column_names: list = None,
    ):
        self.input_column_names = input_column_names
        fc_edge_idx = self.make_graph(self.input_column_names)
        
        
        super().__init__(
            fc_edge_idx.to(device),
            n_nodes=len(self.input_column_names),
            embed_dim=embed_dim,
            out_layer_inter_dim=out_layer_inter_dim,
            slide_win=slide_win,
            out_layer_num=out_layer_num,
            topk=topk,
        )

        """
        Parameters
        ----------
        batch : int, optional (default=128)
            Batch size for training the model
        epoch : int, optional (default=100)
            Number of epochs to train the model
        slide_win : int, optional (default=15)
            Size of sliding window used as feature input
        embed_dim : int, optional (default=64)
            Dimension of the node embeddings in the GDN model
        slide_stride : int, optional (default=5)
            Stride of the sliding window
        random_seed : int, optional (default=0)
            Seed for random number generation for reproducibility
        out_layer_num : int, optional (default=1)
            Number of layers in the output network
        out_layer_inter_dim : int, optional (default=256)
            Internal dimensions of layers in the output network
        decay : float, optional (default=0)
            Weight decay factor for regularization during training
        topk : int, optional (default=20)
            Number of permissable neighbours in the learned graph
        device : str, optional (default="cpu")
            Device to use for training the model ('cpu' or 'cuda')
        save_model_name : str, optional (default="")
            Name to use for saving the trained model
        early_stop_win : int, optional (default=15)
            Number of consecutive epochs without improvement in validation loss to
            trigger early stopping
        lr : float, optional (default=0.001)
            Learning rate for training the model
        shuffle_train : bool, optional (default=True)
            Whether to shuffle the training data during training
        threshold_type : str, optional (default=None)
            Type of threshold to use for anomaly detection ("max_validation")
        suppress_print : bool, optional (default=False)
            Whether to suppress print statements during training
        smoothen_error : bool, optional (default=True)
            Whether to smoothen the anomaly scores before thresholding
        use_deterministic : bool, optional (default=False)
            Whether to use deterministic algorithms for reproducibility and unit testing
        percentile : int, optional (default=99)
            Percentile value to use for thresholding anomaly scores
        errors_topk : int, optional (default=1)
            Number of top errors to consider for thresholding
        loss_func : str, optional (default="mse")
            Loss function to use for training the model ("mse" or "mae")
        validation_size : int, optional (default=None)
            Size of the validation set to use for early stopping
        input_column_names : list (default=None)
            List of column names for the input data
        fc_edge_idx : torch.LongTensor
            Edge indices of fully connected graph for the input time series
        n_nodes : int
            Number of nodes in the graph
        """

        self.batch = batch
        self.epoch = epoch
        self.slide_win = slide_win
        self.slide_stride = slide_stride
        self.random_seed = random_seed
        self.decay = decay
        self.device = device
        self.save_model_name = save_model_name
        self.early_stop_win = early_stop_win
        self.lr = lr
        self.shuffle_train = shuffle_train
        self.threshold_type = threshold_type
        self.smoothen_error = smoothen_error
        self.use_deterministic = use_deterministic
        self.percentile = percentile
        self.errors_topk = errors_topk
        self.loss_func = loss_func
        self.validation_size = validation_size
        self.cfg = {
            "slide_win": self.slide_win,
            "slide_stride": self.slide_stride,
        }

        # set the seeds and load GDN model
        self._set_seeds()
        self._initialise_layers()
        # Move the model to the specified device
        self.to(self.device)
        self._get_model_path()


    ## set the random seeds and determinism for reproducibility
    def _set_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        if self.use_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    

    ## get the loader generator for reproducibility
    def _get_loader_generator(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        return g

    ## load the training, validation and test data
    def make_graph(self, feature_list):
        fc_struc = {
            ft: [x for x in feature_list if x != ft] for ft in feature_list
        }  # fully connected structure

        edge_idx_tuples = [
            (feature_list.index(child), feature_list.index(node_name))
            for node_name, node_list in fc_struc.items()
            for child in node_list
        ] # create tuples where each tuple represents an edge between two nodes.

        fc_edge_idx = [
            [x[0] for x in edge_idx_tuples],
            [x[1] for x in edge_idx_tuples],
        ] # separate the tuples into two lists, one for the source node and the other for the destination node. It is done to in

        fc_edge_idx = torch.tensor(fc_edge_idx, dtype=torch.long)
        return fc_edge_idx
    

    ## Load only test data for _predict method
    def _load_test_data(self, X_test, y_test=None):
        test_input = parse_data(X_test, self.input_column_names, labels=y_test)
        test_dataset = TimeDataset(test_input, mode="test", config=self.cfg)

        # get data loaders
        g = self._get_loader_generator() if self.use_deterministic else None
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch,
            shuffle=False,
            generator=g,
        )

        # save to self
        self.test_dataloader = test_dataloader


    def load_train_data(self, X_train):
        train_subset =  X_train.iloc[: -self.validation_size]
        validate_subset = X_train.iloc[-self.validation_size :]
        
        train_input = parse_data(train_subset, self.input_column_names) # output data will have number_features+1 as rows, where the last row will be the added labels 
        val_input = parse_data(validate_subset, self.input_column_names)

        train_dataset = TimeDataset(train_input, mode="train", config=self.cfg)
        val_dataset = TimeDataset(val_input, mode="val", config=self.cfg)

        g = self._get_loader_generator() if self.use_deterministic else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch,
            shuffle=self.shuffle_train,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        validate_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch,
            shuffle=False,
            generator=g,
        )

        # save to self
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
    

    def load_saved_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.to(self.device)


    ## Make a directory to save the model
    def _get_model_path(self):
        model_path = f"./weights/last_trained_model.pt"
        dirname = os.path.dirname(model_path)

        Path(dirname).mkdir(parents=True, exist_ok=True)

        self.model_path = model_path


    ## testing function to evaluate the model
    def _test(self, dataloader):
        test_loss_list = []
        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []

        self.eval()
        for x, y, labels in dataloader:
            x, y, labels = [
                item.to(self.device).float() for item in [x, y, labels]
            ] # move the variables to device

            with torch.no_grad():
                predicted = self(x).float()

                loss = loss_func(predicted, y, self.loss_func)

                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                # store results
                t_test_predicted_list.append(predicted)
                t_test_ground_list.append(y)
                t_test_labels_list.append(labels)

            test_loss_list.append(loss.item())

        # Concatenate all tensors
        t_test_predicted_list = torch.cat(t_test_predicted_list, dim=0)
        t_test_ground_list = torch.cat(t_test_ground_list, dim=0)
        t_test_labels_list = torch.cat(t_test_labels_list, dim=0)

        # Convert to lists
        test_predicted_list = t_test_predicted_list.cpu().tolist()
        test_ground_list = t_test_ground_list.cpu().tolist()
        test_labels_list = t_test_labels_list.cpu().tolist()

        avg_loss = sum(test_loss_list) / len(test_loss_list)

        return avg_loss, np.array(
            [test_predicted_list, test_ground_list, test_labels_list]
        )


    # training method to update the model weights with early stopping using validation data
    def _train(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.decay
        )

        train_log = []
        max_loss = 1e6
        stop_improve_count = 0

        for i_epoch in range(self.epoch):
            self.train()

            for x, y, _ in self.train_dataloader:
                x, y = [
                    item.float().to(self.device) for item in [x, y]
                ]
                optimizer.zero_grad()
                out = self(x).float()

                loss = loss_func(out, y, self.loss_func)
                loss.backward()
                optimizer.step()

                train_log.append(loss.item())

            # use val dataset to judge
            if self.validate_dataloader is not None:
                val_loss, _ = self._test(self.validate_dataloader)

                if val_loss < max_loss:
                    torch.save(self.state_dict(), self.model_path)
                    max_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1

                if stop_improve_count >= self.early_stop_win:
                    break

        self.train_log = train_log
        
        # Load the best model after training
        self.load_state_dict(torch.load(self.model_path))
    

    def fit(self, X_train):
        self.load_train_data(X_train)
        self._train()
        self.get_threshold()


    def get_threshold(self):
        _, self.validate_result = self._test(self.validate_dataloader)

        # get stacked array of error scores
        validate_err_scores = get_full_err_scores(self.validate_result, self.smoothen_error)

        # get threshold value
        if self.threshold_type == "percentile":
            _, topk_val_err_scores = aggregate_error_scores(validate_err_scores, topk=self.errors_topk)
            threshold = np.percentile(topk_val_err_scores, self.percentile)
        elif self.threshold_type == "max_validation":
            _, topk_val_err_scores = aggregate_error_scores(validate_err_scores, topk=self.errors_topk)
            threshold = np.max(topk_val_err_scores)

        self.validate_err_scores = validate_err_scores
        self.threshold = threshold
        self.threshold_i = threshold # for plots


    ## predict method that will use only test data to get the error scores
    def predict(self, X_test, y_test=None):
        # read in best model
        self._load_test_data(X_test, y_test) # load only test data which will save to self

        # store results to self
        test_avg_loss, self.test_result = self._test(self.test_dataloader)
        test_labels = self.test_result[2, :, 0]
        
        test_err_scores = get_full_err_scores(self.test_result, self.smoothen_error)
        topk_err_indices, topk_err_scores = aggregate_error_scores(test_err_scores, topk=self.errors_topk)

        # get prediction labels for decided threshold
        pred_labels = np.zeros(len(topk_err_scores))
        pred_labels[topk_err_scores > self.threshold] = 1

        pred_labels = pred_labels.astype(int)
        test_labels = test_labels.astype(int)
        
        # if test_labels are not all zeros, calculate metrics
        if not all(element == 0 for element in test_labels):
            # calculate metrics
            precision = precision_score(test_labels, pred_labels)
            recall = recall_score(test_labels, pred_labels)
            f1 = None
            f1 = f1_score(test_labels, pred_labels) if f1 is None else f1

            self.precision = precision
            self.recall = recall
            self.f1 = f1

        else: # if all test_labels are zeros
            self.false_positives = np.sum(pred_labels == 1)
            self.fp_rate = self.false_positives / len(test_labels)

        # save to self
        self.test_err_scores = test_err_scores
        self.topk_err_indices = topk_err_indices
        self.topk_err_scores = topk_err_scores
        self.pred_labels = pred_labels
        self.test_labels = test_labels
        self.test_avg_loss = test_avg_loss

        return pred_labels


    def summary(self):
        return summary(self.model, (self.n_nodes, self.slide_win))

