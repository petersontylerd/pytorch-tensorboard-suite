from datetime import datetime
import os
import struct
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

import torchvision
from torchvision import datasets, models, transforms

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, explained_variance_score, mean_squared_log_error, mean_absolute_error, median_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve, accuracy_score, roc_auc_score, homogeneity_score, completeness_score, classification_report, silhouette_samples


from load_mnist import load_mnist
from mnist_pytorch_dataset import MNISTDataset

class TensorboardExecutor():

    def __init__(self, tensorboard_logs_dir, comment):
        self.tensorboard_logs_dir = tensorboard_logs_dir
        self.comment = comment
        self.writer = SummaryWriter(log_dir=self.tensorboard_logs_dir)

    def add_scalar_(self, tag, value, n_iter):
        self.writer.add_scalar(tag, value, n_iter)

class RunExecuter():

    def __init__(self, config):

        self.config = config

        # random seed settings
        torch.manual_seed(self.config.seed)

        ## network object creation and device assignment
        # if passing in the name of network Class object
        if isinstance(self.config.network, type):
            self.network = self.config.network().to(self.config.device)
        # if network is already instantiated, or if transfer learning network is used
        else:
            self.network = self.config.network.to(self.config.device)

        # name to use when saving network state
        if self.config.network_name is not None:
            self.network_name = self.config.network_name
        else:
            self.network_name = "untitled"

        # network training settings
        self.n_epochs_stop = 5
        self.min_val_loss = np.inf
        self.epochs_no_improve = 0

        # multiclass classification indicator
        if len(np.unique(self.config.train_data.targets)) == 2:
            self.is_multiclass = False
        else:
            self.is_multiclass = True

        # create directory tree for tracking experiment results
        self.create_dirs()

    def update_scalars(self):
        for tag, values in self.metrics.items():
            if "train_" in tag:
                self.tensorboard_logger.add_scalar_("train/" + tag, values[-1], self.epoch_count)
            # else:
            #     self.tensorboard_logger.add_scalar_("validation/" + tag, values[-1], self.epoch_count)

    def create_dirs(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the beginning of a new run.

        """
        ## create directory
        main_path = os.path.join(os.environ["HOME"], "pytorch_experiments")
        if not os.path.exists(main_path):
            os.makedirs(main_path, exist_ok=True)

        # main directory for current experiment
        time_stamp = datetime.today().strftime('%Y%m%d_%H%M') + "_" + self.network_name
        self.experiment_dir = os.path.join(main_path, time_stamp)

        # directory for trained model object
        self.models_dir = os.path.join(self.experiment_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # directory for tensorboard logs
        self.tensorboard_logs_dir = os.path.join(self.experiment_dir, "tensborboard_logs")
        os.makedirs(self.tensorboard_logs_dir, exist_ok=True)

        # directory for images
        self.images_dir = os.path.join(self.experiment_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # directory for raw experiment logs
        self.raw_logs_dir = os.path.join(self.experiment_dir, "raw_logs")
        os.makedirs(self.raw_logs_dir, exist_ok=True)

    def begin_run(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the beginning of a new run.

        """
        # capture time stamp of run start time
        self.run_start_time = time.time()

        # increment number of runs initiated
        self.run_count += 1

        # reset number of epochs initiated
        self.epoch_count = 0

        #
        if self.config.use_tensorboard:
            self.tensorboard_logger = TensorboardExecutor(
                tensorboard_logs_dir=self.tensorboard_logs_dir,
                comment=self.run_count,
            )

        # create a metadata dictionary
        self.meta = OrderedDict([
            ("run", np.empty([0,1])),
            ("run_duration", np.empty([0,1])),
            ("epoch", np.empty([0,1])),
            ("epoch_start_time", np.empty([0,1])),
            ("epoch_end_time", np.empty([0,1])),
            ("epoch_duration", np.empty([0,1])),
        ])

        # add key/value for each hyperparameter to metadata dictionary
        for param in self.run_params.keys():
            self.meta[param] = []

        # create metrics dictionary
        if self.is_multiclass:
            self.metrics = OrderedDict([
                ("train_loss", np.empty([0,1])),
                ("train_accuracy", np.empty([0,1])),
                ("train_precision_macro", np.empty([0,1])),
                ("train_precision_micro", np.empty([0,1])),
                ("train_recall_macro", np.empty([0,1])),
                ("train_recall_micro", np.empty([0,1])),
                ("train_f1_macro", np.empty([0,1])),
                ("train_f1_micro", np.empty([0,1])),
                ("train_number_correct", np.empty([0,1])),
                ("validation_accuracy", np.empty([0,1])),
                ("validation_precision_macro", np.empty([0,1])),
                ("validation_precision_micro", np.empty([0,1])),
                ("validation_recall_macro", np.empty([0,1])),
                ("validation_recall_micro", np.empty([0,1])),
                ("validation_f1_macro", np.empty([0,1])),
                ("validation_f1_micro", np.empty([0,1])),
                ("validation_number_correct", np.empty([0,1])),
            ])

        else:
            self.metrics = OrderedDict([
                ("train_loss", np.empty([0,1])),
                ("train_accuracy", np.empty([0,1])),
                ("train_precision", np.empty([0,1])),
                ("train_recall", np.empty([0,1])),
                ("train_f1", np.empty([0,1])),
                ("train_number_correct", np.empty([0,1])),
                ("validation_accuracy", np.empty([0,1])),
                ("validation_precision", np.empty([0,1])),
                ("validation_recall", np.empty([0,1])),
                ("validation_f1", np.empty([0,1])),
                ("validation_number_correct", np.empty([0,1])),
            ])

    def end_run(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the end of a run.

        """
        # subtract run start time from current time to capture total run time
        self.run_end_time = time.time()
        self.run_duration = self.run_end_time - self.run_start_time

        # append run metrics to csv
        self.dump_metrics_to_csv()

        # reset network parameters
        self.reset_network()

        if self.config.use_tensorboard:
            self.tensorboard_logger.writer.flush()
            self.tensorboard_logger.writer.close()

    def begin_epoch(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the beginning of a new epoch.

        """
        # capture time stamp of epoch start time
        self.epoch_start_time = time.time()

        # increment number of runs initiated
        self.epoch_count += 1

        # reset epoch loss
        self.epoch_loss = 0

        # reset number of batches initiated
        self.batch_count = 0

        # create dictionary for collecting training and validationdata targets
        # and predictions across all batches
        self.epoch_targets_and_predictions = OrderedDict([
            ("train_targets",np.array([0,1])),
            ("train_predictions",np.array([0,1])),
            ("validation_targets",np.array([0,1])),
            ("validation_predictions",np.array([0,1])),
        ])

    def end_epoch(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the end of an epoch.

        """
        # subtract run start time from current time to capture total epoch time
        self.epoch_end_time = time.time()
        self.epoch_duration = self.epoch_end_time - self.epoch_start_time

        # determine current run time
        self.run_duration = time.time() - self.run_start_time

        # append batch targets and predictions to dictionary
        self.track_validation_targets_and_predictions()

        # update metrics dictionary with epoch metrics
        self.track_epoch_metrics(mode="train")
        self.track_epoch_metrics(mode="validation")
        self.track_time()
        self.track_params()

        self.update_scalars()

        # print progress report
        if self.config.verbose:
            pass
        # if batch_idx % 50 == 0 and batch_idx > 0:
        #     print("\nTrain epoch: {} | Batch: {} | [Processed {}/{} ({:.0f}%)]\n\tLoss: {:.6f} | F1: {:.6f} | Precision: {:.6f} | Recall: {:.6f} | Accuracy: {:.6f}".format(
        #         epoch, batch_idx, len(epoch_predictions), len(self.train_data_loader.dataset),
        #         100. * len(epoch_predictions) / len(self.train_data_loader.dataset), train_loss.item(), metric_f1,
        #         metric_precision, metric_recall, metric_accuracy))
        #     print("\tBatch time elapsed: {}\n".format(self.train_timer(batch_beginning_time, time.time())))
        #     print("\n" + "*" * 10)

    def reset_network(self):
        """
        Documentation:

            ---
            Description:
                Reset network parameters.

        """
        for name, module in self.network.named_children():
            module.reset_parameters()

    def dump_metrics_to_csv(self):
        """
        Documentation:

            ---
            Description:
                Export metrics dictionary to csv.

        """
        # merge metadata and metrics dictionaries
        combined_dict = {**self.meta, **self.metrics}

        # insert metrics dictionary into dataframe
        self.results_df = pd.DataFrame(combined_dict, columns=combined_dict.keys())

        # append results dataframe to csv
        results_file = os.path.join(self.raw_logs_dir, "results.csv")
        with open(results_file, "a") as f:

            # if the csv already has data in it, append additional data without header
            try:
                _ = pd.read_csv(os.path.join(results_file))
                self.results_df.to_csv(f, header=False)

            # other append data with header
            except pd.io.common.EmptyDataError:
                self.results_df.to_csv(f, header=True)

    def begin_batch(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the beginning of a new batch.

        """
        # capture time stamp of batch start time
        self.batch_start_time = time.time()

        # increment number of runs initiated
        self.batch_count += 1

    def end_batch(self):
        """
        Documentation:

            ---
            Description:
                Complete operations associated with the end of a batch.

        """
        # subtract batch start time from current time to capture total batch time
        self.batch_end_time = time.time()
        self.batch_duration = self.batch_end_time - self.batch_start_time

        # append batch targets and predictions to dictionary
        self.track_train_targets_and_predictions()

        # add batch loss to cumulative epoch loss
        self.track_loss()

    def track_loss(self):
        """
        Documentation:

            ---
            Description:
                Accumulate loss for each batch in an epoch.

        """
        # add batch loss to cumulative epoch loss
        self.epoch_loss += self.batch_loss.item() * self.train_data_loader.batch_size

    def save(self, file_name):
        """
        Documentation:

            ---
            Description:
                a

        """
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{file_name}.csv')

        with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    def get_parameter_grid(self):
        """
        Documentation:

            ---
            Description:
                Use input parameter dictionary to generate all possible
                parameter combinations.

            ---
            Returns
                parameter_grid : list
                    s

        """
        run = namedtuple('Run', self.config.parameters.keys())

        parameter_grid = []
        for v in product(*self.config.parameters.values()):
            parameter_grid.append(run(*v))

        return parameter_grid

    def track_train_targets_and_predictions(self):
        """
        Documentation:

            ---
            Description:
                Determine prediction values, and collect batch targets
                and predictions.

        """
        # identify class prediction for each sample in batch output
        _, self.batch_predictions = torch.max(self.batch_output, dim=1)

        # append batch targets and predictions to epoch dictionary
        self.epoch_targets_and_predictions["train_targets"].extend(self.batch_targets.detach().cpu().numpy().tolist())
        self.epoch_targets_and_predictions["train_predictions"].extend(self.batch_predictions.detach().cpu().numpy().tolist())

    def track_validation_targets_and_predictions(self):
        """
        Documentation:

            ---
            Description:
                Determine prediction values, and collect batch targets
                and predictions.

        """
        # identify class prediction for each sample in batch output
        _, self.validation_predictions = torch.max(self.validation_output, dim=1)

        # append batch targets and predictions to epoch dictionary
        self.epoch_targets_and_predictions["validation_targets"].extend(self.validation_targets.detach().cpu().numpy().tolist())
        self.epoch_targets_and_predictions["validation_predictions"].extend(self.validation_predictions.detach().cpu().numpy().tolist())

    def track_time(self):
        """
        Documentation:

            ---
            Description:
                Collect runtime information.

        """
        # append run time metadata
        self.meta["run"].append(self.run_count)
        self.meta["run_duration"].append(self.run_duration)

        # append epoch time metadata
        self.meta["epoch"].append(self.epoch_count)
        self.meta["epoch_start_time"].append(self.epoch_start_time)
        self.meta["epoch_end_time"].append(self.epoch_end_time)
        self.meta["epoch_duration"].append(self.epoch_duration)

    def track_params(self):
        """
        Documentation:

            ---
            Description:
                Collect hyperparameter selections.

        """
        # add key/value for each hyperparameter
        for param, value in self.run_params.items():
            self.meta[param].append(value)

    def track_epoch_metrics(self, mode):
        """
        Documentation:

            ---
            Description:
                Collect epoch performance metrics.

        """
        # append epoch cumulative loss
        if mode == "train":
            self.metrics["train_loss"].append(self.epoch_loss)

        # append accuracy score
        self.metrics["{}_accuracy".format(mode)].append(
            accuracy_score(
                self.epoch_targets_and_predictions["{}_targets".format(mode)],
                self.epoch_targets_and_predictions["{}_predictions".format(mode)]
            )
        )
        # append number of correct predictions
        self.metrics["{}_number_correct".format(mode)].append((np.array(self.epoch_targets_and_predictions["{}_predictions".format(mode)]) == np.array(self.epoch_targets_and_predictions["{}_targets".format(mode)])).sum())

        ## if the learning task is a multiclass classificatin problem
        if self.is_multiclass:

            # append f1-macro score
            self.metrics["{}_f1_macro".format(mode)].append(
                f1_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                average="macro"
                )
            )
            # append f1-micro score
            self.metrics["{}_f1_micro".format(mode)].append(
                f1_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                    average="micro"
                )
            )
            self.metrics["{}_precision_macro".format(mode)].append(
                precision_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                average="macro"
                )
            )
            self.metrics["{}_precision_micro".format(mode)].append(
                precision_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                average="micro"
                )
            )
            self.metrics["{}_recall_macro".format(mode)].append(
                recall_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                average="macro"
                )
            )
            self.metrics["{}_recall_micro".format(mode)].append(
                recall_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)],
                average="micro"
                )
            )
        else:
            self.metrics["{}_f1".format(mode)].append(
                f1_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)]
                )
            )
            self.metrics["{}_precision".format(mode)].append(
                precision_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)]
                )
            )
            self.metrics["{}_recall".format(mode)].append(
                recall_score(
                    self.epoch_targets_and_predictions["{}_targets".format(mode)],
                    self.epoch_targets_and_predictions["{}_predictions".format(mode)]
                )
            )

    def execute(self):
        """
        Documentation:

            ---
            Description:
                a

        """
        #
        self.run_count = 0

        for run in self.get_parameter_grid():
            print(run)
            self.run_params = run._asdict()
            self.begin_run()

            # create data loader
            self.train_data_loader = torch.utils.data.DataLoader(
                self.config.train_data,
                batch_size=run.batch_size,
                shuffle=run.shuffle
            )

            # create optimizer
            self.optimizer = self.config.optimizer(
                self.network.parameters(),
                lr=run.lr
            )

            #
            for _ in range(self.config.epochs):
                self.begin_epoch()

                self.train()
                self.validation()

                self.end_epoch()
            self.end_run()
        # self.save('results')


    def train(self):
        """
        Documentation:

            ---
            Description:
                a

        """
        #
        for batch_idx, (batch_data, batch_targets) in enumerate(self.train_data_loader):
            self.begin_batch()

            # reformat data and targets
            self.batch_data = batch_data.flatten(start_dim=2).squeeze(1).to(self.config.device).float()
            self.batch_targets = batch_targets.to(self.config.device).long()

            # determine and track predictions
            self.batch_output = self.network(self.batch_data)

            # determine and track loss
            self.batch_loss = self.config.criterion(
                                            self.batch_output,
                                            self.batch_targets,
                                        )

            # adjust weights
            self.optimizer.zero_grad()
            self.batch_loss.backward()
            self.optimizer.step()

            #
            self.end_batch()

    def validation(self):
        # # sample batch number for data capture
        # num_batches = np.floor(len(self.validation_data_loader.dataset.image_paths) / self.validation_data_loader.batch_size)
        # sample_batch_idx = np.random.randint(0, num_batches)

        # turn off gradients
        self.network.eval()
        with torch.no_grad():

            # reformat data and targets
            self.validation_data = torch.from_numpy(self.config.validation_data.images).flatten(start_dim=1).to(self.config.device).float()
            self.validation_targets = torch.from_numpy(self.config.validation_data.targets).to(self.config.device).long()

            # determine and track predictions
            self.validation_output = self.network(self.validation_data)

# set model architecture
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.inputLayer = nn.Linear(784, 1024)
        self.fullyConnected1 = nn.Linear(1024, 1024)
        self.fullyConnected2 = nn.Linear(1024, 1024)
        self.fullyConnected3 = nn.Linear(1024, 1024)
        self.fullyConnected4 = nn.Linear(1024, 1024)
        self.fullyConnected5 = nn.Linear(1024, 1024)
        self.outputLayer = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.inputLayer(x))
        x = F.relu(self.fullyConnected1(x))
        x = F.relu(self.fullyConnected2(x))
        x = F.relu(self.fullyConnected3(x))
        x = F.relu(self.fullyConnected4(x))
        x = F.relu(self.fullyConnected5(x))
        x = F.log_softmax(self.outputLayer(x), dim=1)
        return x

# set input kwargs as object attributes
class ParamConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":

    ### load data
    ## training data
    # load source files
    X_train, y_train = load_mnist(
        path=os.path.join(os.environ["HOME"], "s3buckets", "mnist"),
        kind="train"
    )

    # transformation instructions
    norm_mean = [0.1307]
    norm_std = [0.3801]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            norm_mean,
            norm_std
        ),
    ])

    # load data into Pytorch Dataset
    train_data = MNISTDataset(
        # images=X_train,
        # targets=y_train,
        images=X_train[:10000,:],
        targets=y_train[:10000],
        transform=train_transform,
    )


    ## validation
    # load source files
    X_valid, y_valid = load_mnist(
        path=os.path.join(os.environ["HOME"], "s3buckets", "mnist"),
        kind="t10k"
    )

    # transformation instructions
    norm_mean = [0.1307]
    norm_std = [0.3801]

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            norm_mean,
            norm_std
        )
    ])

    # load data into Pytorch dataset
    validation_data = MNISTDataset(
        images=X_valid,
        targets=y_valid,
        transform=validation_transform,
    )


    # configure all necessary parameters
    run_params = ParamConfig(
        network = FCNet,
        network_name = "FCNet",
    #     model_object_dir = "/content/drive/model_objects/20191202_1622_VGG16",
    #     model_object_dir = None,
        optimizer = torch.optim.Adam,
        criterion = F.cross_entropy,
        train_data = train_data,
        validation_data = validation_data,
        cuda = True if torch.cuda.is_available() else False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed = 0,
        parameters = dict(
                lr = [.0001]
                ,batch_size = [1000]
                ,shuffle = [True,False]
            ),
        # parameters = dict(
        #         lr = [.01, .001, .0001]
        #         ,batch_size = [100, 500, 1000]
        #         ,shuffle = [True, False]
        #     ),
        epochs = 5,
        use_tensorboard = True,
        verbose = True,
        save_model_objects=True,
        )

    executor = RunExecuter(config=run_params)

    executor.execute()