# Imports
import numpy as np
import pandas as pd

from opendataval.dataloader import Register, DataFetcher, mix_labels, add_gauss_noise
from opendataval.dataval import (
    AME,
    DVRL,
    BetaShapley,
    DataBanzhaf,
    DataOob,
    DataShapley,
    InfluenceSubsample,
    KNNShapley,
    LavaEvaluator,
    LeaveOneOut,
    RandomEvaluator,
    RobustVolumeShapley,
)

from opendataval.experiment import ExperimentMediator

from opendataval.model.api import ClassifierSkLearnWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set up hyperparameters
dataset_name = "random_dataset"
train_count, valid_count, test_count = 50, 10, 10
noise_rate = 0.1
model_name = "sklogreg"
metric_name = "accuracy"

# Generate a random dataset
# Every element of X is generated from a standard Gaussian distribution
X, y= np.random.normal(size=(100, 10)), np.random.choice([0,1], size=100)
# Register a dataset from arrays X and y
pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(X, y)

# After regitering a dataset, we can define `DataFetcher` by its name.
fetcher = (
    DataFetcher(dataset_name, '../data_files/', False)
    .split_dataset_by_count(train_count,
                            valid_count,
                            test_count)  
    .noisify(mix_labels, noise_rate=noise_rate)
)

# pred_model = ClassifierSkLearnWrapper(LogisticRegression, fetcher.label_dim[0]) # example of Logistic regression
pred_model = ClassifierSkLearnWrapper(RandomForestClassifier, fetcher.label_dim[0])
exper_med = ExperimentMediator(fetcher, pred_model)

Data_Evaluators  = [
    LavaEvaluator,
    RobustVolumeShapley
]

experiment_mediator = exper_med.compute_data_values(data_evaluators = Data_Evaluators)




from opendataval.experiment.exper_methods import (

    noisy_detection,
)
from matplotlib import pyplot as plt

# Saving the results
output_dir = f"../tmp/{dataset_name}_{noise_rate=}/"
exper_med.set_output_directory(output_dir)
output_dir



exper_med.evaluate(noisy_detection, save_output=True)

