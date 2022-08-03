# CNN and transfer learning-based classification model for automated cow’s feeding behaviour recognition from accelerometer data

Thus repository contains the code for Bloch et al. 2022 preprint "CNN and transfer learning-based classification model for automated cow’s feeding behaviour recognition from accelerometer data" https://doi.org/10.1101/2022.07.03.498612. Code includes functions for preparing data, training CNN models for cow feeding behavior classification and validation the models. With the help of these functions, dependance between the dataset size and the model accuracy is analyzed.

The data for the model training is available at https://zenodo.org/record/6784671 at the folder Labeled25.zip. This dataset was collected in a research barn (called Barn2) during a barn experiment. For analysis of dependance between the dataset size and the model accuracy, partial datasets were used. The partial datasets are created from the original dataset Barn2.

The data for pretraining of the transfer learning model is available at https://zenodo.org/record/4064802.

Function library *FeedingBehavior_NNlib.py* includes the following models:

1. CNN2 is a CNN with 2 convolutional layers from https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/.

2. CNN4 is a CNN with 4 convolutional layers from https://doi.org/10.3390/s21124050.


# Order of running the files
1. Preparing data H5PY files containing the training data by running `Main_FeedingBehaviour_PrepareDatasetH5PY.py`.
2. Training models for the ranges of the model types (CNN2, CNN4), training datasets (Barn2), window sizes (5..300), folds for validation (1..10) by running `Main_FeedingBehaviour_CNNModelBuilding.py`.
3. Training pre-trained model for transfer learning based on the BarnP dataset for the ranges of the model types (CNN2, CNN4), training dataset (BarnP), window sizes (30,60,90), fold (0) by running `Main_FeedingBehaviour_CNNModelBuilding.py`.
4. Transfer learning of models based on the BarnP dataset by the Barn2 dataset for the ranges of the model types (CNN2, CNN4), training dataset (Barn2), window sizes (30,60,90), frozen layers (0,1,2), folds for validation (1..10) by running `Main_FeedingBehaviour_CNNTransferLearning.py`.


# Preparing partial datasets
1. Preparing datasets including specific number of the training samples by running `Main_DatasetDecreasing_Samples.m`.
2. Preparing datasets including specific percentage of the original dataset by running `Main_DatasetDecreasing_Percent.m`.
