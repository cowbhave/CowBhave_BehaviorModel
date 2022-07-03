# CowBhave_BehaviorModel

Models for classificaion of the feeding behavior.

The data for the model training is available at https://zenodo.org/record/6784671 at the folder Labeled25.zip.

The data for pretraning of the trasfer learning model is available at https://zenodo.org/record/4064802.

Function library *FeedingBehavior_NNlib.py* includes the following models:

1. CNN2 is a CNN with 2 convolutional layers from https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/.

2. CNN4 is a CNN with 4 convolutional layers from https://doi.org/10.3390/s21124050.

[//]: # (2. LSTM-CNN with 2 convolutional layers and 1 LSTM layer from https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

3. LSTM-CNN with 4 convolutional layers and 2 LSTM layer from https://doi.org/10.3390/s16010115

4. Random forest)

# Order of running the files
1. Preparing data H5PY files containing the training data by running `Main_FeedingBehaviour_PrepareDatasetH5PY.py`.
2. Training models for the ranges of the model types (NN1, NN4), traning datasets (Barn2), window sizes (5..300), folds for validation (1..10) by running `Main_FeedingBehaviour_CNNModelBuilding.py`.
3. Training model templates for transfer learning based on the BarnP dataset for the ranges of the model types (NN1, NN4), traning dataset (BarnP), window sizes (30,60,90), fold (0) by running `Main_FeedingBehaviour_CNNModelBuilding.py`.
4. Transfer learning of models based on the BarnP dataset by the Barn2 dataset for the ranges of the model types (NN1, NN4), traning dataset (Barn2), window sizes (30,60,90), frozen layers (0,1,2), folds for validation (1..10) by running `Main_FeedingBehaviour_CNNTransferLearning.py`.
