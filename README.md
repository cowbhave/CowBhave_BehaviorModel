# CowBhave_BehaviorModel

Models for classificaion of the feeding behavior.

Function library *FeedingBehavior_NNlib.py* includes the following models:

1. CNN with 2 convolutional layers from https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/.

2. LSTM-CNN with 2 convolutional layers and 1 LSTM layer from https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

3. LSTM-CNN with 4 convolutional layers and 2 LSTM layer from https://doi.org/10.3390/s16010115

4. Random forest

# Odred of running the files
1. Preparing data H5PY files containing the training data by running *Main_FeedingBehaviour_PrepareDatasetH5PY.py*.
2. Training models for the given ranges of the model types (NN1, NN4), traning datasets (Barn2, BarnP), folds for validation (1..10) by running *Main_FeedingBehaviour_CNNModelBuilding.py*.
3. Transfer learning of models based on the BarnP dataset by the Barn2 dataset by running *Main_FeedingBehaviour_CNNTransferLearning.py*.
