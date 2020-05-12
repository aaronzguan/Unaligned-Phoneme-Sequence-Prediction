# Unaligned Phoneme Sequence Prediction

This is my codes for the Kaggle Competition for the CMU-11785 Introduction to Deep Learning. The competition details can be found in [Kaggle](https://www.kaggle.com/c/11-785-s20-hw3p2).

## Data
The data can be downloaded at [data](https://www.kaggle.com/c/11-785-s20-hw3p2/data).
 
The data is the mel-spectrograms that have 40 band frequencies for each time step of the speech data, whose dimensions are [frames, time step, 40]. The labels are the index of the phonemes in the utterance [0-45] and will not have a direct mapping of each time step of the feature, whose dimensions are [frames, labels len]. The second dimision will have variable length which has no correlation to the time step dimension in feature data.

## Architecture
* 5 convolutional layers as feature extractor. The first two convolutional layers have 128 units and the last three convolutional layers have 256 units. Each convolutional layer uses kernel size 3, padding 1 and stride 1 to make sure the output size is the same as the input size. Each convolutional layer is followed by ReLU activation layer and BatchNorm layer subsequently.
* After 5 convolutional layers, 4 stacked BiLSTM layers, each of 512 units are used.
* The BiLSTM is followed by two linear layers used as classifier, the first layer has
512 units and the last layer is of 47 (number of classes) units.

## CTC Loss
Since there is no alignment between utterances and their corresponding phonemes. Thus, train the network using CTC loss. Decode the predictions using beam search.

## Requirements
- [python-Levenshtein](https://pypi.org/project/python-Levenshtein/)
- [ctcdecode](https://github.com/parlance/ctcdecode)