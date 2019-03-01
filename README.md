# BiRNN-SRL
This is the re-built code for the paper “Chinese Semantic Role Labeling with Bidirectional Recurrent Neural
Networks” http://www.aclweb.org/anthology/D15-1186
## Prerequisites
+ python3
+ tensorflow
## Data

> In this dataset, we extracted the same features propesed in this picture and the processed data has the following format:

> the whole block represents a sentence, and each line contains information about the feature information of a word


>> ![image](https://github.com/zxplkyy/BiRNN-SRL/blob/master/example.PNG)
## Training and Validating
>The src folder contains the source codes, which contains the following files:
+ **src/data_utils.py**  original data processing
+ **src/bilstm.py**  model definition
+ **src/train.py**   model training and validation
+ **src/config.py**  the congiuration of  hyperparameters used in model
>> You can execute **python train.py** to train the model. The training loss  and the result  of validation set for each epoch will be output to console
