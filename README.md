# Deep Short Text Classification with Knowledge Powered Attention Re-Implementation


## Python Version Requirements
- Python==3.7.4
- pytorch==1.3.1
- torchtext==0.3.1
- numpy
- tqdm


## 1) Running the code
Train & Dev & Test:
The dataset is split randomly to dedicate 80% of the data for training and 20% for testing. The 20% of the data is used to create the development dataset. 

The arguments to execute the program is as follows:
* --epoch: Number of epochs to run
* --lr: The learning rate
* --train_data_path: The file path to access the training dataset
* --txt_embedding_path: The file path to access the pretrained word vectors for input short text
* --cpt_embedding_path: The file path to access the pretrained word vectors for concept set
* --embedding_dim: The embedding dimension
* --train_batch_size: The batch_size for the tensors
* --hidden_size: The hidden dimension

An example of how to run the code is shown below:

```console
python3 main.py --epoch 100 --lr 2e-4 --train_data_path dataset/tagmynews.tsv --txt_embedding_path dataset/glove.6B.300d.txt --cpt_embedding_path dataset/glove.6B.300d.txt  --embedding_dim 300 --train_batch_size 128 --hidden_size 64
```

During the execution of the code, you will be asked two things:
1) Which network to implement for STCKA? Linear-STCKA or CNN-STCKA. Respond cnn for CNN-STCKA and linear for Linear-STCKA
2) Whether you want to print the number of parameters of the model? Respond y for yes and n for no

## 2)
a) The following code files are copied from the author's GitHub repository:
- model/__init__.py
- utils/__init__.py
- utils/config.py
- utils/dataset.py
- utils/metrics.py

b) The following code files have been modified:
- main.py => The training and testing loops were re-implemented and written by me. In addition, I included two input() functions to choose between linear and cnn networks for the STCKA and whether or not you want to print out the number of parameters of the model. Also included a print statement to understand the runtime of the model.

c) The following code files are mine:
- model/STCKA.py

## 3) Data Files
The snippets.tsv and tagmynews.tsv were obtained from the Author's GitHub repository. The two tsv files contain the input short sentence, the hard-coded concept set, and the target class. The glove.6B.300d.txt pretrained word vectors were obtained from Kaggle

## Citations
Author's Paper: https://aaai.org/ojs/index.php/AAAI/article/view/4585/4463
Author GitHub Repository: https://github.com/AIRobotZhang/STCKA
GloVe: https://www.kaggle.com/thanakomsn/glove6b300dtxt
