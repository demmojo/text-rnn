# text-rnn

text-rnn allows you to create modern neural network architectures which use modern techniques such as skip-embedding and attention weighting. It trains and generate text at the character-level. It also uses the CuDNN implementation when trained on GPUs which significantly improves training time when compared to the usual implementation of LSTMs. 

You can configure whether to use bidirectional RNNs, the number of RNN layers, RNN size, input length, and size of the embedding layer. 

If you would like to train using a free GPU check out this [Colaboratory notebook](https://colab.research.google.com/github/demmojo/colabrnn/blob/master/colabRNN.ipynb).
