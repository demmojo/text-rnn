from rnn import CharGen
from rnn.train_model import train

train_new_model = True

if train_new_model:
    textgen = CharGen(name="shakespeare",
                      bidirectional=True,
                      rnn_size=128,
                      rnn_layers=3,
                      embedding_dims=75,  # Size of the embedding layer (default 75)
                      input_length=60  # Number of characters considered for prediction (default 60)
                      )
    train(text_filepath='datasets/test.txt',
          textgen=textgen,
          num_epochs=10,
          batch_size=512,
          train_new_model=train_new_model)

    print(textgen.model.summary())
else:
    textgen = CharGen(weights_filepath='models/test_weights.hdf5',
                      vocab_filepath='models/test_vocabulary.json',
                      config_filepath='models/test_config.json')

    train('datasets/test.txt', textgen, train_new_model=train_new_model, num_epochs=10)
