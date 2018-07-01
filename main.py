from rnn import CharGen
from rnn.train_model import train

train_new_model = False
model_name = 'shakespeare'

if train_new_model:
    textgen = CharGen(name=model_name,
                      bidirectional=True,
                      rnn_size=128,
                      rnn_layers=3,
                      embedding_dims=75,  # Size of the embedding layer (default 75)
                      input_length=60  # Number of characters considered for prediction (default 60)
                      )
    train(text_filepath='datasets/shakespeare.txt',
          textgen=textgen,
          num_epochs=10,
          batch_size=512,
          train_new_model=train_new_model)

    print(textgen.model.summary())
else:
    textgen = CharGen(weights_filepath='models/{}_weights.hdf5'.format(model_name),
                      vocab_filepath='models/{}_vocabulary.json'.format(model_name),
                      config_filepath='models/{}_config.json'.format(model_name))

    train('datasets/shakespeare.txt', textgen, train_new_model=train_new_model, num_epochs=10)
