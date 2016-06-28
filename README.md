# hp-word-model
Word-level LSTM to train language models. The data directory provides the concatenatation of all seven Harry Potter books in a text file as input, but any piece of text can be used for the basis of the model.

To use, clone the directory, run train.py to train the model, and sample.py to generate arbitrarily long text samples.

The code also provides the option of using pretrained word embeddings such as word2vec or GloVe. 