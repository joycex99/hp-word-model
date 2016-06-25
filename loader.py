import os
import collections
from six.moves import cPickle
import numpy as np 
import re 


# For pretrained embeddings:
# generate np array: 
# -vocab map: word String -> token id
# -embedding map: token_id -> embeddings
# 
# sess.run(model.embedding.assign(w2v_embeddings ))
# OR
# W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                 trainable=False, name="W")

# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# embedding_init = W.assign(embedding_placeholder)

# # ...
# sess = tf.Session()

# sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})



# embdict:
#   data = f.read().split('\n')
#   obj = {x.split()[0]: x.split()[1:] for x in data}

# npmat.shape = (vocab_size, 300)
# for i in range(len(vocab_size)):
#   word = vocab[i]
#   emb = embdict[word]
#   npmat[i] = emb


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        ''' input_file is data/textfile
            vocab_file is list of individual words/words from most freq to least freq
            tensor_file is each element of the data/textfile mapped to its corresponding word-id 

            self.vocab is dict that maps word to id, with {most frequent word: 0} '''

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(input_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer() 

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()
        data = re.findall(r"\w+|\n|[^\w\s]", data)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # descending frequencing count of words
        self.words, _ = zip(*count_pairs) # get just the words (keys of dict)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(self.vocab_size))) # word map to id based on frequency (i.e. most frequent: 0)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f) # words from most freq to least freq
        self.tensor = np.array(list(map(self.vocab.get, data))) # array: maps input data (words) to corresponding vocab id 
        np.save(tensor_file, self.tensor) # so tensor is input to model!! 

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(self.vocab_size)))
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size smaller"

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length] # ??? 
        xdata = self.tensor 
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0] # ydata is xdata shifted over by 1

        # reshape into batch_size number of rows, then "split" data into num_batches parts along y-axis
        # each i of x_batches[i] refers to ith batch
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1) 
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1) 

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer] # current batch
        self.pointer += 1 
        return x, y 

    def reset_batch_pointer(self):
        self.pointer = 0


