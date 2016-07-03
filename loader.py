import os
import collections
from six.moves import cPickle
import numpy as np 
import re 
import csv
from ast import literal_eval


# For pretrained embeddings:
# generate np array: 
# -vocab map: word String -> token id
# -embedding map: token_id -> embeddings


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        ''' input_file is data/textfile
            vocab_file is list of individual words/words from most freq to least freq
            tensor_file is each element of the data/textfile mapped to its corresponding word-id
            embed_file is np file of pretrained GloVe word embeddings for vocab words

            self.vocab is dict that maps word to id, with {most frequent word: 0} '''

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        embed_file = os.path.join(data_dir, "embed.npy") 

        if not (os.path.exists(input_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)

        # ONLY NECESSARY if using pretrained embeddings such as word2vec or GloVE
        # if not (os.path.exists(embed_file)):
        #     print("compiling word embeddings")
        #     read_file = os.path.join(data_dir, "glove_300d.txt")
        #     self.compile_embeddings(read_file, embed_file)

        self.create_batches()
        self.reset_batch_pointer() 

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()
        data = re.findall(r"\w+|\n|[^\w\s]", data) 
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # descending frequency count of words
        self.words, _ = zip(*count_pairs) 
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(self.vocab_size))) # word map to id based on frequency (i.e. most frequent: 0)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f) # words from most freq to least freq
        self.tensor = np.array(list(map(self.vocab.get, data))) # array: maps input data (words) to corresponding vocab id 
        np.save(tensor_file, self.tensor) # so tensor is input to model

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(self.vocab_size)))
        self.tensor = np.load(tensor_file)

    def compile_embeddings(self, read_file, embed_file):
        # if csv dict mapping from word to embedding does not exist, create it from txt file
        if not os.path.exists(os.path.join(self.data_dir, 'embdict.csv')):   
            data = ''
            with open(read_file, "r") as f:
                for line in f:
                    data += line # necessary for files of large size
            print('file read')
            data_arr = data.split('\n')
            print('data split')
            
            with open(os.path.join(self.data_dir, 'embdict.csv'), 'w+') as f:
                fieldnames = ['word', 'embedding']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, x in enumerate(data_arr):
                    m = re.search('\-?\d+', x)
                    if m:
                        # TODO: solve for situations where word has digit (i.e. "1st")
                        start_emb = m.start() # start of embedding
                        word = (x[:start_emb]).strip()
                        emb = x[start_emb:].split()
                        writer.writerow({'word': word, 'embedding': emb})
                    else:
                        print(i)
                        print("ERROR finding embedding")
                        print(x)

                    if (i%50000)==0:
                        print(i)
            print('embdict created')

        # to catch errors from above situation (e.g. '1st' got sorted into word embedding)
        def _is_float(i):
            try:
                float(i)
                return True
            except ValueError:
                print(i)
                return False

        embdict = {}
        with open(os.path.join(self.data_dir, "embdict.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                embed_str = literal_eval(row['embedding']) # get rid of quotes around list, e.g. "['1.34', '2.76']"

                # turn elements in list back to floats, e.g. ['1.34', '2.76'] => [1.34, 2.76], if error then init to zeros
                embed = [float(i) if _is_float(i) else 0. for i in embed_str] 
                embdict[row['word']] = embed
        print("embdict loaded")

        count = 0
        npmat = np.zeros((self.vocab_size, 300))
        for word, token_id in self.vocab.items():
            word = word.lower() # normalize
            if word in embdict:
                npmat[token_id] = embdict[word][:300]
                if len(embdict[word]) != 300:
                    print(len(embdict[word]))
                    print(embdict[word])
            else: # if not in pretrained vocab, init to zeros
                npmat[token_id] = [0] * 300
                print(word)
                count += 1
        print('npmat created')
        print('not in pretrained vocab: ' + str(count))
        np.save(embed_file, npmat)


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


