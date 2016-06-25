import tensorflow as tf
import numpy as np
import re

''' FOR REFERENCE:
args.seq_length = num steps
rnn_size = hidden size, or num hidden units '''

flags = tf.flags

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        # add support for GRU cell; for now, only lstm
        cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size, forget_bias=1.0) # forget bias was added

        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size]) # num cells in next layer (hidden) by num cells in input (vocab)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size]) # only input (vocab) connects with bias weights

            # Q: why must we do this on the cpu?
            with tf.device('/cpu:0'):
                # Q: why must embedding be the same length as num of hidden units?
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # tf.split(split_dim, num_split, value) splits value along split_dim into num_split smaller tensors
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1)) # ??? tf.stop_gradient prevents individual tensor from contributing to gradients - but why argmax(prev, 1)?
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py
        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits) # normalize probs 

        # returns log perplexity for each seq in batch (see link above)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.logits], # batch size x num decoder symbols
            [tf.reshape(self.targets, [-1])], # flattens to 1-d, same length as logits
            [tf.ones([args.batch_size * args.seq_length])] # same length as logits
        )

        # vs. seq2seq.sequence_loss: seq_loss returns scalar val that is average perplexity of vals
        # so if we call rf.reduce_sum(loss) / args.batch_size / args.seq_length, 
        #    why not just do seq2seq.sequence_loss / args.seq_length?

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length 
        self.final_state = last_state 
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr) # research optimizers
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime="Harry ", sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        prime = re.findall(r"\w+|\n|[^\w\s]", prime)

        # state gets updated with each word
        for word in prime[:-1]: # for everything up to the last word
            x = np.zeros((1, 1)) # array([[0.]])
            x[0, 0] = vocab[word] # word-id (guess)
            [state] = sess.run([self.final_state], {self.input_data: x, self.initial_state:state})

        def weighted_pick(weights):
            t = np.cumsum(weights) # cumulative sum over flattened array at each step
            s = np.sum(weights) 
            return(int(np.searchsorted(t, np.random.rand(1)*s))) # ??

        ret = ''.join(prime)
        word = prime[-1] # last word
        for n in range(num): 
            x = np.zeros((1, 1)) 
            x[0, 0] = vocab[word]
            # replacement for top two lines: x = np.array([[vocab[word]]]) 

            # state input is, at point of input, reflective of everything EXCEPT word
            [probs, state] = sess.run([self.probs, self.final_state], 
                                      {self.input_data: x, self.initial_state:state})
            p = probs[0] # ? what is p?

            # ?? what are these sampling types?
            # "0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces"

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p) 
            else: # default 
                sample = weighted_pick(p)

            pred = words[sample] 

            ret += pred
            word = pred 
        return ret 

