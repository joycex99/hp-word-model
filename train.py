from __future__ import print_function 
import numpy as np 
import tensorflow as tf 
import numpy as np
import argparse 
import time
import os
from six.moves import cPickle

from loader import TextLoader 
from model import Model 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/hp',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=300,
                       help='size of RNN hidden state, i.e. num hidden cells')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50, #50,
                       help='RNN sequence length, i.e. num steps to unfold')
    parser.add_argument('--num_epochs', type=int, default=50, #50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency (num batches)')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005, #0.02
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                       help='probability of keeping inputs/outputs during dropout')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size 

    # check compatibility 
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config & check model compatibility
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model", "rnn_size", "num_layers", "seq_length"]
        for tocheck in need_be_same:
            # ?? next line
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], "Command line argument and saved model disagree on '%s' " %checkme

        # check vocab/dict combatibility 
        with open(os.path.join(args.init_from, 'words_vocab.pkl')) as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words == data_loader.words, "Data & lodaded model disagree on words"
        # ? what does vocab have exactly again?
        assert saved_vocab == data_loader.vocab, "Data & lodaded model disagree on dict mappings"

    # ?? next lines
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        sess_start = time.time()
        print(time.ctime(int(time.time()))) 
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 3)
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # load embeddings IF USING PRETRAINED ONES
        # embedding = np.load(os.path.join(args.data_dir, 'embed.npy'))
            
        for e in range(args.num_epochs):
            # update lr based on epoch, reset pointer to beginning, load state
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()

            # pretrained embeddings:
            # sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding})
 
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op],
                                                {model.input_data: x,
                                                 model.targets: y,
                                                 model.initial_state: state})
                end = time.time()
                current_batch = e * data_loader.num_batches + b
                if current_batch % 100 == 0:
                    # print: current batch/total batches, epoch, loss, time
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                         .format(current_batch,
                                 args.num_epochs * data_loader.num_batches,
                                 e, train_loss, end-start))
                # if time to save, or is last result
                if current_batch % args.save_every == 0\
                    or (e==args.num_epochs-1 and b==data_loader.num_batches-1): 
                    checkpoint_path = os.path.join(args.save_dir, 'model.cpkt')
                    saver.save(sess, checkpoint_path, global_step = current_batch)
                    print("model saved to {}".format(checkpoint_path))
        sess_end = time.time()
        total_time = sess_end - sess_start
        print("Total time: {} seconds = {} hours".format(total_time, total_time/3600))

if __name__ == '__main__':
    main()
