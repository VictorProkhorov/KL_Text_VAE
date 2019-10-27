import os
import sys
import math
import time
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.reset_default_graph()
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
import tensorflow.contrib.eager as tfe
import numpy as np
from scipy.linalg import orth
import six
import argparse


# many parts of the code were taken from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb

is_gpu = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

if is_gpu == True:
    print("Using GPU")
else:
    print("Using CPU")

#### LOADING DATA ####

class LanguageIndex():
    def  __init__(self, lang=None, vocab=set(), is_vocab=False):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = vocab
        if is_vocab == False:
            self.create_index_with_sentences()
        else:
            self.vocab = sorted(self.vocab)
            self.create_index_with_vocab()
    
    def create_index_with_vocab(self):
        
        self.vocab.append('<EOS>')
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
  
    def create_index_with_sentences(self):
        for sentence in self.lang:
            self.vocab.update(sentence.split(' '))
        self.vocab = sorted(self.vocab)
        self.vocab.append('<EOS>')
        if '_UNK' not in self.vocab:
            self.vocab.append('_UNK')
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            # + 1 to take into account <pad>
            self.word2idx[word] = index + 1
    
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
    

def preprocess_sentence(sentence):
    #w = unicode_to_ascii(w.strip())
   # sentence =  sentence
    return sentence


def create_dataset(path):
    
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    word_pairs = [preprocess_sentence(line.lstrip().rstrip()) for line in lines if len(line) > 0]
    
    return word_pairs


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0., eos=1.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
        eos = end of sentence index to end each sentence
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen+1) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen] + [float(eos)]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x




def get_vocab(vocab_file):
    vocab = set()
    with open(vocab_file, 'r') as f:
        for idx,word in enumerate(f):
           # if not word.rstrip():
           #     print ('hi', word, idx+1)
            word = word.rstrip()
            vocab.add(word)
    return vocab


def load_dataset(path,  text_lang=None, vocab_file=None):
    # creating cleaned input, output pairs
    sentences = create_dataset(path)
    print('Number of sentences:', len(sentences))
    # index language using the class defined above   
    if text_lang == None: 
        if vocab_file is not None:
            vocab = get_vocab(vocab_file)
            print('loaded vocab:', len(vocab))
            text_lang = LanguageIndex(vocab=vocab, is_vocab=True)
        else:
            text_lang = LanguageIndex(lang=sentences)
    
    print('Number of unique words in text:', len(text_lang.vocab))
    
   
    # text definitions of concepts
    unk = '<unk>'
   # unk = '_UNK'
    text_tensor = [[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in txt.split(' ')] for txt in sentences]
    
    
    def max_length(tensor):
        return max([len(t) for t in tensor])

    max_length_text = max_length(text_tensor)

    # Padding the input and output tensor to the maximum length
    text_tensor = pad_sequences(text_tensor, maxlen=max_length_text, padding='post', truncating='post', eos=text_lang.word2idx['<EOS>'])
    
    max_length_text = max_length_text + 1 # add one to compensate for <EOS>
    print('the longest sentence is of N symbols:', max_length_text)
    return text_tensor, text_lang, max_length_text 


#### MODELS ####

def last_relevant(output, length):
    "taken form: https://danijar.com/variable-sequence-lengths-in-tensorflow/"
    #print('out', output.shape)
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length -1 )
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

class Encoder(tf.keras.Model):
    def __init__(self, rnn_dim, z_dim, model):
        super(Encoder, self).__init__()
        self.rnn = self._init_rnn(rnn_dim, model)
        self.mean = tf.keras.layers.Dense(z_dim, activation='linear')
        self.log_var = tf.keras.layers.Dense(z_dim, activation='linear')


    def call(self, x, embeddings, annealing_step, n_annealing_steps, C_target, is_evaluate=False):
        outputs = self.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.mean(output)
        log_var = self.log_var(output)
        z = self._sampling([mean, log_var])

        kl_loss, kl_plot, C = self.kl_div_loss(log_var, mean, annealing_step, n_annealing_steps, C_target, is_evaluate=is_evaluate)

        return z, kl_loss, kl_plot, C
    
    def _init_rnn(self, units, model):
        if model == 'LSTM-LSTM' or model == 'LSTM-CONV':
             return tf.compat.v1.keras.layers.CuDNNLSTM(units,kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal' , return_state=False, return_sequences=True)
        else:
             return tf.compat.v1.keras.layers.CuDNNGRU(units,kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal' , return_state=False, return_sequences=True)
        

    @staticmethod
    def kl_div_loss(z_log_var, z_mean, annealing_step, n_annealing_steps, C_target, is_evaluate=False):
        beta = 1
        kl_loss = tf.reduce_mean(-0.5 * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), -1))
        if is_evaluate == False:
            C = min(C_target, C_target*annealing_step/n_annealing_steps)
            return beta*tf.abs(kl_loss - C), kl_loss, C
        else: 
            return kl_loss, kl_loss, 0

    @staticmethod
    def _sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], tf.keras.backend.shape(z_mean)[1]), mean=0., stddev=1)
        return z_mean + tf.keras.backend.exp(z_log_var / 2) * epsilon


class Decoder_RNN(tf.keras.Model):
    def __init__(self, rnn_dim, vocab_size, model):
        super(Decoder_RNN, self).__init__()
        self.rnn = self._init_rnn(rnn_dim, model)
        self.z_expander = tf.keras.layers.Dense(rnn_dim, activation='linear')
        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, x, embeddings):
        
        #hs, _, _ = self.rnn(embeddings)#, initial_state=[h, c])
        result = self.rnn(embeddings)
        #print(result)
        hs = result[0]
        predictions = self.out(hs)
        loss = self.reconstruction_loss_function(x, predictions)
        return loss
    
    def _init_rnn(self, units, model):
        if model == 'LSTM-LSTM':
            return tf.compat.v1.keras.layers.CuDNNLSTM(units, kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal', return_state=True, return_sequences=True)
        elif model == 'GRU-GRU':
            return tf.compat.v1.keras.layers.CuDNNGRU(units, kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal', return_state=True, return_sequences=True)
        else:
            print('Error: you have chosen a wrong neural network. In this decoder (Decoder_RNN) one can only choose either LSTM or GRU')
 
   
    @staticmethod
    def reconstruction_loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        x_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        xe_loss = tf.reduce_sum(x_, axis=-1)
        return xe_loss


class ConvLM(tf.keras.Model):
    def __init__(self, context, internal_dim, external_dim):
        super(ConvLM, self).__init__()
        
        self.to_output_dim = tf.keras.layers.Conv1D(external_dim, 1, padding='valid', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))
        
        self.conv_A_1 = tf.keras.layers.Conv1D(internal_dim, 1, padding='valid', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))
        self.conv_A_2 = tf.keras.layers.Conv1D(internal_dim, context, padding='causal', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))
        self.conv_A_3 = tf.keras.layers.Conv1D(external_dim, 1, padding='valid', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))

        self.conv_B_1 = tf.keras.layers.Conv1D(internal_dim, 1, padding='valid', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))
        self.conv_B_2 = tf.keras.layers.Conv1D(internal_dim, context, padding='causal', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))
        self.conv_B_3 = tf.keras.layers.Conv1D(external_dim, 1, padding='valid', kernel_initializer=tf.keras.initializers.he_uniform(seed=None))

    def call(self, x):
        x =  self.to_output_dim(x)
        A_1 = self.conv_A_1(x)
        A_1 = tf.nn.dropout(A_1, 0.9)
        A_2 = self.conv_A_2(A_1)
        A_2 = tf.nn.dropout(A_2, 0.9)
        A = self.conv_A_3(A_2)
        A = tf.nn.dropout(A, 0.9)


        B_1 = self.conv_B_1(x)
        B_1 = tf.nn.dropout(B_1, 0.9)
        B_2 = self.conv_B_2(B_1)
        B_2 = tf.nn.dropout(B_2, 0.9)
        B = self.conv_B_3(B_2)
        B = tf.nn.dropout(B, 0.9)
        
        B = tf.keras.activations.sigmoid(B)
        hs = A * B
        hs = hs + x
        return hs


class Decoder_CNN(tf.keras.Model):
    def __init__(self, cnn_dim,  vocab_size):
        super(Decoder_CNN, self).__init__()
    
        
        self.context = 20
        external_dim = cnn_dim
        internal_dim = int(cnn_dim/4)
        self.conv_lm1 = ConvLM(self.context, internal_dim, external_dim)
        self.conv_lm2 = ConvLM(self.context, internal_dim, external_dim)
        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, x, embeddings):
        
        
        hs = self.conv_lm1(embeddings)
        hs = self.conv_lm2(hs)
        predictions = self.out(hs)
        
        loss = self.reconstruction_loss_function(x, predictions)
        return loss
    
    @staticmethod
    def reconstruction_loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        x_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        xe_loss = tf.reduce_sum(x_, axis=-1)
        return xe_loss




class Sentence_VAE(tf.keras.Model):
    def __init__(self, embed_dim, vocab_size, rnn_dim, z_dim, model=0):
        super(Sentence_VAE, self).__init__()
        self.model_type = ['LSTM-LSTM', 'GRU-GRU', 'LSTM-CONV']
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        ### ENCODER ###
        self.encoder = Encoder(rnn_dim, z_dim, model = self.model_type[model])
        ### DECODER ###
        if self.model_type[model] == 'LSTM-LSTM' or self.model_type[model] == 'GRU-GRU':
            self.decoder = Decoder_RNN(rnn_dim,  vocab_size, model = self.model_type[model])
        else:
            self.decoder = Decoder_CNN(rnn_dim,  vocab_size)
       
    
   

    def call(self, x, batch_size, text_lang, annealing_step, n_annealing_steps, C_target, is_evaluate=False, is_random_evaluate=False):
        ### ENCODING ###
        embeddings = self.embeddings(x)
        z, kl_loss, kl_plot, C = self.encoder(x, embeddings, annealing_step, n_annealing_steps, C_target, is_evaluate=is_evaluate)
        

        ### DECODING ###
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
        in_ = tf.concat([dec_input, x], axis=-1)
        embeddings = self.embeddings(in_[:,:-1])
        embeddings = tf.keras.layers.concatenate([embeddings,tf.keras.layers.RepeatVector(embeddings.shape[1])(z)])
        loss = self.decoder(x, embeddings)
        
        return loss, kl_loss, kl_plot, C

    def sample_z(self, x):
        embeddings = self.embeddings(x)
        outputs = self.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = self.encoder.mean(output)
        log_var = self.encoder.log_var(output)
        z = self.encoder._sampling([mean, log_var])
        return z


global_step = tf.train.get_or_create_global_step()


def train(epochs, buffer_size, vae, dataset, optimizer, text_lang, n_batch, checkpoint, valid_data, n_annealing_steps, C_target):
    device = "/cpu:0"
    if is_gpu:
        device="/gpu:0"
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(len(valid_data))
    valid_batch_size = 128
    valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=False)
    annealing_initial_step = 0
    for epoch in range(1, epochs + 1):
        global_step.assign_add(1)
        start = time.time()
        total_loss = 0
        total_kl_loss = 0 # difference between C and KL
        total_kl_plot = 0 # actual KL

        dataset = dataset.shuffle(buffer_size)
        for (batch, sentences) in enumerate(dataset):
            batch_size = sentences.shape[0]
            annealing_initial_step += 1
            with tf.device(device):
                with tf.GradientTape() as tape:
                    rec_loss, kl_loss, kl_plot, _= vae(sentences, batch_size, text_lang, annealing_initial_step, n_annealing_steps, C_target)
                    loss = kl_loss + tf.reduce_mean(rec_loss)
               
                variables = vae.variables 
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables), global_step)
                
    
            batch_loss = loss 
            total_loss += tf.reduce_mean(rec_loss) #batch_loss
            total_kl_loss += kl_loss
            total_kl_plot += kl_plot          
                 
            if batch % 100 == 0:
                    print('Epoch {} Batch {} Train Loss {:.4f}'.format(epoch,
                                                         batch,
                                                         batch_loss.numpy()))
                
    
        print('Epoch {}, NLL Loss {:.4f},  KL Loss {:.4f}, KL Loss PLot {:.4f}'.format(epoch, total_loss/n_batch,  total_kl_loss/n_batch, total_kl_plot/n_batch))

        
        if epoch % 1 == 0:
           
            valid_rec_loss, valid_kl_loss = evaluate_nnl_and_rate_batch(valid_dataset, vae, text_lang)
            print('Epoch {}; Valid Rec Loss {:.4f};  Valid KL-loss {:.4f}'.format(epoch,valid_rec_loss, valid_kl_loss))
        

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    save = checkpoint.save(file_prefix=checkpoint_prefix)
    print('save', save)


#### BASIC EVALUATION ####



def evaluate_nnl_and_rate_batch(data, vae, text_lang, is_random_evaluate=False):

    valid_rec_loss = 0
    valid_kl_loss = 0
    n_epochs = 0
    for x in data:

        batch_size = x.shape[0]
        rec_loss, kl_loss, _ , _ = vae(x, batch_size, text_lang, 0, 0, 0, is_evaluate=True, is_random_evaluate=is_random_evaluate)
        valid_rec_loss += tf.reduce_mean(rec_loss)
        valid_kl_loss += kl_loss
        n_epochs += 1
        
    valid_rec_loss /= n_epochs
    valid_kl_loss /= n_epochs
    return float(valid_rec_loss), float( valid_kl_loss)





if __name__ == "__main__":
    print(tf.__version__)
    descr = "Tensorflow (Eager) implementation for beta-VAE model (Burgess et al. (2018): Understanding disentangling in beta-VAE). In all experiments Tensorflow (GPU) 1.13.1 and python 3. were used."
    epil  = "See: On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation [V. Prokhorov, E. Shareghi, Y. Li, M.T. Pilehvar, N. Collier (WNGT 2019)]"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--corpus', required=True, type=str,
                         help='path to a corpus')
    
    parser.add_argument('--C', required=True, type=int,
                         help='specify value of C (any integer number e.g. 10)')

    parser.add_argument('--checkpoint', required=True,
                         help='Directory where a trained model is stored')

    parser.add_argument('--is_load', required=True, type=int, default=0,
                         help='Train a new model or load existing one')

    parser.add_argument('--model', required=True, type=str,
                         help='name of a model: [LSTM-LSTM, GRU-GRU, LSTM-CONV]')
    

    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Size of data batches during training')

    parser.add_argument('--encoder_dims', type=int, default=512,
                        help='Number of embedding dimensions')
    
    parser.add_argument('--embed_dims', type=int, default=256,
                        help='Number of embedding dimensions')

    parser.add_argument('--z_dims', type=int, default=64,
                        help='Number of embedding dimensions')

    
  
    args = parser.parse_args()



   
    # Model Params#

    is_load = bool(args.is_load)
    epochs = args.epochs
    
    batch_size =  args.batchsize
    
    embedding_dim = args.embed_dims
    encoder_dim = args.encoder_dims
    z_dim = args.z_dims
    C_target = args.C  

    models_config = {'LSTM-LSTM':0, 'GRU-GRU':1, 'LSTM-CONV':2}
    model_config = models_config[args.model]

    # Load Data #
    
    training_data_path = '../../Data/Gen/'+args.corpus+'/train.unk.txt'
    vocab_path = '../../Data/Gen/'+args.corpus+'/vocab.txt'
    valid_data_path='../../Data/Gen/'+args.corpus+'/valid.unk.txt'
    test_data_path='../../Data/Gen/'+args.corpus+'/test.unk.txt'

    train_text_tensor, text_lang, max_length_text = load_dataset(training_data_path, vocab_file=vocab_path)
    test_text_tensor, _, _ = load_dataset(test_data_path,  text_lang=text_lang)
    valid_text_tensor, _, _ = load_dataset(valid_data_path,  text_lang=text_lang)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_text_tensor).shuffle(len(train_text_tensor))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_text_tensor)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)

    vocab_size = len(text_lang.word2idx)
    
    n_batch = math.ceil(len(train_text_tensor)/batch_size)
    

    
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00075)
    # Creating the Model #
    vae = Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim, model =model_config)
    
    # Save Model #
    checkpoint_dir = args.checkpoint
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                 vae=vae,
                                 optimizer_step=tf.train.get_or_create_global_step())
    # Select between loading the loading an existing Model and training a new model #
    if is_load == True:
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        print('load path', load_path)
        load = checkpoint.restore(load_path)
    else:
        n_epoch_to_kl_one = 2 
        n_annealing_steps = n_batch*n_epoch_to_kl_one
        train(epochs, len(train_text_tensor) ,vae, train_dataset, optimizer, text_lang, n_batch, checkpoint, valid_text_tensor, n_annealing_steps, C_target)
    

            
   
    with tf.device('/gpu:0'):
        test_rec_loss,  test_kl_loss = evaluate_nnl_and_rate_batch(test_dataset, vae, text_lang)
    print('BATCH: Test Rec Loss {:.4f};  Test KL-loss {:.4f}'.format(test_rec_loss, test_kl_loss))

   

