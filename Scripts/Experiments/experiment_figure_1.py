import os
import sys
sys.path.insert(0, '../Model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import model_wngt as model




def calc_log_var(model, test_data_batch):
    z_prev = []
    for batch_data in test_data_batch:
        z  = model.sample_z(batch_data).numpy()
        if len(z_prev) > 0:
            z_prev = np.concatenate((z_prev, z), axis=0)
        else:
            z_prev = z
    mean = np.mean(z_prev, axis=0)
    return np.log(np.linalg.det(np.cov(z_prev.T))), np.dot(mean,mean)






if __name__ == "__main__":
    print('You are using Tensorflow version:', tf.__version__)
    descr = "Tensorflow (Eager) implementation for experiments in Figure 1."
    epil  = "See: On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation [V. Prokhorov, E. Shareghi, Y. Li, M.T. Pilehvar, N. Collier (WNGT 2019)]"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--corpus', required=True, type=str,
                         help='name of a corpus you want to test: [Yahoo, Yelp]')
    parser.add_argument('--model', required=True, type=str,
                         help='name of a model you want to test: [LSTM-LSTM, GRU-GRU, LSTM-CONV]')

    
    args = parser.parse_args()


    print ('Corpus:', args.corpus)
    # Loading Data #
    if args.corpus == 'Yahoo':
        training_data_path = '../../Data/Yahoo/yahoo.train.txt'
        vocab_path = '../../Data/Yahoo/vocab.txt'
        test_data_path='../../Data/Yahoo/yahoo.test.txt'
    
    else:
        training_data_path = '../../Data/Yelp/yelp.train_.txt'
        vocab_path = None
        test_data_path = '../../Data/Yelp/yelp.test_.txt'
   
    
    batch_size = 128

    train_text_tensor, text_lang, max_length_text = model.load_dataset(training_data_path, vocab_file=vocab_path)
    test_text_tensor, _, _ = model.load_dataset(test_data_path,  text_lang=text_lang)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_text_tensor).shuffle(len(train_text_tensor))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_text_tensor)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)

    vocab_size = len(text_lang.word2idx)

    # Model Param #
    embedding_dim = 256
    encoder_dim = 512
    z_dim = 64
    models_config = {'LSTM-LSTM':0, 'GRU-GRU':1, 'LSTM-CONV':2}
    model_config = models_config[args.model]

    Cs = [15,60,100]
    for C in Cs:
        # 'Distortion (D), Rate (R), LogDetCov, ||mu||^2', AU #
        print('Current value of C is:', C)
        name_of_pretrained_model =  'BETA_VAE_C_'+str(C)+'_'+args.model.split('-')[1]+'_512_'+ args.corpus
        print(name_of_pretrained_model)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00075)
        vae = model.Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim, model =model_config)
        
        checkpoint_dir = '../../Data/Trained_Models/'+ name_of_pretrained_model
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tfe.Checkpoint(optimizer=optimizer,vae=vae,optimizer_step=tf.train.get_or_create_global_step())
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        load = checkpoint.restore(load_path)
        with tf.device('/gpu:1'):
            test_D,  test_R = model.evaluate_nnl_and_rate_batch(test_dataset, vae, text_lang)
        print ('Distortion (D):', test_D)
        print ('Rate (R):', test_R)
        log_vars = []
        means = []
        for _ in range(5):
            with tf.device('/gpu:1'):
                log_var, mean = calc_log_var(vae, test_dataset)
                log_vars.append(log_var)
                means.append(mean)
        print('LogDetCov:', sum(log_vars)/len(log_vars))
       
    
