import os
import sys
sys.path.insert(0, '../Model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import model_wngt as model



def make_corpus_parallel(vae,batch_size, z_dim, max_length_seq, text_lang, ps=[], ks=[]):
        #print('Making Corpus')
        all_sampled_sent_greedy = []
        all_sampled_sent_top_k = dict()
        all_sampled_sent_nucleus = dict()

        z_noise = tf.keras.backend.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1)#[0]  
        
        
        def nucleus_threshold(values, p=0.9):
            cumsum_of_val = tf.math.cumsum(values, axis=1)
            zero_idx = cumsum_of_val[:, 0]
            rest_idx = cumsum_of_val[:, 1:]

            zero_idx_zeros = tf.zeros(zero_idx.shape)
            zero_idx_ones = tf.ones(zero_idx.shape)
            zero_idx_mask = tf.reshape(tf.where(zero_idx > p, zero_idx_ones, zero_idx_ones), [values.shape[0], 1])
     
            rest_idx_zeros = tf.zeros(rest_idx.shape)
            rest_idx_ones = tf.ones(rest_idx.shape)
            rest_idx_mask = tf.where(rest_idx > p, rest_idx_zeros, rest_idx_ones)
           
            mask = tf.concat([zero_idx_mask, rest_idx_mask], 1)
            masked_values = values*mask
            norm_const = tf.reduce_sum(masked_values, axis=1)
            norm_values = masked_values / tf.reshape(norm_const, [values.shape[0], 1])
          
            return norm_values

    
        
        def topk_renormalise(values):
            norm_const = tf.reduce_sum(values, axis=1)
            return values / tf.reshape(norm_const, [values.shape[0], 1])


        
        
        
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']]*batch_size, 1)  
        z = tf.convert_to_tensor(z_noise, dtype=tf.float32)   
        hidden = None
        result = []
        dec_sent = dec_input
        for _ in range(max_length_seq):
            embedding = vae.embeddings(dec_input)
            embedding = tf.keras.layers.concatenate([embedding, tf.expand_dims(z, axis=1)])
            _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
            hidden = [h,c]
            predictions = vae.decoder.out(h)
            _, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=1, sorted=True)
            
            dec_input = topi
            dec_sent = tf.concat([dec_sent,dec_input], 1)
        
        
        myfunc = lambda x: text_lang.idx2word[x]
        myfunc_vec = np.vectorize(myfunc)
        all_sampled_sent_greedy = myfunc_vec(dec_sent.numpy()[:,1:]).tolist()
        
        logit_f = lambda x: tf.log(x)-tf.log(1-x) 

        #### NUCLEUS ####  
        for p in ps:
            dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']]*batch_size, 1)
            hidden = None
            result = []
            dec_sent = dec_input
         
            for _ in range(max_length_seq):
                embedding = vae.embeddings(dec_input)
                embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                hidden = [h,c]
                predictions = vae.decoder.out(h)        
                vocab_prob_dist = tf.nn.softmax(predictions)
                topv, topi = tf.nn.top_k(vocab_prob_dist, k=vocab_prob_dist.shape[-1], sorted=True)
                nucleus_values = nucleus_threshold(topv, p=p)
          
                epsilon_ = tf.constant(tf.keras.backend.epsilon(), nucleus_values.dtype.base_dtype)
                nucleus_values = tf.clip_by_value(nucleus_values, epsilon_, 1 - epsilon_)
                nucleus_values = tf.log(nucleus_values)

                predicted_nuce_id = tf.multinomial(nucleus_values, num_samples=1, output_dtype=tf.int32)
                rows = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        
                idx_to_fetch = tf.concat([rows,predicted_nuce_id], 1)
                dec_input = tf.reshape(tf.gather_nd(topi, idx_to_fetch), [batch_size, 1])
                dec_sent = tf.concat([dec_sent, dec_input], 1)

            myfunc = lambda x: text_lang.idx2word[x]
            myfunc_vec = np.vectorize(myfunc)
            all_sampled_sent_nucleus[str(p)] = myfunc_vec(dec_sent.numpy()[:,1:]).tolist()
        
        
        #### TOP K ##### 
        for k in ks:
            dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']]*batch_size, 1)   
            hidden = None
            result = []
            dec_sent=dec_input
            for _ in range(max_length_seq):
                embedding = vae.embeddings(dec_input)
                embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                hidden = [h,c]
                predictions = vae.decoder.out(h)
               
                topv, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=k, sorted=True)
                topk_values = topk_renormalise(topv)
          
                
                epsilon_ = tf.constant(tf.keras.backend.epsilon(), topk_values.dtype.base_dtype)
                topk_values = tf.clip_by_value(topk_values, epsilon_, 1 - epsilon_)
                topk_values = tf.log(topk_values)

                predicted_topk_id = tf.multinomial(topk_values, num_samples=1, output_dtype=tf.int32)
                rows = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
                idx_to_fetch = tf.concat([rows,predicted_topk_id], 1)
                dec_input = tf.reshape(tf.gather_nd(topi, idx_to_fetch), [batch_size, 1])

                dec_sent = tf.concat([dec_sent, dec_input], 1)

            myfunc = lambda x: text_lang.idx2word[x]
            myfunc_vec = np.vectorize(myfunc)
            all_sampled_sent_top_k[str(k)] = myfunc_vec(dec_sent.numpy()[:,1:]).tolist()
            
        return all_sampled_sent_greedy, all_sampled_sent_nucleus, all_sampled_sent_top_k
    
    




def sample_sentences_from_prior(vae, batch_size, z_dim,  max_length_text, text_lang):
       
        n_copies_of_corpus = 3
        C_value = C_target
        file_directory = './Artificial_Corpora/CBT/'+'C_'+str(C_value)+'/'
        p_vals = [0.5, 0.7, 0.9]
        k_vals = [5, 10, 15]
        number_of_sentences =  len(test_data)
        batch_size = 2048

        for i in range(n_copies_of_corpus):
            corpus_copy = i + 1
            file_name_greedy = file_directory  + '/copy_'+str(corpus_copy) + '.greedy'
        
            artificial_corpus_greedy = open(file_name_greedy, 'w')   
        
            opened_files = dict()
            for k_p in zip(k_vals,p_vals):
                k, p = k_p
                file_name_top_k = file_directory    + '/copy_'+str(corpus_copy) + '._k_'+str(k) 
                file_name_nucleus = file_directory  + '/copy_'+str(corpus_copy) + '._p_'+str(p)
        
                opened_files[str(k)] = open(file_name_top_k, 'w')
                opened_files[str(p)] = open(file_name_nucleus, 'w')
        
        

        
            for _ in range(int(math.ceil(number_of_sentences/batch_size))):
                with tf.device('/gpu:0'):
                    interpolated_greedy, interpolated_nucleus, interpolated_topk = make_corpus_parallel(vae, batch_size, z_dim, max_length_text, text_lang, ps=p_vals, ks=k_vals)
                    
                for sent_id in range(batch_size): 
                
                    sent_greedy = interpolated_greedy[sent_id]#.tolist()
                   # artificial_corpus_greedy.write(' '.join(sent_greedy)+ '\n')

                    greedy_sent_len = len(sent_greedy)
                    if '<EOS>' in sent_greedy:
                        greedy_sent_len = sent_greedy.index('<EOS>')
                    artificial_corpus_greedy.write(' '.join(sent_greedy[:greedy_sent_len])+ '\n')
            
                for k_p in zip(k_vals, p_vals):
                    k, p = k_p
                    for sent_id in range(batch_size):
                    
                        sent_top_k = interpolated_topk[str(k)][sent_id]#.tolist()
                        #opened_files[str(k)].write(' '.join(sent_top_k)+'\n')

                        topk_sent_len = len(sent_top_k)
                        if '<EOS>' in sent_top_k:
                            topk_sent_len = sent_top_k.index('<EOS>')
                        opened_files[str(k)].write(' '.join(sent_top_k[:topk_sent_len])+'\n')
                    
                        sent_nucleus = interpolated_nucleus[str(p)][sent_id]#.tolist()
                        #opened_files[str(p)].write(' '.join(sent_nucleus) + '\n')

                        nucleus_sent_len = len(sent_nucleus)
                        if '<EOS>' in sent_nucleus:
                            nucleus_sent_len = sent_nucleus.index('<EOS>')
                        opened_files[str(p)].write(' '.join(sent_nucleus[:nucleus_sent_len]) + '\n')
        
        
            artificial_corpus_greedy.close()

            for k_p in zip(k_vals, p_vals):
                k, p = k_p
                opened_files[str(k)].close()
                opened_files[str(p)].close()














if __name__ == "__main__":
    print('You are using Tensorflow version:', tf.__version__)
    descr = "Tensorflow (Eager) implementation for experiments in Table 2."
    epil  = "See: On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation [V. Prokhorov, E. Shareghi, Y. Li, M.T. Pilehvar, N. Collier (WNGT 2019)]"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--corpus', required=True, type=str,
                         help='name of a corpus you want to test: [CBT, WebText, Wiki]')

    
    args = parser.parse_args()

    reconstruct_sentences = False

    print ('Corpus:', args.corpus)
    # Loading Data #
    training_data_path = '../../Data/Gen/'+args.corpus+'/train.unk.txt'
    vocab_path = '../../Data/Gen/'+args.corpus+'/vocab.txt'
    test_data_path='../../Data/Gen/'+args.corpus+'/test.unk.txt'
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

    Cs = [3,15,100]
    for C in Cs:
        # 'Distortion (D), Rate (R), LogDetCov, ||mu||^2', AU #
        print('Current value of C is:', C)
        name_of_pretrained_model ='BETA_VAE_C_'+str(C)+'_LSTM_512_GEN_'+args.corpus
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00075)
        vae = model.Sentence_VAE(embedding_dim, vocab_size, encoder_dim, z_dim, model =0)
        
        checkpoint_dir = '../../Data/Trained_Models/'+ name_of_pretrained_model
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tfe.Checkpoint(optimizer=optimizer,vae=vae,optimizer_step=tf.train.get_or_create_global_step())
        load_path = tf.train.latest_checkpoint(checkpoint_dir)
        load = checkpoint.restore(load_path)
        with tf.device('/gpu:0'):
            test_D,  test_R = model.evaluate_nnl_and_rate_batch(test_dataset, vae, text_lang)
        # print rate and distortion for sanity check (essentially want to double check if the model has been loaded)
        print ('Distortion (D):', test_D)
        print ('Rate (R):', test_R)


        sample_sentences_from_prior(vae, batch_size, z_dim,  max_length_text, text_lang)


