import os
import sys
sys.path.insert(0, '../Model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import model_wngt as model



def interpolate_g_n_k_all_models(vae_c_5, vae_c_15, vae_c_50 ,vae_c_100, steps, z_dim, max_length_seq, text_lang,p=0.9, k=10):
        'An example of interpolation for more than one model'
        vaes = {'vae_c_5':vae_c_5, 'vae_c_15':vae_c_15, 'vae_c_50':vae_c_50, 'vae_c_100':vae_c_100}
        z_noise_start = tf.keras.backend.random_normal(shape=(1, z_dim), mean=0., stddev=1)[0]  
        z_noise_end = tf.keras.backend.random_normal(shape=(1, z_dim), mean=0., stddev=1)[0]
        sample = np.linspace(0, 1, steps, endpoint = True)
        interpolation = []
        def nucleus_threshold(values, p=0.9):
            n_elements = values.shape[-1]
            #print('n_elem', n_elements)
            for i in range(1, n_elements+1):
                if tf.math.reduce_sum(values[0][:i]) < p:
                    continue
                else:
                    return i
        def nucleus_renormalise(values, index):
            norm_constant = tf.math.reduce_sum(values[0][:index])
            #print(norm_constant)
            #print(values[0]
            return values[0][:index]/norm_constant
        
        def topk_renormalise(values):
            norm_constant = tf.math.reduce_sum(values[0])
            #print(norm_constant)
            #print(values[0]
            return values[0]/norm_constant
        

        
        for s in sample:
            interpolation.append(z_noise_start *(1-s)+ s * z_noise_end)
        vae_sent_dict = {'vae_c_5':[], 'vae_c_15':[],'vae_c_50':[], 'vae_c_100':[]} 
        for vae_name in vaes:
            vae = vaes[vae_name]

            all_sampled_sent_greedy = []
            all_sampled_sent_top_k = []
            all_sampled_sent_nucleus = []

            for z_noise in interpolation:
                dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
                z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
                hidden = None
                result = []
                for _ in range(max_length_seq):
                    embedding = vae.embeddings(dec_input)
                    embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                    _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                    hidden = [h,c]
                    predictions = vae.decoder.out(h)
                    _, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=1, sorted=True)
                    predicted_id = topi.numpy()[0][0]
                
                    if text_lang.idx2word[predicted_id] == '<EOS>':
                        result.append(text_lang.idx2word[predicted_id])
                        all_sampled_sent_greedy.append(result)
                        break
                    # the predicted ID is fed back into the model
                    dec_input = tf.expand_dims([predicted_id], 0)
                    result.append(text_lang.idx2word[predicted_id])

                if '<EOS>' not in result:
                    all_sampled_sent_greedy.append(result)
        
         

            vae_sent_dict[vae_name].append(all_sampled_sent_greedy) 
            for z_noise in interpolation:
                dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
                z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
                hidden = None
                result = []
                for _ in range(max_length_seq):
                    embedding = vae.embeddings(dec_input)
                    embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                    _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                    hidden = [h,c]
                    predictions = vae.decoder.out(h)        
                    vocab_prob_dist = tf.nn.softmax(predictions)
                    topv, topi = tf.nn.top_k(vocab_prob_dist, k=vocab_prob_dist.shape[-1], sorted=True)
                    nuce_indx = nucleus_threshold(topv, p=p)
                
                    nucleus_values = tf.convert_to_tensor([nucleus_renormalise(topv, nuce_indx)])
                
                    epsilon_ = tf.constant(tf.keras.backend.epsilon(), nucleus_values.dtype.base_dtype)
                    nucleus_values = tf.clip_by_value(nucleus_values, epsilon_, 1 - epsilon_)
                    

                    nucleus_values = tf.log(nucleus_values)
                               


                # nucleus_values = tf.map_fn(logit_f, nucleus_values)

                    predicted_nuce_id = tf.multinomial(nucleus_values, num_samples=1).numpy()[0][0]
                    predicted_id = topi[0][predicted_nuce_id].numpy()
                
                    if text_lang.idx2word[predicted_id] == '<EOS>':
                        result.append(text_lang.idx2word[predicted_id])
                        all_sampled_sent_nucleus.append(result)
                        break
                    # the predicted ID is fed back into the model
                    dec_input = tf.expand_dims([predicted_id], 0)
                    result.append(text_lang.idx2word[predicted_id])

                if '<EOS>' not in result:
                    all_sampled_sent_nucleus.append(result)

            vae_sent_dict[vae_name].append(all_sampled_sent_nucleus) 
            for z_noise in interpolation:
                dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
                z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
                hidden = None
                result = []
                for _ in range(max_length_seq):
                    embedding = vae.embeddings(dec_input)
                    embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                    _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                    hidden = [h,c]
                    predictions = vae.decoder.out(h)
               
                    topv, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=k, sorted=True)
                    topk_values = tf.convert_to_tensor([topk_renormalise(topv)]) 
                    epsilon_ = tf.constant(tf.keras.backend.epsilon(), topk_values.dtype.base_dtype)
                    topk_values = tf.clip_by_value(topk_values, epsilon_, 1 - epsilon_)
                    topk_values = tf.log(topk_values)
                
                    predicted_topk_id = tf.multinomial(topk_values, num_samples=1).numpy()[0][0]
                    predicted_id = topi[0][predicted_topk_id].numpy()
            
                    if text_lang.idx2word[predicted_id] == '<EOS>':
                        result.append(text_lang.idx2word[predicted_id])
                        all_sampled_sent_top_k.append(result)
                        break
                # the predicted ID is fed back into the model
                    dec_input = tf.expand_dims([predicted_id], 0)
                    result.append(text_lang.idx2word[predicted_id])

                if '<EOS>' not in result:
                    all_sampled_sent_top_k.append(result)
            vae_sent_dict[vae_name].append(all_sampled_sent_top_k) 
        
        return vae_sent_dict




def interpolate_g_n_k(vae, steps, z_dim, max_length_seq, text_lang,p=0.9, k=10):
        print('INTERPOLATING')
        all_sampled_sent_greedy = []
        all_sampled_sent_top_k = []
        all_sampled_sent_nucleus = []

        z_noise_start = tf.keras.backend.random_normal(shape=(1, z_dim), mean=0., stddev=1)[0]  
        z_noise_end = tf.keras.backend.random_normal(shape=(1, z_dim), mean=0., stddev=1)[0]
        sample = np.linspace(0, 1, steps, endpoint = True)
        interpolation = []
       
        def nucleus_threshold(values, p=0.9):
            n_elements = values.shape[-1]
            for i in range(1, n_elements+1):
                if tf.math.reduce_sum(values[0][:i]) < p:
                    continue
                else:
                    return i
        
        def nucleus_renormalise(values, index):
            norm_constant = tf.math.reduce_sum(values[0][:index])
            return values[0][:index]/norm_constant
        
        def topk_renormalise(values):
            norm_constant = tf.math.reduce_sum(values[0])
            return values[0]/norm_constant
        

        
        for s in sample:
            interpolation.append(z_noise_start *(1-s)+ s * z_noise_end)
        
        for z_noise in interpolation:
            dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
            z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
            hidden = None
            result = []
            for _ in range(max_length_seq):
                embedding = vae.embeddings(dec_input)
                embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                hidden = [h,c]
                predictions = vae.decoder.out(h)
                _, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=1, sorted=True)
                predicted_id = topi.numpy()[0][0]
                
                if text_lang.idx2word[predicted_id] == '<EOS>':
                    result.append(text_lang.idx2word[predicted_id])
                    all_sampled_sent_greedy.append(result)
                    break
                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)
                result.append(text_lang.idx2word[predicted_id])

            if '<EOS>' not in result:
                all_sampled_sent_greedy.append(result)
        

        
        for z_noise in interpolation:
            dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
            z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
            hidden = None
            result = []
            for _ in range(max_length_seq):
                embedding = vae.embeddings(dec_input)
                embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                hidden = [h,c]
                predictions = vae.decoder.out(h)        
                vocab_prob_dist = tf.nn.softmax(predictions)
                topv, topi = tf.nn.top_k(vocab_prob_dist, k=vocab_prob_dist.shape[-1], sorted=True)
                nuce_indx = nucleus_threshold(topv, p=p)
                
                nucleus_values = tf.convert_to_tensor([nucleus_renormalise(topv, nuce_indx)])
                
                epsilon_ = tf.constant(tf.keras.backend.epsilon(), nucleus_values.dtype.base_dtype)
                nucleus_values = tf.clip_by_value(nucleus_values, epsilon_, 1 - epsilon_)
                

                nucleus_values = tf.log(nucleus_values)
                 


               

                predicted_nuce_id = tf.multinomial(nucleus_values, num_samples=1).numpy()[0][0]
                predicted_id = topi[0][predicted_nuce_id].numpy()
                
                if text_lang.idx2word[predicted_id] == '<EOS>':
                    result.append(text_lang.idx2word[predicted_id])
                    all_sampled_sent_nucleus.append(result)
                    break
                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)
                result.append(text_lang.idx2word[predicted_id])

            if '<EOS>' not in result:
                all_sampled_sent_nucleus.append(result)

        
        for z_noise in interpolation:
            dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']], 0)
            z = tf.convert_to_tensor([z_noise], dtype=tf.float32)
           
            hidden = None
            result = []
            for _ in range(max_length_seq):
                embedding = vae.embeddings(dec_input)
                embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
                _, h,c = vae.decoder.rnn(embedding, initial_state=hidden)
                hidden = [h,c]
                predictions = vae.decoder.out(h)
               
                topv, topi = tf.nn.top_k(tf.nn.softmax(predictions), k=k, sorted=True)
                topk_values = tf.convert_to_tensor([topk_renormalise(topv)]) 
                epsilon_ = tf.constant(tf.keras.backend.epsilon(), topk_values.dtype.base_dtype)
                topk_values = tf.clip_by_value(topk_values, epsilon_, 1 - epsilon_)
               
                topk_values = tf.log(topk_values)
 
                predicted_topk_id = tf.multinomial(topk_values, num_samples=1).numpy()[0][0]
                predicted_id = topi[0][predicted_topk_id].numpy()
            
                if text_lang.idx2word[predicted_id] == '<EOS>':
                    result.append(text_lang.idx2word[predicted_id])
                    all_sampled_sent_top_k.append(result)
                    break
                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)
                result.append(text_lang.idx2word[predicted_id])

            if '<EOS>' not in result:
                all_sampled_sent_top_k.append(result)

        
        return all_sampled_sent_greedy, all_sampled_sent_nucleus, all_sampled_sent_top_k







def interpolate(vae, max_length_text, z_dim, text_lang):
	with tf.device('/gpu:1'):
		interpolated_greedy, interpolated_nucleus, interpolated_topk = interpolate_g_n_k(vae, 8, z_dim, max_length_text, text_lang, p=0.9, k=15)
	print('Greedy')
	for idx, sentence in enumerate(interpolated_greedy):
		line = str(idx+1)+': ' + ' '.join(sentence)
		print(line)
	print('\n')
	print('Nucleus')
	for idx, sentence in enumerate(interpolated_nucleus):
		line = str(idx+1)+': ' + ' '.join(sentence) 
		print(line)
	print('\n')
	print('Top K')
	for idx, sentence in enumerate(interpolated_topk):
		line = str(idx+1)+': ' + ' '.join(sentence)
		print(line)
	print('\n')
	print('----------------' + '\n') 




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
        with tf.device('/gpu:1'):
            test_D,  test_R = model.evaluate_nnl_and_rate_batch(test_dataset, vae, text_lang)
        # print rate and distortion for sanity check (essentially want to double check if the model has been loaded)
        print ('Distortion (D):', test_D)
        print ('Rate (R):', test_R)

        interpolate(vae,max_length_text, z_dim, text_lang )
        # uncomment and adjust accordingly if you wish to interpolate for more than one model simultaneously
  		#interpolate_g_n_k_all_models(vae_c_5, vae_c_15, vae_c_50 ,vae_c_100, steps, z_dim, max_length_seq, text_lang,p=0.9, k=10)



