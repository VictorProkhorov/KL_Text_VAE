import os
import sys
sys.path.insert(0, '../Model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import model_wngt as model
import preprocessing_reconstruction as preprocess
import rouge
import bleu



def calc_bleu_scores(gold_sentences_file_name, rec_sentences_file_name, n=2):
    original_s = []
    restored_s = []
    with open(gold_sentences_file_name, 'r') as g_f, open(rec_sentences_file_name, 'r') as rec_f:
        for gold_rec_sent in zip(g_f, rec_f):
            gold_sent, rec_sent = gold_rec_sent
            gold_sent = gold_sent.strip().split(' ')
            rec_sent = rec_sent.strip().split(' ')
            
            original_s.append([gold_sent])
            restored_s.append(rec_sent)
    bleu_score = bleu.compute_bleu(original_s, restored_s, max_order=n, smooth=False)
    #b_score = nltk.translate.bleu_score.corpus_bleu(original_s, restored_s)
    #print(b_score)
    print('BLEU:', bleu_score[0]*100)



def calc_rouge_scores(gold_sentences_file_name, rec_sentences_file_name, n=2):
   
    f1s = []
    
    with open(gold_sentences_file_name, 'r') as g_f, open(rec_sentences_file_name, 'r') as rec_f:
        for gold_rec_sent in zip(g_f, rec_f):
            gold_sent, rec_sent = gold_rec_sent
            
            gold_sent = gold_sent.strip()
            rec_sent = rec_sent.strip()
            f1, precision, recal = rouge.rouge_n([rec_sent],[gold_sent] , n=n)
            f1s.append(f1)
    print('ROUGE:', (sum(f1s)/len(f1s))*100)


def calc_au(vae, test_data_batch, delta=0.01):
    """ 
        Compute the number of active units. The original code was taken from https://github.com/jxhe/vae-lagging-encoder/blob/master/text.py

    """

    cnt = 0.0
    for batch_data in test_data_batch:
        
        embeddings = vae.embeddings(batch_data)
        outputs = vae.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(batch_data, 1, keepdims=False, dtype=tf.int32)
        output = model.last_relevant(outputs, original_sentences_length)
        mean = vae.encoder.mean(output)
        
        if cnt == 0:
            means_sum = tf.math.reduce_sum(mean, keepdims=True, axis=0)
            #print('s', means_sum.shape)
        else:
            means_sum = means_sum + tf.math.reduce_sum(mean, keepdims=True, axis=0)
            
        cnt += mean.shape[0]
    # (1, nz)
    mean_mean = means_sum / tf.to_float(cnt)

    cnt = 0.0
    for batch_data in test_data_batch:
        embeddings = vae.embeddings(batch_data)
        outputs = vae.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(batch_data, 1, keepdims=False, dtype=tf.int32)
        output = model.last_relevant(outputs, original_sentences_length)
        mean = vae.encoder.mean(output)

        
        if cnt == 0:
            var_sum = tf.math.reduce_sum( ((mean - mean_mean) ** 2), axis=0)  #sum(dim=0)
        else:
            var_sum = var_sum +  tf.math.reduce_sum( ((mean - mean_mean) ** 2), axis=0) #((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.shape[0]

    # (nz)
    au_var = var_sum / (tf.to_float(cnt) - 1)
    above_threshold = tf.cast((au_var >= delta), dtype=tf.int32) 
    return int(tf.math.reduce_sum(above_threshold)), au_var

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




def restore_sentences_batch(vae, x, text_lang, is_sample_z=False):
        
        batch_size = x.shape[0]
        max_seq_len = x.shape[1]
        embeddings = vae.embeddings(x)
        outputs = vae.encoder.rnn(embeddings)
        original_sentences_length = tf.count_nonzero(x, 1, keepdims=False, dtype=tf.int32)
        output = last_relevant(outputs, original_sentences_length)
        mean = vae.encoder.mean(output)
        log_var = vae.encoder.log_var(output)
        
        if is_sample_z:
            z = vae.encoder._sampling([mean, log_var]) 
        else:
            z = mean
 

 
        ### DECODING ###
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
        hidden = None
        dec_sent = dec_input
        for _ in range(0, max_seq_len):         
            embedding = vae.embeddings(dec_input)
            embedding = tf.keras.layers.concatenate([embedding,tf.expand_dims(z, axis=1)])
            _, h, c = vae.decoder.rnn(embedding, initial_state = hidden)
            hidden = [h, c]
            predictions = vae.decoder.out(h)
            probs = tf.nn.softmax(predictions)
            _, topi = tf.nn.top_k(probs, k=1, sorted=True)
            
            dec_input = topi
            dec_sent = tf.concat([dec_sent,dec_input], 1)
 
        myfunc = lambda x: str(x)
        myfunc_vec = np.vectorize(myfunc)
        result = myfunc_vec(dec_sent.numpy()[:,1:]).tolist()
        x = myfunc_vec(x.numpy()).tolist()

        return result, x
    


def sentence_reconstruct(vae, C, test_text_tensor):
    
    sentences_file_name = '../../Data/Reconstruction/WebText/C_'+str(C)+'/'+'reconstruct_sentences.txt'
    vocab_mapping_file = '../../Data/Reconstruction/WebText/C_'+str(C)+'/'+'vocab_map.txt'
        
    with open(vocab_mapping_file, 'w') as vocab_f:
        for word in text_lang.word2idx:
            vocab_f.write(word + ' ' + str(text_lang.word2idx[word])+'\n')



    test_dataset = tf.data.Dataset.from_tensor_slices(test_text_tensor)
    test_batch_size = 512
    test_dataset = test_dataset.batch(test_batch_size, drop_remainder=False)

    rec_sent_file = open(sentences_file_name, 'w')
    for (batch, sentences) in enumerate(test_dataset):
        with tf.device('/gpu:1'):
            dec_sentences, org_sentences =  vrestore_sentences_batch(vae, sentences, text_lang, is_sample_z=False)    
            for i in range(sentences.shape[0]):
                rec_sent_file.write(' '.join(dec_sentences[i]) + ':' + ' '.join(org_sentences[i])+ '\n')
    rec_sent_file.close()



if __name__ == "__main__":
    print('You are using Tensorflow version:', tf.__version__)
    descr = "Tensorflow (Eager) implementation for experiments in Table 1."
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
        print('||mu||^2:', sum(means)/len(means))

        with tf.device('/gpu:1'):
            au, _ = calc_au(vae, test_dataset, delta=0.01)
        print('AU:', au)

        # Reconstruction #

        # (Optional) the reconstructed sentences are provided. However if you wishe to generate your use the function below:
        if reconstruct_sentences:
            sentence_reconstruct(vae, C, test_text_tensor)
            preprocess.preprocess(args.corpus, C)

        print('bucket 1')
        original_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_org_1.txt'
        rec_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_rec_1.txt'
        print('2-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file)
        print('4-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file, n=4)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file, n=4)
        

        print('bucket 2')
        original_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_org_2.txt'
        rec_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_rec_2.txt'
        print('2-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file)
        
        print('4-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file, n=4)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file, n=4)
        

        print('bucket 3')
        original_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_org_3.txt'
        rec_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'bucket_rec_3.txt'
        print('2-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file)
        
        print('4-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file, n=4)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file, n=4)

        print('All sentenes')
        original_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'word_org_sent_final.txt'
        rec_sent_final_file = '../../Data/Reconstruction/'+args.corpus+'/C_'+str(C)+'/'+'word_rec_sent_final.txt'
        print('2-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file)
        
        print('4-gram')
        calc_bleu_scores(original_sent_final_file, rec_sent_final_file, n=4)
        calc_rouge_scores(original_sent_final_file, rec_sent_final_file, n=4)
        



