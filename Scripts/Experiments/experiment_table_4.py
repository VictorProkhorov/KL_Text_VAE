import os
import sys
sys.path.insert(0, '../Model/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import OrderedDict
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pickle
import argparse
import model_wngt as model





agrmt_cases_dict = {1:['simple_agrmt'], 2:['sent_comp'], 3:['vp_coord'], 4:['long_vp_coord'], 5:['prep_anim', 'prep_inanim'], 6:['subj_rel'], 7:['obj_rel_across_anim', 'obj_rel_across_inanim'], 8:['obj_rel_no_comp_across_anim', 'obj_rel_no_comp_across_inanim'], 9:['obj_rel_within_anim', 'obj_rel_within_inanim'], 10:['obj_rel_no_comp_within_anim', 'obj_rel_no_comp_within_inanim']}   

reflexive_cases_dict = {3:['reflexives_across'], 1:['simple_reflexives'],2:['reflexive_sent_comp']}
negative_pol_cases_dict = {1: ['simple_npi_anim', 'simple_npi_inanim'], 2:['npi_across_anim', 'npi_across_inanim']}




syntactic_cases = OrderedDict()
syntactic_cases['agrmt_cases']= agrmt_cases_dict
syntactic_cases['reflexive_cases'] = reflexive_cases_dict
syntactic_cases['negative_pol_cases'] = negative_pol_cases_dict

def get_syntactic_zs(cases, vae, text_lang, mean_file_name, gpu):
    case_sentences = []
    for case in cases:
        with open('../../Data/SYN_EVAL/templates_emnlp/'+case+".pickle", "rb") as input_file:
            f = pickle.load(input_file)
            name_of_tests = list(sorted(list(f.keys())))
            print(name_of_tests)
            for test_name in name_of_tests:
                case_sentences = f[test_name]
                with open(mean_file_name+'.'+case+'.'+test_name, 'w' ) as f_m:

                    unk = '<unk>'
                    print('n_exemplars', len(case_sentences))
                    batch_size = 2
                    z_pos_mean = 0.0
                    z_neg_mean = 0.0
                    for idx, case_sentence_pair in enumerate(case_sentences):
                        case_sentence_pair = case_sentence_pair[:2]
                        x = tf.convert_to_tensor([[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in sentence.split(' ')] for sentence in case_sentence_pair ])
                        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
                        with tf.device(gpu):
                            x_enc = tf.concat([x, dec_input], axis=-1)
                            embeddings = vae.embeddings(x_enc)
                            outputs = vae.encoder.rnn(embeddings)
                            original_sentences_length = tf.count_nonzero(x_enc, 1, keepdims=False, dtype=tf.int32)
                            output =model.last_relevant(outputs, original_sentences_length)
           
                            mean = vae.encoder.mean(output)
                            z = mean # no smapling
           
                            z_pos_mean += z[0, :]
                            z_neg_mean += z[1, :]
		
       


                    n = len(case_sentences)
                    z_pos_mean /= n
                    z_neg_mean /= n
    
                    f_m.write(' '.join(list(map(str,z_pos_mean.numpy()))) + ':'+ ' '.join(list(map(str,z_neg_mean.numpy()))) +'\n' )

   


def get_accuracy_for_syntactic_case_mean(cases, vae, text_lang, gpu):
    case_sentences = []
    for case in cases:
        with open('../../Data/SYN_EVAL/templates_emnlp/'+case+".pickle", "rb") as input_file:
            f = pickle.load(input_file)
            name_of_tests = sorted(list(f.keys()))
            print(name_of_tests)

            for name_of_test in name_of_tests:
               # print(sentences)
                case_sentences += f[name_of_test]
    
    unk = '<unk>'
    print('n_exemplars', len(case_sentences))
    batch_size = 2
    accuracy_pos = 0.0
    accuracy_neg = 0.0
    gap_pos = 0.0
    gap_neg = 0.0
    for case_sentence_pair in case_sentences:
        case_sentence_pair = case_sentence_pair[:2]
     
        x = tf.convert_to_tensor([[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in sentence.split(' ')] for sentence in case_sentence_pair ])
        dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1) 
      
        x_enc = tf.concat([x, dec_input], axis=-1)
        with tf.device(gpu):   
            embeddings = vae.embeddings(x_enc)
            outputs = vae.encoder.rnn(embeddings)
            original_sentences_length = tf.count_nonzero(x_enc, 1, keepdims=False, dtype=tf.int32)
            output = model.last_relevant(outputs, original_sentences_length)
 
            mean = vae.encoder.mean(output)
            z = mean # no smapling
      
            z_pos = tf.concat([[z[0, :]], [z[0,:]] ], 0)
            z_neg = tf.concat([ [z[1, :]], [z[1,:]] ], 0)
        
        
            x_out = tf.concat([x, dec_input], axis=-1)
            x_inp = tf.concat([dec_input, x], axis=-1)

            embeddings = vae.embeddings(x_inp) 
            embeddings = tf.keras.layers.concatenate([embeddings, tf.keras.layers.RepeatVector(embeddings.shape[1])(z_pos)])
    
            loss_pos = vae.decoder(x_out, embeddings).numpy()
        
            embeddings = vae.embeddings(x_inp)
            embeddings = tf.keras.layers.concatenate([embeddings, tf.keras.layers.RepeatVector(embeddings.shape[1])(z_neg)])
            loss_neg = vae.decoder(x_out, embeddings).numpy()


        # correctly classify x+
        if loss_pos[0] < loss_pos[1]:  #and loss_neg[0] < loss_neg[1]:
            accuracy_pos += 1
       
        # miss classification of x+; p(x+|z-)>p(x-|z-)
        if loss_neg[0] < loss_neg[1]:
            accuracy_neg += 1
       
    
    return accuracy_pos/len(case_sentences), accuracy_neg/len(case_sentences)




def get_accuracy_for_syntactic_case_adversary_mean(cases, vae, text_lang, mean_file_name, gpu):
    accuracies_pos = []
    accuracies_neg = []
    gaps_pos = []
    gaps_neg = []
    for case in cases:
        with open('../../Data/SYN_EVAL/templates_emnlp/'+case+".pickle", "rb") as input_file:
            f = pickle.load(input_file)
            name_of_tests = list(sorted(list(f.keys())))

            print(name_of_tests)
            for name_of_test in name_of_tests: #f.values():
   
                case_sentences = f[name_of_test]
          
                z_mean_pos, z_mean_neg = read_file(mean_file_name+'.'+case+'.'+name_of_test)
           
                unk = '<unk>'
                print('n_exemplars', len(case_sentences))
                batch_size = 2
                accuracy_pos = 0.0
                accuracy_neg = 0.0
                for case_sentence_pair in case_sentences:
                    case_sentence_pair = case_sentence_pair[:2]

                    x = tf.convert_to_tensor([[text_lang.word2idx[word] if word in text_lang.word2idx else text_lang.word2idx[unk] for word in sentence.split(' ')] for sentence in case_sentence_pair ])
                    dec_input = tf.expand_dims([text_lang.word2idx['<EOS>']] * batch_size, 1)       
       
                    z_pos = tf.concat([[z_mean_pos], [z_mean_pos ] ], 0)
                    z_neg = tf.concat([ [z_mean_neg], [z_mean_neg] ], 0)
       
                    x_out = tf.concat([x, dec_input], axis=-1)
                    x_inp = tf.concat([dec_input, x], axis=-1)
                    with tf.device(gpu):
                        embeddings = vae.embeddings(x_inp) 
                        embeddings = tf.keras.layers.concatenate([embeddings, tf.keras.layers.RepeatVector(embeddings.shape[1])(z_pos)])
    
                        loss_pos = vae.decoder(x_out, embeddings).numpy()
        
                        embeddings = vae.embeddings(x_inp)
                        embeddings = tf.keras.layers.concatenate([embeddings, tf.keras.layers.RepeatVector(embeddings.shape[1])(z_neg)])
                        loss_neg = vae.decoder(x_out, embeddings).numpy()
                        
                    # correctly classify x+
                    if loss_pos[0] < loss_pos[1]:  #and loss_neg[0] < loss_neg[1]:
                        accuracy_pos += 1
                    # miss classification of x+; p(x+|z-)>p(x-|z-)
                    if loss_neg[0] < loss_neg[1]:
                        accuracy_neg += 1
                accuracy_pos /=len(case_sentences)
                accuracy_neg /=len(case_sentences)
               

                accuracies_pos.append(accuracy_pos)
                accuracies_neg.append(accuracy_neg)
    return sum(accuracies_pos)/len(accuracies_pos), sum(accuracies_neg)/len(accuracies_neg)




def read_file(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip().split(':')
            z_pos = np.array(list(map(float, line[0].split(' ')) ))
            z_neg = np.array(list(map(float, line[1].split(' ')) ))
    return tf.convert_to_tensor(z_pos, tf.float32), tf.convert_to_tensor(z_neg, tf.float32)




if __name__ == "__main__":
    print('You are using Tensorflow version:', tf.__version__)
    descr = "Tensorflow (Eager) implementation for experiments in Table 4."
    epil  = "See: On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation [V. Prokhorov, E. Shareghi, Y. Li, M.T. Pilehvar, N. Collier (WNGT 2019)]"
    parser = argparse.ArgumentParser(description=descr, epilog=epil)
    parser.add_argument('--C', required=True, type=int,
                         help='specify value of C: [3, 100]')

    
    args = parser.parse_args()

    reconstruct_sentences = False

    #print ('Corpus:', args.corpus)
    # Loading Data #
    training_data_path = '../../Data/Wiki/wiki_train.txt'
    vocab_path = '../../Data/Wiki/vocab.txt'
    test_data_path='../../Data/Wiki/wiki_test.txt'
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

  	
    print('Current value of C is:', args.C)
    name_of_pretrained_model ='BETA_VAE_C_'+str(args.C)+'_LSTM_WIKI'
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
    
    for synt_case_name in syntactic_cases:
        cases_dict = syntactic_cases[synt_case_name]
        mean_file_name = './Synt_Zs_Mean/_'+synt_case_name+'_'+str(args.C)
        for synt_exp in cases_dict:
            print('experiment:', synt_exp)
            cases = cases_dict[synt_exp]
            gpu = '/gpu:0'
            get_syntactic_zs(cases, vae, text_lang, mean_file_name, gpu)

    for synt_case_name in syntactic_cases:
       	cases_dict = syntactic_cases[synt_case_name]
        mean_file_name = './Synt_Zs_Mean/_'+synt_case_name+'_'+str(args.C)
        for synt_exp in cases_dict:
            print('experiment:', synt_exp)
            cases = cases_dict[synt_exp] 
            gpu = '/gpu:0'
            accuracy_pos, accuracy_neg = get_accuracy_for_syntactic_case_adversary_mean(cases, vae, text_lang,mean_file_name ,gpu)
            print('av accuracy_pos:', accuracy_pos, 'av accuracy neg:', accuracy_neg)


    for synt_case_name in syntactic_cases:
       	cases_dict = syntactic_cases[synt_case_name]
        for synt_exp in cases_dict:
            print('experiment:', synt_exp)
            cases = cases_dict[synt_exp]
            gpu = '/gpu:0'
            accuracy_pos, accuracy_neg = get_accuracy_for_syntactic_case_mean(cases, vae, text_lang, gpu)
            print('accuracy_pos:', accuracy_pos, 'accuracy neg:', accuracy_neg)


