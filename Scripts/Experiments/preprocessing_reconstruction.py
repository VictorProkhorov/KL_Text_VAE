import sys
import numpy as np






def load_vocab(vocab_file):
    vocab = dict()
    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            idx = line[1]
            vocab[idx] = word
    return vocab







def map_idx_to_words(vocab, raw_sent_pairs_file, original_sent_file, rec_sent_file):
    id_word_map_f = lambda x: vocab[x]
    id_word_map_f_vec = np.vectorize(id_word_map_f)
    with open(raw_sent_pairs_file, 'r') as pair_f, open(original_sent_file, 'w') as org_sent_f, open(rec_sent_file, 'w') as rec_sent_f:
        for line in pair_f:
            line = line.strip().split(':')
            orig_sent = line[1].split(' ')
            rec_sent = line[0].split(' ')

            # mapping word_id to actual words
            orig_sent = id_word_map_f_vec(orig_sent)
            rec_sent = id_word_map_f_vec(rec_sent)
            org_sent_f.write(' '.join(orig_sent)+'\n')
            rec_sent_f.write(' '.join(rec_sent) + '\n')

    return None







def truncate_the_eos(sent_file, sent_final_file):
    with open(sent_file, 'r') as sent_f, open(sent_final_file, 'w') as sent_final_f:
        for line in sent_f:
            line = line.strip().split(' ')
            if '<EOS>' in line:
                eos_idx = line.index('<EOS>')
            else:
                eos_idx = len(line)
            sent_final_f.write(' '.join(line[:eos_idx]) + '\n')


    return None


def bucketing(org_file_name, rec_file_name, bucket_org_file_name, bucket_rec_file_name, b):
    
    with open(org_file_name, 'r') as org_f, open(rec_file_name, 'r') as rec_f, open(bucket_org_file_name, 'w') as b_org_f, open(bucket_rec_file_name, 'w') as b_rec_f:
        for org_sent, rec_sent in zip(org_f, rec_f):
            x = org_sent.strip().split(' ')
            if b(x):
                b_org_f.write(org_sent)
                b_rec_f.write(rec_sent)

    return None


    

def preprocess(corpus, C):
    vocab_file = './'+corpus+'/C_'+C+'/'+'vocab_map.txt'
    corpus_file = './'+corpus+'/C_'+C+'/'+'reconstruct_sentences.txt'
    
    original_sent_file = './'+corpus+'/C_'+C+'/'+'word_org_sent.txt'
    rec_sent_file = './'+corpus+'/C_'+C+'/'+'word_rec_sent.txt'
    
    original_sent_final_file = './'+corpus+'/C_'+C+'/'+'word_org_sent_final.txt'
    rec_sent_final_file = './'+corpus+'/C_'+C+'/'+'word_rec_sent_final.txt'
    
    vocab = load_vocab(vocab_file)
    map_idx_to_words(vocab, corpus_file, original_sent_file, rec_sent_file)
    truncate_the_eos(original_sent_file, original_sent_final_file)
    truncate_the_eos(rec_sent_file, rec_sent_final_file)
    
    bucket_org_file_name = './'+corpus+'/C_'+C+'/'+'bucket_org_1.txt'
    bucket_rec_file_name = './'+corpus+'/C_'+C+'/'+'bucket_rec_1.txt'
    b1 = lambda x: True if len(x) <= 10 else False
    bucketing(original_sent_final_file,  rec_sent_final_file, bucket_org_file_name, bucket_rec_file_name, b1)
    
    bucket_org_file_name = './'+corpus+'/C_'+C+'/'+'bucket_org_2.txt'
    bucket_rec_file_name = './'+corpus+'/C_'+C+'/'+'bucket_rec_2.txt'
    b2 = lambda x: True if len(x) > 10 and len(x) <=20 else False
    bucketing(original_sent_final_file, rec_sent_final_file, bucket_org_file_name, bucket_rec_file_name, b2)
    
    bucket_org_file_name = './'+corpus+'/C_'+C+'/'+'bucket_org_3.txt'
    bucket_rec_file_name = './'+corpus+'/C_'+C+'/'+'bucket_rec_3.txt'
    b3 = lambda x: True if len(x) > 20 and len(x) <= 30 else False
    bucketing(original_sent_final_file, rec_sent_final_file, bucket_org_file_name, bucket_rec_file_name, b3)



