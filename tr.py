from keras.utils.np_utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional,TimeDistributed
import pickle
from keras.preprocessing.text import Tokenizer
import re
import pre
import numpy as np
from keras.models import load_model
import unicodedata
import os
# fix random seed for reproducibility
np.random.seed(1)
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import operator
from sklearn.metrics import f1_score
from keras.models import load_model

def mean_pred(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

#seperating the pivots to unigrams and bigrams features
def get_unigram_bigram(my_list):
    unigram = []
    bigram = []
    for feature in my_list:
        if (len(feature.split()) == 1):
            unigram.append(feature)
        if (len(feature.split()) == 2):
            word=feature.split()
            temp=word[0]+'_'+word[1]
            bigram.append(temp)
    return unigram,bigram

#initializng pivot to index dictionary
def fill_pivot_dict(names,hidden,meta):
    i=1
    pivot2int = dict()
    pivot_num = len(names)
    pos_index = 2
    neg_index = 1
    index = 2
    for i in range(pivot_num):
        if names[i] in hidden:
            sen, MI, src, dest = meta[names[i]]
            if(sen == "neg"):
                pivot2int[names[i]] = neg_index
            if(sen == "pos"):
                pivot2int[names[i]] = pos_index
    for i in range(pivot_num):
        if not (names[i] in hidden):
            index += 1
            pivot2int[names[i]] = index

    return  pivot2int,index

#fix some encoding issues
def getClear_full(sentence):
    r = re.findall(r'\b\w+\b', sentence.lower())
    r = " ".join(r)
    r = (r.decode('unicode_escape').encode('ascii', 'ignore'))
    return r


def getClearList(sentences):
    clearSentences = []
    for s in sentences:
        clearSentences.append(getClear_full(s))
    return  clearSentences


#prepare the PBLM labels, for each word the corresponding label should be the pivot name if the next word is a pivot and
#a NONE symbol otherwise, if the next next word is a unigram pivot and the next couple of words are bigram pivot the
#label will be the bigram pivot, as mentioned in the article
def makeLSTMinput(raw_sentences,direction,names ,Pdict, id2word,pivot_num):
    lstm_labels = []
    noPivotIndex = pivot_num + 1
    index = 0
    for sen in raw_sentences:
        #in this code we use only the left to right PBLM, one can add a right to left PBLM if he wishes
        if(direction == 'L2R'):
            index=index+1
            words = [id2word[idx] for idx in sen]
            sen_len = len(words)
            if(sen_len == 0 ):
                print index
            new_labels = []
            for i in range(sen_len-2):
                bigram_found = False
                unigram_found = False
                if((words[i+1] + " " + words[i+2]) in Pdict):
                    new_labels.append(Pdict[words[i+1] + " " + words[i+2]])
                    bigram_found = True
                if((not(bigram_found)) and (words[i+1] in Pdict) and (not((words[i] + " " + words[i+1]) in Pdict))):
                    new_labels.append(Pdict[words[i+1]])
                    unigram_found = True
                if((not(bigram_found)) and (not(unigram_found))):
                    new_labels.append(noPivotIndex)
            unigram_found = False
            if(sen_len>1):
                if ((words[sen_len-1] in Pdict) and (not((words[sen_len-2] + " " + words[sen_len-1]) in Pdict))):
                    new_labels.append(Pdict[words[sen_len-1]])
                    unigram_found = True
            if(not(unigram_found)):
                new_labels.append(noPivotIndex)
            new_labels.append(noPivotIndex)
            str_lbl = new_labels
            if(len(new_labels)-sen_len != 0 ):
                print "the diff is ",len(new_labels)-sen_len
        lstm_labels.append(str_lbl)
    return lstm_labels

#data generator for the PBLM training
def generator(x, batch_size, names, Pdict,tok,max_review_len,pivot_num):
    index = np.arange(len(x))
    start = 0
    word2id = tok.word_index
    id2word = {idx: word for (word, idx) in word2id.items()}
    noPivotIndex = pivot_num + 1
    while True:
        if start == 0 :
            #shuffels the data every epoch
            np.random.shuffle(index)
        batch = index[start:start + batch_size]
        x_batch = [x[ind] for ind in batch]
        #prepare the lstm labels
        y_batch = makeLSTMinput(x_batch,"L2R",names, Pdict,id2word,pivot_num)
        x_batch = sequence.pad_sequences(x_batch, maxlen=max_review_len)
        y_batch = sequence.pad_sequences(y_batch, maxlen=max_review_len, value = noPivotIndex)
        y_batch = np.array([to_categorical(sent_label, pivot_num + 2) for sent_label in y_batch])
        yield x_batch, y_batch
        start += batch_size
        if start >= len(x):
            start = 0

#the val generator is the same as the regular generator, Keras can't handle  the same generator passed for both
#the training data and the validation data therefor I copied the generator for different instant
def generator_val(x, batch_size, names, Pdict,tok,max_review_len,pivot_num):
    index = np.arange(len(x))
    start = 0
    word2id = tok.word_index
    id2word = {idx: word for (word, idx) in word2id.items()}
    noPivotIndex = pivot_num + 1
    while True:
        if start == 0 :
            np.random.shuffle(index)
        batch = index[start:start + batch_size]
        x_batch = [x[ind] for ind in batch]
        y_batch = makeLSTMinput(x_batch,"L2R",names, Pdict,id2word,pivot_num)
        x_batch = sequence.pad_sequences(x_batch, maxlen=max_review_len)
        y_batch = sequence.pad_sequences(y_batch, maxlen=max_review_len,value = noPivotIndex)
        y_batch = np.array([to_categorical(sent_label, pivot_num + 2) for sent_label in y_batch])
        yield x_batch, y_batch
        start += batch_size
        if start >= len(x):
            start = 0

def GetNewModel(model_str, model, names ,meta):
    neg_index = 1
    pos_index = 2
    pivot_num = len(names)
    former_model = load_model(model_str)
    for i in range(len(former_model.layers) - 1):
        model.layers[i].set_weights(former_model.layers[i].get_weights())
    old_W, old_b = former_model.layers[len(former_model.layers) - 1].get_weights()
    new_W, new_b = model.layers[len(model.layers) - 1].get_weights()
    former_class_num = old_W.shape[1]
    new_class_num = new_W.shape[1]
    for i in range(former_class_num-1):
        new_W[:,i] = old_W[:,i]
        new_b[i] = old_b[i]
    new_W[:, -1] = old_W[:, -1]
    new_b[-1] = old_b[-1]
    shift = 3
    for i in range(former_class_num-1,new_class_num-1):
        sen, MI, src, dest = meta[names[i-shift]]
        print names[i-shift]," ",i

        if (sen == "neg"):
            new_W[:, i] = old_W[:,neg_index]
            new_b[i] = old_b[neg_index]
        if (sen == "pos"):
            new_W[:, i] = old_W[:,pos_index]
            new_b[i] = old_b[pos_index]
    model.layers[len(model.layers) - 1].set_weights([new_W,new_b])

    '''
    print "\n\nformer w "
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in old_W]))
    print "\n\nformer b ", old_b

    print "\n\nnew w "
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in new_W]))
    print "\n\nnew b ", new_b

    #print "\n\nformer w ", old_W
    #print "\n\nformer w ", old_b
    #print "\n\nnew w ", new_W
    #print "\n\nformer w ", new_W
    '''
    return model




def train_PBLM(src,dest,pivot_num,pivot_min_st,word_vector_size,topWords,max_review_len,hidden_units_num,names,
               current_iter, iter_num,criteria):

    split_dir =src+"_to_"+dest
    # gets all the train sentiment classification
    with open(split_dir+"/split/train", 'rb') as f:
        train = pickle.load(f)

    unlabeled, source, target = pre.XML2arrayRAW("data/" + src + "/" + src + "UN.txt",
                                             "data/" + dest + "/" + dest + "UN.txt")
    unlabeled = getClearList(unlabeled)
    train = getClearList(train)
    source_valid = len(source)/5
    target_valid = len(target)/5
    tok = Tokenizer(num_words = topWords, split=" ")
    tok.fit_on_texts(train + unlabeled)
    x_valid = unlabeled[:source_valid]+unlabeled[-target_valid:]
    x = unlabeled[source_valid:-target_valid]+train
    

    #you can reload the pivots if you want to avoid the pivot extraction
    '''
    filename =src + "_to_" + dest + "/pivots/"+str(pivot_num)

    with open(filename, 'rb') as f:
        names = pickle.load(f)
    '''
    filename = src + "_to_" + dest + "/pivots/" + str(pivot_num) + "_meta"
    with open(filename, 'rb') as f:
        meta = pickle.load(f)
    revel_num = pivot_num/(iter_num-1)
    print "revel ",revel_num*(current_iter)
    print "the names   ",names
    print "the hiddens ",names[revel_num*(current_iter):]


    Pdict,class_count = fill_pivot_dict(names,names[revel_num*(current_iter):],meta)
    X_train = tok.texts_to_sequences(x)
    X_test  = tok.texts_to_sequences(x_valid)

    #creates the model
    embedding_vecor_length = word_vector_size
    model = Sequential()
    model.add(Embedding(topWords, embedding_vecor_length, mask_zero=True, embeddings_initializer="glorot_uniform", input_length=max_review_len ))
    model.add(LSTM(hidden_units_num, return_sequences=True))
    model_str = src + "_to_" + dest + "/models/" + criteria + "-"+str(iter_num) + "/" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(
        hidden_units_num) + "_" +str(word_vector_size)+ "_" + ".model"
    num_class = class_count + 2
    model.add(TimeDistributed(Dense(num_class, activation='softmax')))

    if(current_iter != 0):
        old_model_str = model_str+"."+str(current_iter-1)
        model = GetNewModel(old_model_str, model,names ,meta)

    new_model_str = model_str+"."+str(current_iter)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],sample_weight_mode =  "temporal")
    print(model.summary())
    if not os.path.exists(os.path.dirname(new_model_str)):
        os.makedirs(os.path.dirname(new_model_str))
    #saves only the best model with respect to the validaion loss
    modelCheckpoint = ModelCheckpoint(new_model_str, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='min', period=1)
    #stops the training if the validation loss has not decreased during the last 2 epochs
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    model.fit_generator(generator(X_train, 16, names, Pdict,tok,max_review_len,class_count),
                        steps_per_epoch=(len(X_train)/16), epochs=10, validation_data =
                        generator_val(X_test, 16, names, Pdict,tok,max_review_len,class_count),validation_steps=(len(X_test)/16)
                        ,callbacks=[earlyStopping,modelCheckpoint])

def train(src,dest,pivot_num,pivot_min_st,word_vector_size,topWords,max_review_len,hidden_units_num,iter_num,criteria):
    if(criteria == "BasicTRL"):
        assert (iter_num == 2)
    names,source_count,target_count = pre.preproc(pivot_num, pivot_min_st, src, dest)
    filename = src + "_to_" + dest + "/pivots/" + str(pivot_num) + "_meta"
    with open(filename, 'rb') as f:
        meta = pickle.load(f)
    namesByCriteria = SortByCriteria(criteria, meta, names,source_count,target_count)

    for i in range(iter_num):
        train_PBLM(src, dest, pivot_num, pivot_min_st, word_vector_size, topWords, max_review_len, hidden_units_num,
                   namesByCriteria, i, iter_num,criteria)

def SortByCriteria(criteria, meta, names,source_count,target_count):
    sen = []
    MI = []
    src_count = []
    dest_count = []
    sortedByCriteria = []
    if (criteria == "BasicTRL"):
        return names

    for n in names:
        sentiment, MutualInformation, src_frequency, dest_frequency = meta[n]
        sen.append(sentiment)
        MI.append(MutualInformation)
        src_count.append(src_frequency)
        dest_count.append(dest_frequency)
    if(criteria == "RMI"):
        sortedByCriteria = [names for _, names in sorted(zip(MI, names), reverse=True)]

    if(criteria == "RF"):
        freq = []
        pivot_num = len(names)
        for i in range(pivot_num):
            freq.append(src_count[i]+dest_count[i])
        sortedByCriteria = [names for _, names in sorted(zip(freq, names), reverse=True)]

    if(criteria == "RSF"):
        freq = []
        pivot_num = len(names)
        for i in range(pivot_num):
            relative_src = src_count[i]/float(source_count)
            relative_dest = dest_count[i]/float(target_count)
            similarity = min(relative_src,relative_dest)/float(max(relative_src,relative_dest))
            freq.append(similarity)
        sortedByCriteria = [names for _, names in sorted(zip(freq,names), reverse=True)]

    return sortedByCriteria
