import tr
import sentiment
import pre
import lm
import os
import itertools


if __name__ == '__main__':
    domain = []
    domain.append("books")
    domain.append("kitchen")
    domain.append("dvd")
    domain.append("electronics")

    sorting_criteria = []
    sorting_criteria.append("BasicTRL")
    sorting_criteria.append("RF")
    sorting_criteria.append("RSF")
    sorting_criteria.append("RMI")

    # training the PBLM model in order to create structure aware
    #input:
    # shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth param: the embedding dimension
    # sixth param: maximum number of words to work with
    # seventh param: maximum review length to work with
    # eighth param: hidden units number for the PBLM model
    #output: the software will create corresponding directory with the model

    tr.train(domain[0], domain[1], 100, 10, 128, 10000, 500, 256, 2, sorting_criteria[0])





    # training the sentiment cnn using PBLM's representation
    # shared representation for both source domain and target domain
    # this phase needs a corresponding trained PBLM model in order to work
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: maximum review length to work with
    # fifth param: the embedding dimension
    # sixth param: maximum number of words to work with
    # seventh param: hidden units number for the PBLM model
    # eighth param: hidden units number for the lstm model
    # output: the results file will be created in the same directory
    # of the model under the results directory in the "lstm" dir
    #sentiment.PBLM_LSTM(domain[0], domain[1], 500, 500, 256, 10000, 256, 256)
    sentiment.PBLM_CNN(domain[0], domain[1], 100, 10, 128, 10000, 500, 256, 250, 3,  2, sorting_criteria[0])

