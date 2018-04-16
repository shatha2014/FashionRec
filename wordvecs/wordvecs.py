#!/usr/bin/env python
# coding=utf-8
# Author: Kim Hammar <kimham@kth.se> KTH 2018
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

from fastText import train_unsupervised
from fastText import load_model
import os
import gensim, logging
from gensim.scripts.glove2word2vec import glove2word2vec
import errno
from glove import Corpus, Glove
from datetime import datetime
import sys
from copy import deepcopy
import re

""" 
Script for training and evaluating word vectors. 
Non-distributed training
"""

WORD2VEC_ANALOGIES = "eval/questions-words.txt"
WORDSIM353 = "eval/wordsim353.tsv"
FASHION_WORDSIM = "eval/fashion_wordsim.tsv"
SIMLEX99 = 'eval/simlex999.txt'
reload(sys)
sys.setdefaultencoding('utf-8')
isNumber = re.compile(r'\d+.*')


def readCorpus():
    """Reads input corpus, assumes it is already cleaned"""
    with open("data/clean2_corpus.txt", 'r') as datafile:
        return datafile.read()


def corpus_stats(corpus_file):
    """Computes some stats of the corpus"""
    vocab = set()
    total_count = 0
    with open(corpus_file, 'r') as corpusFile:
        text = corpusFile.read()
        lines = text.split("\n")
        lines = map(lambda x: x.split(" "), lines)
        for line in lines:
            for word in line:
                total_count += 1
                vocab.add(word.lower())
    vocab_size = len(vocab)
    return total_count, vocab_size


def accuracy_percentage(acc):
    """Utility function for pretty-printing evaluation results"""
    num_questions = len(acc)
    semantic_questions = range(5)
    syntactic_questions = range(5, num_questions)
    overall_nr_correct_answers = sum(len(acc[i]["correct"]) for i in range(num_questions))
    overall_nr_incorrect_answers = sum(len(acc[i]["incorrect"]) for i in range(num_questions))
    if (overall_nr_correct_answers + overall_nr_incorrect_answers) > 0:
        overall_acc_percent = 100 * float(
            overall_nr_correct_answers / (overall_nr_correct_answers + overall_nr_incorrect_answers))
    else:
        overall_acc_percent = 0
    sem_nr_correct_answers = sum(len(acc[i]["correct"]) for i in semantic_questions)
    sem_nr_incorrect_answers = sum(len(acc[i]["incorrect"]) for i in semantic_questions)
    if (sem_nr_correct_answers + sem_nr_incorrect_answers) > 0:
        sem_acc_percent = 100 * float(sem_nr_correct_answers / (sem_nr_correct_answers + sem_nr_incorrect_answers))
    else:
        sem_acc_percent = 0
    syn_nr_correct_answers = sum(len(acc[i]["correct"]) for i in syntactic_questions)
    syn_nr_incorrect_answers = sum(len(acc[i]["incorrect"]) for i in syntactic_questions)
    if (syn_nr_correct_answers + syn_nr_incorrect_answers) > 0:
        syn_acc_percent = 100 * float(syn_nr_correct_answers / (syn_nr_correct_answers + syn_nr_incorrect_answers))
    else:
        syn_acc_percent = 0
    frac = "sem: {0}/{1}, syn: {2}/{3}".format(sem_nr_correct_answers,
                                               (sem_nr_correct_answers + sem_nr_incorrect_answers),
                                               syn_nr_correct_answers,
                                               (syn_nr_correct_answers + syn_nr_incorrect_answers))
    return overall_acc_percent, sem_acc_percent, syn_acc_percent, frac


def save_results(model, dim, context, train_model, algorithm, data, name):
    """Save evaluation results to CSV file"""
    acc = model.accuracy(WORD2VEC_ANALOGIES)
    overall_acc_percent, sem_acc_percent, syn_acc_percent, frac = accuracy_percentage(acc)
    pearson, spearman, oov_ration = model.evaluate_word_pairs(WORDSIM353)
    pearson2, spearman2, oov_ration2 = model.evaluate_word_pairs(SIMLEX99)
    pearson3, spearman3, oov_ration3 = model.evaluate_word_pairs(FASHION_WORDSIM)
    fields = [
        str(syn_acc_percent), str(sem_acc_percent), str(overall_acc_percent),
        str(pearson[0]), str(pearson[1]), str(spearman[0]), str(spearman[1]), str(oov_ration),
        str(pearson2[0]), str(pearson2[1]), str(spearman2[0]), str(spearman2[1]), str(oov_ration2),
        str(pearson3[0]), str(pearson3[1]), str(spearman3[0]), str(spearman3[1]), str(oov_ration3),
        str(dim), str(context), str(train_model), str(algorithm), str(data), str(name)
    ]
    strFields = ",".join(fields)
    strFields = strFields + "\n"
    append_to_file("results/eval/results.csv", strFields)


def test_word2vec_google_news_300():
    """ Evaluate word2vec pretrained on google news"""
    model = gensim.models.KeyedVectors.load_word2vec_format('pretrained/googlenews_negative_300d_100B.bin', binary=True)
    name = "googlenews_negative_300d_100B"
    save_results(model, 300, 5, "skipgram", "word2vec", "100billion_googlenews_en", name)


def test_fasttext_wiki_300():
    """ Evaluate fastttext pretrained on Eng Wikipedia"""
    model = gensim.models.KeyedVectors.load_word2vec_format("pretrained/fasttext_wiki_300d_en.vec")
    name = "fasttext_wiki_300d_en"
    save_results(model, 300, 5, "skipgram", "fasttext", "wiki_en", name)


def test_glove_wiki_300():
    """ Evaluate Glove pretrained on Eng wikipedia"""
    model = gensim.models.KeyedVectors.load_word2vec_format('pretrained/glove_wiki_6B_300d.vec', binary=False)
    name = "glove_wiki_6B_300d"
    save_results(model, 300, "?", "-", "glove", "6billion_wiki", name)


def test_glove_twitter_200():
    """ Evaluate Glove pretrained on google Twitter"""
    model = gensim.models.KeyedVectors.load_word2vec_format('pretrained/twitter_glove_27B_200d.vec', binary=False)
    name = "twitter_glove_27B_200d"
    save_results(model, 200, "?", "-", "glove", "27billion_twitter", name)


def test_glove_commoncrawl_300():
    """ Evaluate Glove pretrained on common crawl corpora"""
    model = gensim.models.KeyedVectors.load_word2vec_format('pretrained/commoncrawl_glove_840B_300d.vec', binary=False)
    name = "commoncrawl_glove_840B_300d"
    save_results(model, 300, "?", "-", "glove", "840billion_commoncrawl", name)


def test_fashion_retrofitted():
    """ Evaluate retrofitted fashion vectors"""
    vectorFile = "retrofitted/test.vec"
    model = gensim.models.KeyedVectors.load_word2vec_format(vectorFile, binary=False)
    save_results(model, 300, 3, "?", "glove", "74million_fashion", "test")


def test_fashion(dim, context, train_model, algorithm, binary):
    """ Evaluate our own vectors trained on IG corpora """
    vectorFile = "trained/" + str(algorithm) + "_fashion_dim" + str(dim) + "_c" + str(context) + "_" + str(
        train_model) + ".vec"
    name = str(algorithm) + "_fashion_dim" + str(dim) + "_c" + str(context) + "_" + str(train_model)
    model = gensim.models.KeyedVectors.load_word2vec_format(vectorFile, binary=binary)
    save_results(model, dim, context, train_model, algorithm, "74million_fashion", name)


def convert_gensim_to_word2vec_format(fileName):
    """Converts gensim exportation format to word2vec format"""
    model = gensim.models.KeyedVectors.load(fileName)
    word_vectors = model.wv
    word_vectors.save_word2vec_format(fileName)


def convert_glove_to_word2vec_format():
    """ Converts Glove format to Word2Vec format"""
    glove2word2vec(glove_input_file="pretrained/glove_wiki_6B_300d.txt",
                   word2vec_output_file="pretrained/glove_wiki_6B_300d.vec")
    glove2word2vec(glove_input_file="pretrained/glove_wiki_6B_200d.txt",
                   word2vec_output_file="pretrained/glove_wiki_6B_200d.vec")
    glove2word2vec(glove_input_file="pretrained/glove_wiki_6B_100d.txt",
                   word2vec_output_file="pretrained/glove_wiki_6B_100d.vec")
    glove2word2vec(glove_input_file="pretrained/glove_wiki_6B_50d.txt",
                   word2vec_output_file="pretrained/glove_wiki_6B_50d.vec")
    glove2word2vec(glove_input_file="pretrained/twitter_glove_27B_25d.txt",
                   word2vec_output_file="pretrained/twitter_glove_27B_25d.vec")
    glove2word2vec(glove_input_file="pretrained/twitter_glove_27B_50d.txt",
                   word2vec_output_file="pretrained/twitter_glove_27B_50d.vec")
    glove2word2vec(glove_input_file="pretrained/twitter_glove_27B_100d.txt",
                   word2vec_output_file="pretrained/twitter_glove_27B_100d.vec")
    glove2word2vec(glove_input_file="pretrained/twitter_glove_27B_200d.txt",
                   word2vec_output_file="pretrained/twitter_glove_27B_200d.vec")


def save_fasttext_bin_to_vec(model, output_path):
    """Converts FastText binary format to word2vec format"""
    words = model.get_words()
    with open(output_path, 'w+') as vecFile:
        vecFile.write((str(len(words)) + " " + str(model.get_dimension())) + "\n")
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                vecFile.write((w + vstr + "\n"))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass


def save_glove_bin_to_vec(model, output_path):
    """Converts Glove binary format to word2vec format"""
    with open(output_path, 'w+') as vecFile:
        (rows, cols) = model.word_vectors.shape
        vecFile.write(str(rows) + " " + str(cols) + "\n")
        for word, idx in model.dictionary.iteritems():
            v = model.word_vectors[idx]
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                vecFile.write((word + vstr + "\n"))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass


def save_retrofitted_to_vec(wordVecs, output_path):
    """ Save retrofitted vectors to word2vec format"""
    with open(output_path, 'w+') as vecFile:
        rows = len(wordVecs.keys())
        cols = len(wordVecs.itervalues().next())
        vecFile.write(str(rows) + " " + str(cols) + "\n")
        for word, v in wordVecs.iteritems():
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                vecFile.write((word + vstr + "\n"))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass


# Default params
# epochs:15
# threads:8
# minCount:5
# learning rate: 0.05
# learning rate update reate: 100
# wordNgrams: 1
# minn: 3
# maxn: 6
# neg: 5
# t= 1e-4
def train_fasttext_fashionrec(dimensionality, context, train_model, epochs):
    """ Train with FastText on IG corpora"""
    total_count, vocab_size = corpus_stats("data/clean2_corpus.txt")
    print("total word count: {}, vocabulary size: {}".format(total_count, vocab_size))
    start_time = datetime.now()
    model = train_unsupervised(
        input=os.path.join("data/clean2_corpus.txt"),
        dim=dimensionality,
        ws=context,
        model=train_model,
        epoch=15
    )
    time_elapsed = datetime.now() - start_time
    output_path = "trained/fasttext_fashion_dim" + str(dimensionality) + "_c" + str(context) + "_" + str(train_model)
    model.save_model(output_path + ".bin")
    save_fasttext_bin_to_vec(load_model(output_path + ".bin"), output_path + ".vec")
    fileName = "results/training/fasttext_fashion_epoch" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_" + str(train_model) + ".txt"
    notes = "FastText FashionData, " + str(epochs) + " epochs, " + str(dimensionality) + " dim, " + str(
        context) + " context, " + str(train_model) + " train mode\n" + "Training time: " + str(time_elapsed)
    save_to_file(fileName, notes)


# Default params:
# epochs: 15
# threads: 8
# alpha (learning rate):0.025
# min_count=5
# seed=1
# negative=5 (number of negative samples)
# cbow_mean=1
def train_word2vec_fashionrec(dimensionality, context, train_model, epochs):
    """ Train with Word2Vec on IG corpora"""
    total_count, vocab_size = corpus_stats("data/clean2_corpus.txt")
    print("total word count: {}, vocabulary size: {}".format(total_count, vocab_size))
    sentences = gensim.models.word2vec.LineSentence("data/clean2_corpus.txt")
    start_time = datetime.now()
    # sg = 1 => skip-gram, sg = 0 => CBOW
    model = gensim.models.Word2Vec(sentences, size=dimensionality, window=context, workers=8, sg=train_model, iter=15)
    time_elapsed = datetime.now() - start_time
    word_vectors = model.wv
    output_path = "trained/word2vec_fashion_dim" + str(dimensionality) + "_c" + str(context) + "_" + str(train_model)
    word_vectors.save(output_path + ".vec")
    fileName = "results/training/word2vec_fashion_epoch" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_" + str(train_model) + ".txt"
    notes = "Word2Vec Fashion Data, " + str(epochs) + " epochs, " + str(dimensionality) + " dim, " + str(
        context) + " context, " + str(train_model) + " train mode\n" + "Training time: " + str(time_elapsed)
    save_to_file(fileName, notes)


def train_word2vec_wordrank(dimensionality, context, train_model, epochs):
    """ Train with Word2vec on IG corpora"""
    total_count, vocab_size = corpus_stats("data/clean2_corpus.txt")
    print("total word count: {}, vocabulary size: {}".format(total_count, vocab_size))
    sentences = gensim.models.word2vec.LineSentence("data/clean2_corpus.txt")
    start_time = datetime.now()
    # sg = 1 => skip-gram, sg = 0 => CBOW
    model = gensim.models.Word2Vec(sentences, size=dimensionality, window=context, workers=8, sg=train_model, iter=15)
    time_elapsed = datetime.now() - start_time
    word_vectors = model.wv
    output_path = "trained/word2vec_fashion_dim" + str(dimensionality) + "_c" + str(context) + "_" + str(train_model)
    word_vectors.save(output_path + ".vec")
    fileName = "results/training/word2vec_fashion_epoch" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_" + str(train_model) + ".txt"
    notes = "Word2Vec Fashion Data, " + str(epochs) + " epochs, " + str(dimensionality) + " dim, " + str(
        context) + " context, " + str(train_model) + " train mode\n" + "Training time: " + str(time_elapsed)
    save_to_file(fileName, notes)


# Default params:
# epochs: 15
# threads: 8
# learning-rate:0.05
# alpha:0.75
# max_count:100
# max_loss: 10
# no_components:30
# symmetric context
def train_glove_fashionrec(dimensionality, context, epochs):
    """ Train with Glove on IG corpora"""
    total_count, vocab_size = corpus_stats("data/clean2_corpus.txt")
    print("total word count: {}, vocabulary size: {}".format(total_count, vocab_size))
    fileName = "results/training/glove_fashion_epochs" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_" + ".txt"
    corpus = readCorpus()
    lines = corpus.split("\n")
    linessplit = map(lambda x: x.split(" "), lines)
    corpus_model = Corpus()
    start_time = datetime.now()
    corpus_model.fit(linessplit, window=context)
    corpusModelFile = "trained/glove_fashion_epochs" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_corpus" + ".model"
    corpus_model.save(corpusModelFile)
    glove = Glove(no_components=dimensionality, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=int(epochs),
              no_threads=8, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    time_elapsed = datetime.now() - start_time
    gloveModelFile = "trained/glove_fashion_epochs" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_vecs" + ".model"
    glove.save(gloveModelFile)
    notes = "Glove Fashion Data," + str(dimensionality) + " dim, " + str(context) + " context, " + str(
        epochs) + " epochs \n" + "Training time: " + str(time_elapsed)
    save_to_file(fileName, notes)
    gloveVecFile = "trained/glove_fashion_epochs" + str(epochs) + "_d" + str(dimensionality) + "_c" + str(
        context) + "_vecs" + ".vec"
    save_glove_bin_to_vec(glove, gloveVecFile)


def save_to_file(fileName, text):
    """Utility function for saving to string to file"""
    with open(fileName, 'w+') as file:
        file.write(text)


def append_to_file(fileName, text):
    """Utility function for appending string to file"""
    with open(fileName, 'a') as file:
        file.write(text)

def my_vector_getter(words, word, my_coordinates):
    """function that returns word vector as numpy array"""
    index = words.index(word)
    word_array = my_coordinates[index].ravel()
    return (word_array)

def norm_word(word):
    """ Computes normalized form of word for Retrofitting"""
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()

def read_lexicon(filename):
    """ Reads lexicon file"""
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def retrofit(wordVecs, lexicon, numIters):
    """ Retrofit word vectors """
    newWordVecs = deepcopy(wordVecs)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    for it in range(numIters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loopVocab:
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            numNeighbours = len(wordNeighbours)
            # no neighbours, pass - use data estimate
            if numNeighbours == 0:
                continue
            # the weight of the data estimate if the number of neighbours
            newVec = numNeighbours * wordVecs[word]
            # loop over neighbours and add to new vector (currently with weight 1)
            for ppWord in wordNeighbours:
                newVec += newWordVecs[ppWord]
            newWordVecs[word] = newVec / (2 * numNeighbours)
        return newWordVecs


def gensimModelToDict(model):
    """ Convert gensim model to dict to use in Retrofitting algorithm"""
    wordVecs = {}
    for word in model.wv.vocab.keys():
        wordVecs[word] = model.wv.syn0[model.wv.vocab[word].index]
        # Optional normalization
        # wordVecs[word] /= math.sqrt((wordVecs[word]**2).sum() + 1e-6) #normalize
    return wordVecs


def does_not_match():
    """ Does-not-match evaluation of word vectors"""
    # model = gensim.models.KeyedVectors.load_word2vec_format('pretrained/googlenews_negative_300d_100B.bin', binary=True)
    model = gensim.models.KeyedVectors.load_word2vec_format("trained/fasttext_fashion_dim300_c3_skipgram.vec",
                                                            binary=False)
    print("dress jacket shirt coat green, does not match: {0}".format(
        model.doesnt_match("dress jacket shirt coat green".split())))
    print(
        "sweater jersey hoodie pullover shoe".format(model.doesnt_match("sweater jersey hoodie pullover shoe".split())))
    print("shoe boot sneaker trainer sandal hat".format(
        model.doesnt_match("shoe boot sneaker trainer sandal hat".split())))


def retrofitting():
    """ Orchestrates retrofitting of word vectors"""
    # model = gensim.models.KeyedVectors.load_word2vec_format("trained/fasttext_fashion_dim300_c3_skipgram.vec", binary=False)
    # model = gensim.models.KeyedVectors.load_word2vec_format("trained/word2vec_fashion_dim300_c3_1.vec", binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format("trained/glove_fashion_dim300_c3_-.vec", binary=False)
    wordVecs = gensimModelToDict(model)
    lexicon = read_lexicon("./lexicon/framenet.txt")
    numIter = int(10)
    outFileName = "retrofitted/test.vec"
    retrofittedVecs = retrofit(wordVecs, lexicon, numIter)
    save_retrofitted_to_vec(retrofittedVecs, outFileName)


def test():
    """ Evaluate All """
    test_word2vec_google_news_300()
    test_fasttext_wiki_300()
    test_glove_wiki_300()
    test_glove_twitter_200()

    test_fashion(300, 1, "skipgram", "fasttext", False)
    test_fashion(300, 2, "skipgram", "fasttext", False)
    test_fashion(300, 3, "skipgram", "fasttext", False)
    test_fashion(300, 4, "skipgram", "fasttext", False)
    test_fashion(300, 5, "skipgram", "fasttext", False)
    test_fashion(300, 6, "skipgram", "fasttext", False)
    test_fashion(300, 7, "skipgram", "fasttext", False)
    test_fashion(300, 8, "skipgram", "fasttext", False)
    test_fashion(300, 9, "skipgram", "fasttext", False)
    test_fashion(300, 10, "skipgram", "fasttext", False)
    test_fashion(300, 11, "skipgram", "fasttext", False)
    test_fashion(300, 12, "skipgram", "fasttext", False)
    test_fashion(300, 13, "skipgram", "fasttext", False)
    test_fashion(300, 14, "skipgram", "fasttext", False)

    test_fashion(300, 1, "cbow", "fasttext", False)
    test_fashion(300, 2, "cbow", "fasttext", False)
    test_fashion(300, 3, "cbow", "fasttext", False)
    test_fashion(300, 4, "cbow", "fasttext", False)
    test_fashion(300, 5, "cbow", "fasttext", False)
    test_fashion(300, 6, "cbow", "fasttext", False)
    test_fashion(300, 7, "cbow", "fasttext", False)
    test_fashion(300, 8, "cbow", "fasttext", False)
    test_fashion(300, 9, "cbow", "fasttext", False)
    test_fashion(300, 10, "cbow", "fasttext", False)
    test_fashion(300, 11, "cbow", "fasttext", False)
    test_fashion(300, 12, "cbow", "fasttext", False)
    test_fashion(300, 13, "cbow", "fasttext", False)
    test_fashion(300, 14, "cbow", "fasttext", False)

    test_fashion(300, 1, "-", "glove", False)
    test_fashion(300, 2, "-", "glove", False)
    test_fashion(300, 3, "-", "glove", False)
    test_fashion(300, 4, "-", "glove", False)
    test_fashion(300, 5, "-", "glove", False)
    test_fashion(300, 6, "-", "glove", False)
    test_fashion(300, 7, "-", "glove", False)
    test_fashion(300, 8, "-", "glove", False)
    test_fashion(300, 9, "-", "glove", False)
    test_fashion(300, 10, "-", "glove", False)
    test_fashion(300, 11, "-", "glove", False)
    test_fashion(300, 12, "-", "glove", False)
    test_fashion(300, 13, "-", "glove", False)
    test_fashion(300, 14, "-", "glove", False)

    test_fashion(300, 1, "1", "word2vec", False)
    test_fashion(300, 2, "1", "word2vec", False)
    test_fashion(300, 3, "1", "word2vec", False)
    test_fashion(300, 4, "1", "word2vec", False)
    test_fashion(300, 5, "1", "word2vec", False)
    test_fashion(300, 6, "1", "word2vec", False)
    test_fashion(300, 7, "1", "word2vec", False)
    test_fashion(300, 8, "1", "word2vec", False)
    test_fashion(300, 9, "1", "word2vec", False)
    test_fashion(300, 10, "1", "word2vec", False)
    test_fashion(300, 11, "1", "word2vec", False)
    test_fashion(300, 12, "1", "word2vec", False)
    test_fashion(300, 13, "1", "word2vec", False)
    test_fashion(300, 14, "1", "word2vec", False)

    test_fashion(300, 1, "0", "word2vec", False)
    test_fashion(300, 2, "0", "word2vec", False)
    test_fashion(300, 3, "0", "word2vec", False)
    test_fashion(300, 4, "0", "word2vec", False)
    test_fashion(300, 5, "0", "word2vec", False)
    test_fashion(300, 6, "0", "word2vec", False)
    test_fashion(300, 7, "0", "word2vec", False)
    test_fashion(300, 8, "0", "word2vec", False)
    test_fashion(300, 9, "0", "word2vec", False)
    test_fashion(300, 10, "0", "word2vec", False)
    test_fashion(300, 11, "0", "word2vec", False)
    test_fashion(300, 12, "0", "word2vec", False)
    test_fashion(300, 13, "0", "word2vec", False)
    test_fashion(300, 14, "0", "word2vec", False)

    test_fashion(50, 2, "skipgram", "fasttext", False)
    test_fashion(100, 2, "skipgram", "fasttext", False)
    test_fashion(150, 2, "skipgram", "fasttext", False)
    test_fashion(200, 2, "skipgram", "fasttext", False)
    test_fashion(250, 2, "skipgram", "fasttext", False)
    test_fashion(350, 2, "skipgram", "fasttext", False)
    test_fashion(400, 2, "skipgram", "fasttext", False)
    test_fashion(450, 2, "skipgram", "fasttext", False)
    test_fashion(500, 2, "skipgram", "fasttext", False)
    test_fashion(550, 2, "skipgram", "fasttext", False)
    test_fashion(600, 2, "skipgram", "fasttext", False)

def train_all():
    """ Train All"""

    train_fasttext_fashionrec(300, 3, "skipgram", 15)
    train_word2vec_fashionrec(300, 3, 1, 15)
    train_glove_fashionrec(300, 3, 15)
    train_fasttext_fashionrec(300, 3, "cbow", 15)
    train_word2vec_fashionrec(300, 3, 0, 15)
    train_fasttext_fashionrec(300, 4, "skipgram", 15)
    train_word2vec_fashionrec(300, 4, 1, 15)
    train_fasttext_fashionrec(300, 4, "cbow", 15)
    train_word2vec_fashionrec(300, 4, 0, 15)
    train_glove_fashionrec(300, 4, 15)
    train_fasttext_fashionrec(300, 5, "skipgram", 15)
    train_word2vec_fashionrec(300, 5, 1, 15)
    train_fasttext_fashionrec(300, 5, "cbow", 15)
    train_word2vec_fashionrec(300, 5, 0, 15)
    train_glove_fashionrec(300, 5, 15)
    train_fasttext_fashionrec(300, 6, "skipgram", 15)
    train_word2vec_fashionrec(300, 6, 1, 15)
    train_fasttext_fashionrec(300, 6, "cbow", 15)
    train_word2vec_fashionrec(300, 6, 0, 15)
    train_glove_fashionrec(300, 6, 15)
    train_fasttext_fashionrec(300, 7, "skipgram", 15)
    train_word2vec_fashionrec(300, 7, 1, 15)
    train_fasttext_fashionrec(300, 7, "cbow", 15)
    train_word2vec_fashionrec(300, 7, 0, 15)
    train_glove_fashionrec(300, 7, 15)
    train_fasttext_fashionrec(300, 8, "skipgram", 15)
    train_word2vec_fashionrec(300, 8, 1, 15)
    train_fasttext_fashionrec(300, 8, "cbow", 15)
    train_word2vec_fashionrec(300, 8, 0, 15)
    train_glove_fashionrec(300, 8, 15)
    train_fasttext_fashionrec(300, 9, "skipgram", 15)
    train_word2vec_fashionrec(300, 9, 1, 15)
    train_fasttext_fashionrec(300, 9, "cbow", 15)
    train_word2vec_fashionrec(300, 9, 0, 15)
    train_glove_fashionrec(300, 9, 15)
    train_fasttext_fashionrec(300, 10, "skipgram", 15)
    train_word2vec_fashionrec(300, 10, 1, 15)
    train_fasttext_fashionrec(300, 10, "cbow", 15)
    train_word2vec_fashionrec(300, 10, 0, 15)
    train_glove_fashionrec(300, 10, 15)

    train_fasttext_fashionrec(50, 3, "skipgram", 15)
    train_word2vec_fashionrec(50, 3, 1, 15)
    train_glove_fashionrec(50, 3, 15)
    train_fasttext_fashionrec(50, 3, "cbow", 15)
    train_word2vec_fashionrec(50, 3, 0, 15)
    train_fasttext_fashionrec(50, 4, "skipgram", 15)
    train_word2vec_fashionrec(50, 4, 1, 15)
    train_fasttext_fashionrec(50, 4, "cbow", 15)
    train_word2vec_fashionrec(50, 4, 0, 15)
    train_glove_fashionrec(50, 4, 15)
    train_fasttext_fashionrec(50, 5, "skipgram", 15)
    train_word2vec_fashionrec(50, 5, 1, 15)
    train_fasttext_fashionrec(50, 5, "cbow", 15)
    train_word2vec_fashionrec(50, 5, 0, 15)
    train_glove_fashionrec(50, 5, 15)
    train_fasttext_fashionrec(50, 6, "skipgram", 15)
    train_word2vec_fashionrec(50, 6, 1, 15)
    train_fasttext_fashionrec(50, 6, "cbow", 15)
    train_word2vec_fashionrec(50, 6, 0, 15)
    train_glove_fashionrec(50, 6, 15)
    train_fasttext_fashionrec(50, 7, "skipgram", 15)
    train_word2vec_fashionrec(50, 7, 1, 15)
    train_fasttext_fashionrec(50, 7, "cbow", 15)
    train_word2vec_fashionrec(50, 7, 0, 15)
    train_glove_fashionrec(50, 7, 15)
    train_fasttext_fashionrec(50, 8, "skipgram", 15)
    train_word2vec_fashionrec(50, 8, 1, 15)
    train_fasttext_fashionrec(50, 8, "cbow", 15)
    train_word2vec_fashionrec(50, 8, 0, 15)
    train_glove_fashionrec(50, 8, 15)
    train_fasttext_fashionrec(50, 9, "skipgram", 15)
    train_word2vec_fashionrec(50, 9, 1, 15)
    train_fasttext_fashionrec(50, 9, "cbow", 15)
    train_word2vec_fashionrec(50, 9, 0, 15)
    train_glove_fashionrec(50, 9, 15)
    train_fasttext_fashionrec(50, 10, "skipgram", 15)
    train_word2vec_fashionrec(50, 10, 1, 15)
    train_fasttext_fashionrec(50, 10, "cbow", 15)
    train_word2vec_fashionrec(50, 10, 0, 15)
    train_glove_fashionrec(50, 10, 15)

    train_fasttext_fashionrec(100, 3, "skipgram", 15)
    train_word2vec_fashionrec(100, 3, 1, 15)
    train_glove_fashionrec(100, 3, 15)
    train_fasttext_fashionrec(100, 3, "cbow", 15)
    train_word2vec_fashionrec(100, 3, 0, 15)
    train_fasttext_fashionrec(100, 4, "skipgram", 15)
    train_word2vec_fashionrec(100, 4, 1, 15)
    train_fasttext_fashionrec(100, 4, "cbow", 15)
    train_word2vec_fashionrec(100, 4, 0, 15)
    train_glove_fashionrec(100, 4, 15)
    train_fasttext_fashionrec(100, 5, "skipgram", 15)
    train_word2vec_fashionrec(100, 5, 1, 15)
    train_fasttext_fashionrec(100, 5, "cbow", 15)
    train_word2vec_fashionrec(100, 5, 0, 15)
    train_glove_fashionrec(100, 5, 15)
    train_fasttext_fashionrec(100, 6, "skipgram", 15)
    train_word2vec_fashionrec(100, 6, 1, 15)
    train_fasttext_fashionrec(100, 6, "cbow", 15)
    train_word2vec_fashionrec(100, 6, 0, 15)
    train_glove_fashionrec(100, 6, 15)
    train_fasttext_fashionrec(100, 7, "skipgram", 15)
    train_word2vec_fashionrec(100, 7, 1, 15)
    train_fasttext_fashionrec(100, 7, "cbow", 15)
    train_word2vec_fashionrec(100, 7, 0, 15)
    train_glove_fashionrec(100, 7, 15)
    train_fasttext_fashionrec(100, 8, "skipgram", 15)
    train_word2vec_fashionrec(100, 8, 1, 15)
    train_fasttext_fashionrec(100, 8, "cbow", 15)
    train_word2vec_fashionrec(100, 8, 0, 15)
    train_glove_fashionrec(100, 8, 15)
    train_fasttext_fashionrec(100, 9, "skipgram", 15)
    train_word2vec_fashionrec(100, 9, 1, 15)
    train_fasttext_fashionrec(100, 9, "cbow", 15)
    train_word2vec_fashionrec(100, 9, 0, 15)
    train_glove_fashionrec(100, 9, 15)
    train_fasttext_fashionrec(100, 10, "skipgram", 15)
    train_word2vec_fashionrec(100, 10, 1, 15)
    train_fasttext_fashionrec(100, 10, "cbow", 15)
    train_word2vec_fashionrec(100, 10, 0, 15)
    train_glove_fashionrec(100, 10, 15)

    train_fasttext_fashionrec(150, 3, "skipgram", 15)
    train_word2vec_fashionrec(150, 3, 1, 15)
    train_glove_fashionrec(150, 3, 15)
    train_fasttext_fashionrec(150, 3, "cbow", 15)
    train_word2vec_fashionrec(150, 3, 0, 15)
    train_fasttext_fashionrec(150, 4, "skipgram", 15)
    train_word2vec_fashionrec(150, 4, 1, 15)
    train_fasttext_fashionrec(150, 4, "cbow", 15)
    train_word2vec_fashionrec(150, 4, 0, 15)
    train_glove_fashionrec(150, 4, 15)
    train_fasttext_fashionrec(150, 5, "skipgram", 15)
    train_word2vec_fashionrec(150, 5, 1, 15)
    train_fasttext_fashionrec(150, 5, "cbow", 15)
    train_word2vec_fashionrec(150, 5, 0, 15)
    train_glove_fashionrec(150, 5, 15)
    train_fasttext_fashionrec(150, 6, "skipgram", 15)
    train_word2vec_fashionrec(150, 6, 1, 15)
    train_fasttext_fashionrec(150, 6, "cbow", 15)
    train_word2vec_fashionrec(150, 6, 0, 15)
    train_glove_fashionrec(150, 6, 15)
    train_fasttext_fashionrec(150, 7, "skipgram", 15)
    train_word2vec_fashionrec(150, 7, 1, 15)
    train_fasttext_fashionrec(150, 7, "cbow", 15)
    train_word2vec_fashionrec(150, 7, 0, 15)
    train_glove_fashionrec(150, 7, 15)
    train_fasttext_fashionrec(150, 8, "skipgram", 15)
    train_word2vec_fashionrec(150, 8, 1, 15)
    train_fasttext_fashionrec(150, 8, "cbow", 15)
    train_word2vec_fashionrec(150, 8, 0, 15)
    train_glove_fashionrec(150, 8, 15)
    train_fasttext_fashionrec(150, 9, "skipgram", 15)
    train_word2vec_fashionrec(150, 9, 1, 15)
    train_fasttext_fashionrec(150, 9, "cbow", 15)
    train_word2vec_fashionrec(150, 9, 0, 15)
    train_glove_fashionrec(150, 9, 15)
    train_fasttext_fashionrec(150, 10, "skipgram", 15)
    train_word2vec_fashionrec(150, 10, 1, 15)
    train_fasttext_fashionrec(150, 10, "cbow", 15)
    train_word2vec_fashionrec(150, 10, 0, 15)
    train_glove_fashionrec(150, 10, 15)

    train_fasttext_fashionrec(200, 3, "skipgram", 15)
    train_word2vec_fashionrec(200, 3, 1, 15)
    train_glove_fashionrec(200, 3, 15)
    train_fasttext_fashionrec(200, 3, "cbow", 15)
    train_word2vec_fashionrec(200, 3, 0, 15)
    train_fasttext_fashionrec(200, 4, "skipgram", 15)
    train_word2vec_fashionrec(200, 4, 1, 15)
    train_fasttext_fashionrec(200, 4, "cbow", 15)
    train_word2vec_fashionrec(200, 4, 0, 15)
    train_glove_fashionrec(200, 4, 15)
    train_fasttext_fashionrec(200, 5, "skipgram", 15)
    train_word2vec_fashionrec(200, 5, 1, 15)
    train_fasttext_fashionrec(200, 5, "cbow", 15)
    train_word2vec_fashionrec(200, 5, 0, 15)
    train_glove_fashionrec(200, 5, 15)
    train_fasttext_fashionrec(200, 6, "skipgram", 15)
    train_word2vec_fashionrec(200, 6, 1, 15)
    train_fasttext_fashionrec(200, 6, "cbow", 15)
    train_word2vec_fashionrec(200, 6, 0, 15)
    train_glove_fashionrec(200, 6, 15)
    train_fasttext_fashionrec(200, 7, "skipgram", 15)
    train_word2vec_fashionrec(200, 7, 1, 15)
    train_fasttext_fashionrec(200, 7, "cbow", 15)
    train_word2vec_fashionrec(200, 7, 0, 15)
    train_glove_fashionrec(200, 7, 15)
    train_fasttext_fashionrec(200, 8, "skipgram", 15)
    train_word2vec_fashionrec(200, 8, 1, 15)
    train_fasttext_fashionrec(200, 8, "cbow", 15)
    train_word2vec_fashionrec(200, 8, 0, 15)
    train_glove_fashionrec(200, 8, 15)
    train_fasttext_fashionrec(200, 9, "skipgram", 15)
    train_word2vec_fashionrec(200, 9, 1, 15)
    train_fasttext_fashionrec(200, 9, "cbow", 15)
    train_word2vec_fashionrec(200, 9, 0, 15)
    train_glove_fashionrec(200, 9, 15)
    train_fasttext_fashionrec(200, 10, "skipgram", 15)
    train_word2vec_fashionrec(200, 10, 1, 15)
    train_fasttext_fashionrec(200, 10, "cbow", 15)
    train_word2vec_fashionrec(200, 10, 0, 15)
    train_glove_fashionrec(200, 10, 15)

    train_fasttext_fashionrec(250, 3, "skipgram", 15)
    train_word2vec_fashionrec(250, 3, 1, 15)
    train_glove_fashionrec(250, 3, 15)
    train_fasttext_fashionrec(250, 3, "cbow", 15)
    train_word2vec_fashionrec(250, 3, 0, 15)
    train_fasttext_fashionrec(250, 4, "skipgram", 15)
    train_word2vec_fashionrec(250, 4, 1, 15)
    train_fasttext_fashionrec(250, 4, "cbow", 15)
    train_word2vec_fashionrec(250, 4, 0, 15)
    train_glove_fashionrec(250, 4, 15)
    train_fasttext_fashionrec(250, 5, "skipgram", 15)
    train_word2vec_fashionrec(250, 5, 1, 15)
    train_fasttext_fashionrec(250, 5, "cbow", 15)
    train_word2vec_fashionrec(250, 5, 0, 15)
    train_glove_fashionrec(250, 5, 15)
    train_fasttext_fashionrec(250, 6, "skipgram", 15)
    train_word2vec_fashionrec(250, 6, 1, 15)
    train_fasttext_fashionrec(250, 6, "cbow", 15)
    train_word2vec_fashionrec(250, 6, 0, 15)
    train_glove_fashionrec(250, 6, 15)
    train_fasttext_fashionrec(250, 7, "skipgram", 15)
    train_word2vec_fashionrec(250, 7, 1, 15)
    train_fasttext_fashionrec(250, 7, "cbow", 15)
    train_word2vec_fashionrec(250, 7, 0, 15)
    train_glove_fashionrec(250, 7, 15)
    train_fasttext_fashionrec(250, 8, "skipgram", 15)
    train_word2vec_fashionrec(250, 8, 1, 15)
    train_fasttext_fashionrec(250, 8, "cbow", 15)
    train_word2vec_fashionrec(250, 8, 0, 15)
    train_glove_fashionrec(250, 8, 15)
    train_fasttext_fashionrec(250, 9, "skipgram", 15)
    train_word2vec_fashionrec(250, 9, 1, 15)
    train_fasttext_fashionrec(250, 9, "cbow", 15)
    train_word2vec_fashionrec(250, 9, 0, 15)
    train_glove_fashionrec(250, 9, 15)
    train_fasttext_fashionrec(250, 10, "skipgram", 15)
    train_word2vec_fashionrec(250, 10, 1, 15)
    train_fasttext_fashionrec(250, 10, "cbow", 15)
    train_word2vec_fashionrec(250, 10, 0, 15)
    train_glove_fashionrec(250, 10, 15)

    train_glove_fashionrec(50, 12, 15)
    train_glove_fashionrec(100, 12, 15)
    train_glove_fashionrec(150, 12, 15)
    train_glove_fashionrec(200, 12, 15)
    train_glove_fashionrec(250, 12, 15)
    train_glove_fashionrec(300, 12, 15)
    train_glove_fashionrec(350, 12, 15)
    train_glove_fashionrec(400, 12, 15)
    train_glove_fashionrec(450, 12, 15)
    train_glove_fashionrec(500, 12, 15)
    train_glove_fashionrec(550, 12, 15)
    train_glove_fashionrec(600, 12, 15)
    train_word2vec_fashionrec(50, 13, 0, 15)
    train_word2vec_fashionrec(100, 13, 0, 15)
    train_word2vec_fashionrec(150, 13, 0, 15)
    train_word2vec_fashionrec(200, 13, 0, 15)
    train_word2vec_fashionrec(250, 13, 0, 15)
    train_word2vec_fashionrec(300, 13, 0, 15)
    train_word2vec_fashionrec(350, 13, 0, 15)
    train_word2vec_fashionrec(400, 13, 0, 15)
    train_word2vec_fashionrec(450, 13, 0, 15)
    train_word2vec_fashionrec(500, 13, 0, 15)
    train_word2vec_fashionrec(550, 13, 0, 15)
    train_word2vec_fashionrec(600, 13, 0, 15)

    train_word2vec_fashionrec(50, 3, 1, 15)
    train_word2vec_fashionrec(100, 3, 1, 15)
    train_word2vec_fashionrec(150, 3, 1, 15)
    train_word2vec_fashionrec(200, 3, 1, 15)
    train_word2vec_fashionrec(250, 3, 1, 15)
    train_word2vec_fashionrec(300, 3, 1, 15)
    train_word2vec_fashionrec(350, 3, 1, 15)
    train_word2vec_fashionrec(400, 3, 1, 15)
    train_word2vec_fashionrec(450, 3, 1, 15)
    train_word2vec_fashionrec(500, 3, 1, 15)
    train_word2vec_fashionrec(550, 3, 1, 15)
    train_word2vec_fashionrec(600, 3, 1, 15)

    train_fasttext_fashionrec(50, 2, "skipgram", 15)
    train_fasttext_fashionrec(100, 2, "skipgram", 15)
    train_fasttext_fashionrec(150, 2, "skipgram", 15)
    train_fasttext_fashionrec(200, 2, "skipgram", 15)
    train_fasttext_fashionrec(250, 2, "skipgram", 15)
    train_fasttext_fashionrec(350, 2, "skipgram", 15)
    train_fasttext_fashionrec(400, 2, "skipgram", 15)
    train_fasttext_fashionrec(450, 2, "skipgram", 15)
    train_fasttext_fashionrec(500, 2, "skipgram", 15)
    train_fasttext_fashionrec(550, 2, "skipgram", 15)
    train_fasttext_fashionrec(600, 2, "skipgram", 15)
    train_fasttext_fashionrec(300, 10, "cbow", 15)

    train_fasttext_fashionrec(50, 11, "cbow", 15)
    train_fasttext_fashionrec(100, 11, "cbow", 15)
    train_fasttext_fashionrec(150, 11, "cbow", 15)
    train_fasttext_fashionrec(200, 11, "cbow", 15)
    train_fasttext_fashionrec(250, 11, "cbow", 15)
    train_fasttext_fashionrec(350, 11, "cbow", 15)
    train_fasttext_fashionrec(400, 11, "cbow", 15)
    train_fasttext_fashionrec(450, 11, "cbow", 15)
    train_fasttext_fashionrec(500, 11, "cbow", 15)
    train_fasttext_fashionrec(550, 11, "cbow", 15)
    train_fasttext_fashionrec(600, 11, "cbow", 15)


def train():
    """ Train single"""
    train_fasttext_fashionrec(600, 2, "skipgram", 15)


def main():
    """ Program entrypoint"""
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # train_all()
    # train()
    # convert_gensim_to_word2vec_format("trained/word2vec_fashion_dim300_c13_0.vec")
    # convert_gensim_to_word2vec_format("trained/word2vec_fashion_dim300_c13_1.vec")
    # test()
    # retrofitting()
    # most_similar()
    # train_word2vec_fashionrec(300, 3, 1, 15)


if __name__ == "__main__":
    main()
