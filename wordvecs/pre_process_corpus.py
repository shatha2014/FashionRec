# coding=utf-8
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from collections import Counter

"""
Script for preprocessing corpora for training word vectors.
Cleans the text, removes un-frequent words, merges corpus to single file, computes some stats etc.  
"""

IG_CORPUS = "data/corpora/ig/ig_corpus.txt"
PDF_CORPUS = "data/corpora/pdf/pdf_corpus.txt"
TWITTER_CORPUS = "data/corpora/twitter/tweets.txt"
WIKI_CORPUS = "data/corpora/wiki/wiki_cleaned.txt"
ZALANDO_CORPUS = "data/corpora/zalando/zalando_corpus.txt"
FLICKR_CORPUS = "data/corpora/flickr/flickr_corpus.txt"
MAGENTO_CORPUS = "data/corpora/magento/magento_corpus.txt"
BLOG_CORPUS = "data/corpora/blog/blog_corpus.txt"
CORPORA = [IG_CORPUS, PDF_CORPUS, TWITTER_CORPUS, WIKI_CORPUS, ZALANDO_CORPUS, FLICKR_CORPUS, MAGENTO_CORPUS, BLOG_CORPUS]
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
wordnet_lemmatizer = WordNetLemmatizer()

def normalize_clean(corpusInput, corpusOutput):
    """remove stopwords, remove usertags, tokenize, lemmatize"""
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    count = 0
    with open(corpusInput, "r") as corpusInputFile:
        with open(corpusOutput, "w+") as corpusOutputFile:
            text = corpusInputFile.read()
            print "remove all non unicode characters"
            text = text.decode('utf-8','ignore').encode("utf-8")
            print "removing stopwords, usertags, urls, normalizing length, lower case only and tokenizing"
            list_of_words = [i.lower() for i in tknzr.tokenize(text.decode("utf-8")) if i.lower() not in stop_words] #remove stopwords & tokenize (preserve hashtags)
            print "remove twitter pics"
            list_of_words = filter(lambda x: "pic.twitter" not in x and not "twitter.com" in x, list_of_words)
            print "remove URLs"
            list_of_words = filter(lambda x: "http" not in x, list_of_words)
            print "lemmatizing"
            list_of_words = map(lambda x: wordnet_lemmatizer.lemmatize(x), list_of_words) # lemmatize
            print "encoding, counting, and saving to file"
            list_of_words = map(lambda x: x.encode("utf-8"), list_of_words)
            count += len(list_of_words)
            corpusOutputFile.write(" ".join(list_of_words))
    stat = corpusInput + "," + str(count)
    print stat
    return stat

def append_corpora(output):
    """Append corpora to single file"""
    count = 0
    with open(output, "w+") as corpusFile:
        for file in CORPORA:
            print("appending corpus {0}".format(file))
            fileH = open(file, "r")
            text = fileH.read()
            count += len(text.split(" "))
            corpusFile.write(text)
    stat = output + "," + str(count)
    print stat
    return stat

def corpora_stats():
    """Compute stats for each corpus"""
    stats = []
    for file in CORPORA:
        print("calculating stats for {0}".format(file))
        count = 0
        with open(file, 'r') as corpus:
            words = corpus.read().split(" ")
            count += len(words)
            vocabCount = len(set(words))
            stat = file + "," + str(count) + "," + str(vocabCount)
            stats.append(stat)
    with open("data/corpora_stats.txt", 'w+') as statsFile:
        print "\n".join(stats)
        statsFile.write("\n".join(stats))

def removeUnFrequentWords(corpusInput, corpusOutput):
    """Remove unfrequent words as they are useless for learning word vectors"""
    with open(corpusInput, "r") as corpusInputFile:
        with open(corpusOutput, "w+") as corpusOutputFile:
            text = corpusInputFile.read()
            words = text.split(" ")
            totalCount = len(words)
            totalUniqueCount = len(set(words))
            cnt = Counter(words)
            words = filter(lambda x: cnt[x] >= 5, words)
            totalFreqCount = len(words)
            totalUniqueFreqCount = len(set(words))
            corpusOutputFile.write(" ".join(words))
            stat = corpusInput + "," + str(totalCount) + "," + str(totalUniqueCount) +"\n" + corpusOutput + "," + str(totalFreqCount) + "," + str(totalUniqueFreqCount)
            print stat
            return stat

def main():
    """Program entry-point, orchestrates the pipeline"""
    print "calculating corpora stats"
    corpora_stats()
    print "appending corpora to single corpus file"
    append_stat = append_corpora("data/raw_corpus.txt")
    print "normalize text"
    after_clean_stat = normalize_clean("data/raw_corpus.txt", "data/clean1_corpus.txt")
    print "remove unfrequent words"
    after_freq_filter_stat = removeUnFrequentWords("data/clean1_corpus.txt", "data/clean2_corpus.txt")
    stats = "\n".join([append_stat, after_clean_stat, after_freq_filter_stat])
    with open("data/corpus_stats.txt", 'w+') as statsFile:
        statsFile.write(stats)


if __name__ == '__main__':
    main()