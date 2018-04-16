# Author: Kim Hammar <kimham@kth.se> KTH 2018

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tag.perceptron import PerceptronTagger
import nltk
import emoji
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

class PreProcessor(object):
    """
    Preprocessor module in the Information Extraction Process of Fashion Related Properties of Instagram posts.
    Performs text normalization and parsing.
    """

    # Class variables shared by all instances
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    tagger = PerceptronTagger()

    def __init__(self, ids, comments, captions, tags):
        """ Class Constructor"""

        # Raw text
        self.raw_id = ids
        self.raw_comments = comments
        self.raw_captions = captions
        self.raw_tags = tags
        print("Read in Raw Text")

        # Preprocess raw text
        self.remove_non_unicode()
        self.lower_case()
        self.to_unicode()
        print("Normalized Raw Text")

        # Tokenize and preprocess tokens
        self.tokenize()
        print("Tokenized the text")
        self.remove_stopwords()
        #self.remove_urls()
        print("Normalized tokens")

        # Extract specific tokens
        self.lemmatize()
        print("Extracted lemmas")
        self.extract_emojis()
        print("Extracted emojis")
        self.extract_hashtags()
        print("Extracted hashtags")
        #self.pos_tag()
        #print("Extracted POS")

    def remove_non_unicode(self):
        """ Remove non-unicode tokens"""
        self.raw_comments = map(lambda x: x.decode('utf-8','ignore').encode("utf-8"), self.raw_comments)
        self.raw_captions = map(lambda x: x.decode('utf-8', 'ignore').encode("utf-8"), self.raw_captions)
        self.raw_tags = map(lambda x: x.decode('utf-8','ignore').encode("utf-8"), self.raw_tags)

    def to_unicode(self):
        """ Convert text to unicode """
        self.raw_comments = map(lambda x: x.decode('utf-8'), self.raw_comments)
        self.raw_captions = map(lambda x: x.decode('utf-8'), self.raw_captions)
        self.raw_tags = map(lambda x: x.decode('utf-8'), self.raw_tags)

    def tokenize(self):
        """ Tokenize text with TweetTokenizer, preserve emojis, hashtags etc """
        self.tokens_captions = [self.tknzr.tokenize(caption) for caption in self.raw_captions]
        self.tokens_comments = [self.tknzr.tokenize(comment) for comment in self.raw_comments]
        self.tokens_tags = [self.tknzr.tokenize(tag) for tag in self.raw_tags]
        self.tokens_all = []
        for i in range(len(self.raw_id)):
            self.tokens_all.append(self.tokens_captions[i] + self.tokens_comments[i] + self.tokens_tags[i])

    def lower_case(self):
        """ Convert raw text into lowercase"""
        self.raw_captions = [caption.lower() for caption in self.raw_captions]
        self.raw_comments = [comments.lower() for comments in self.raw_comments]
        self.raw_tags = [tags.lower() for tags in self.raw_tags]

    def lemmatize(self):
        """ Lemmatize tokens"""
        self.lemma_caption = [map(lambda x: self.wordnet_lemmatizer.lemmatize(x), caption) for caption in self.tokens_captions]
        self.lemma_comments = [map(lambda x: self.wordnet_lemmatizer.lemmatize(x), comments) for comments in self.tokens_comments]
        self.lemma_tags = [map(lambda x: self.wordnet_lemmatizer.lemmatize(x), tags) for tags in self.tokens_tags]
        self.lemma_all = [map(lambda x: self.wordnet_lemmatizer.lemmatize(x), tokens) for tokens in self.tokens_all]

    def remove_urls(self):
        """ Remove urls from tokens """
        self.tokens_captions = [filter(lambda x: "http" not in x, caption) for caption in self.tokens_captions]
        self.tokens_comments = [filter(lambda x: "http" not in x, comments) for comments in self.tokens_comments]
        self.tokens_tags = [filter(lambda x: "http" not in x, tags) for tags in self.tokens_tags]
        self.tokens_all = [filter(lambda x: "http" not in x, tokens) for tokens in self.tokens_all]

    def remove_stopwords(self):
        """ Remove stopwords from tokens """
        self.tokens_captions = [[token for token in caption if token not in self.stop_words] for caption in self.tokens_captions]
        self.tokens_comments = [[token for token in comments if token not in self.stop_words] for comments in self.tokens_comments]
        self.tokens_tags = [[token for token in tags if token not in self.stop_words] for tags in self.tokens_tags]
        self.tokens_all = [[token for token in tokens if token not in self.stop_words] for tokens in self.tokens_all]

    def extract_emojis(self):
        """ Extract emojis """
        self.emojis = [[c for c in tokens if c in emoji.UNICODE_EMOJI] for tokens in self.tokens_all]

    def extract_hashtags(self):
        """ Extract hashtags """
        self.hashtags = [[x for x in tokens if x.startswith("#")] for tokens in self.tokens_all]

    def pos_tag(self):
        """ Extract POS tags """
        self.pos_tokens = [self.tagger.tag(tokens) for tokens in self.tokens_all]
