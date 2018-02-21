import gensim
import json
import urllib
from nltk.metrics import edit_distance
from nltk.stem import WordNetLemmatizer
import math
import lda
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from collections import Counter
import wikipedia
from googlesearch.googlesearch import GoogleSearch

class InformationExtractor(object):
    """ Module with functions for information Extraction """

    wordnet_lemmatizer = WordNetLemmatizer()
    api_key = open('/media/limmen/HDD/Dropbox/wordvec/ie/.api_key').read()
    google_service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    probase_service_url = "https://concept.research.microsoft.com/api/Concept/ScoreByProb"

    def __init__(self, word_vectors, companies, styles, materials, items,
                 brands_keywords_google, materials_keywords_google,
                 probase_brands,
                 probase_materials, colors, patterns):
        self.wordvec_model = gensim.models.KeyedVectors.load_word2vec_format(word_vectors, binary=False)
        self.companies = companies
        self.styles = styles
        self.materials = materials
        self.items = items
        self.brands_keywords_google = brands_keywords_google
        self.materials_keywords_google = materials_keywords_google
        self.probase_brands = probase_brands
        self.probase_materials = probase_materials
        self.colors = colors
        self.patterns = patterns
        self.lemmatize()

    def lemmatize(self):
        """ Lemmatize domain lists"""
        self.styles_lemmas = {self.wordnet_lemmatizer.lemmatize(style): style for style in self.styles}
        self.materials_lemmas = {self.wordnet_lemmatizer.lemmatize(material): material for material in self.materials}
        self.items_lemmas = {self.wordnet_lemmatizer.lemmatize(item): item for item in self.items}

    def find_closest_semantic(self, tokens, num, topic):
        """ Finds num semantically closest candidates for a given topic"""
        topic = map(lambda x: x.decode('utf-8','ignore').encode("utf-8"), topic)
        freq_scores = {}
        for x in topic:
            freq_scores[x] = 0
        for token in tokens:
            sim_scores = {}
            for x in topic:
                tokens2 = x.lower().split(" ")
                similarity = 0
                for token2 in tokens2:
                    if self.wordnet_lemmatizer.lemmatize(token) in self.wordvec_model.wv.vocab and self.wordnet_lemmatizer.lemmatize(token2) in self.wordvec_model.wv.vocab:
                        similarity = similarity + float(self.wordvec_model.wv.similarity(self.wordnet_lemmatizer.lemmatize(token), self.wordnet_lemmatizer.lemmatize(token2)))/len(tokens2)
                    else:
                        dist = edit_distance(token, token2)
                        if dist < 3:
                            similarity = similarity + float(1)/float(1 + math.pow(dist, 2))
                            #similarity = similarity + float(1)/(2+math.pow((float(edit_distance(token, token2))),4))
                sim_scores[x] = similarity
            top_similar = sorted([(k, v) for k, v in sim_scores.iteritems()], reverse=True, key=lambda x: x[1])[:num]
            for i, x in enumerate(top_similar):
                freq_scores[x[0]] = freq_scores[x[0]] + 1*math.pow(x[1],2)
        top = sorted([(k, v) for k, v in freq_scores.iteritems()], reverse=True, key=lambda x: x[1])[:num]
        return top

    def find_closest_syntactic(self, tokens, num, topic):
        """ Finds num syntactically closest candidates for a given topic"""
        freq_scores = {}
        for x in topic:
            freq_scores[x] = 0
        for token in tokens:
            sim_scores = {}
            for x in topic:
                tokens2 = x.lower().split(" ")
                distance = 0
                for token2 in tokens2:
                    distance = distance + float(edit_distance(token, token2))/len(tokens2)
                sim_scores[x] = distance
            top_similar = sorted([(k, v) for k, v in sim_scores.iteritems()], reverse=False, key=lambda x: x[1])[:num]
            for i, x in enumerate(top_similar):
                freq_scores[x[0]] = freq_scores[x[0]] + 1/float(math.pow((x[1]+1),2))
        top = sorted([(k, v) for k, v in freq_scores.iteritems()], reverse=True, key=lambda x: x[1])[:num]
        return top


    def lookup_google(self, params):
        """ Lookup in Google Search"""
        #curl "https://kgsearch.googleapis.com/v1/entities:search?query=bebe&key=<key>&limit=2&indent=True&types=Organization"
        url = self.google_service_url + '?' + urllib.urlencode(params)
        #result score = an indicator of how well the entity matched the request constraints.
        response = json.loads(urllib.urlopen(url).read())
        results = []
        if "itemListElement" in response:
            for element in response['itemListElement']:
                dict_result = {}
                if "resultScore" in element:
                    dict_result["resultScore"] = element['resultScore']
                if "result" in element:
                    if "detailedDescription" in element["result"]:
                        dict_result["detailedDescription"] = element["result"]['detailedDescription']
                    if "description" in element["result"]:
                        dict_result["description"] = element["result"]['description']
                    if "url" in element["result"]:
                        dict_result["url"] = element["result"]["url"]
                results.append(dict_result)
        return results

    def rank_google_result_company(self, results):
        """ Binary rank  of google search results"""
        for result in results:
            for keyword in self.brands_keywords_google:
                if "detailedDescription" in result:
                    if keyword in result["detailedDescription"]:
                        return 1
                if "description" in result:
                    if keyword in result["description"]:
                        return 1
        return 0

    def rank_google_result_material(self, results):
        """ Binary rank  of google search results"""
        for result in results:
            for keyword in self.materials_keywords_google:
                if keyword in result["detailedDescription"] or keyword in result["description"]:
                    return 1
        return 0

    def rank_probase_result_company(self, result):
        """Probase probability ranking [0,1]"""
        keywords = filter(lambda x: x in result, self.probase_brands)
        keywords = map(lambda x: result[x], keywords)
        if len(keywords) > 0:
            return max(keywords)
        else:
            return 0

    def rank_probase_result_material(self, result):
        """Probase probability ranking [0,1]"""
        keywords = filter(lambda x: x in result, self.probase_materials)
        keywords = map(lambda x: result[x], keywords)
        if len(keywords) > 0:
            return max(keywords)
        else:
            return 0

    def lookup_probase(self, params):
        """Probase lookup"""
        #curl "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=adidas&topK=10"
        url = self.probase_service_url + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        return response

    def get_liketoknowitlinks(self, tokens):
        """ Extract liketoknowit links"""
        links = []
        for token in tokens:
            match = re.search("http://liketk.it/([^\s]+)", token)
            if match is not None:
                link = match.group(0)
                links.append(link)
        return links

    def lda_topic_models(self, num_topics, num_iter, min_occ, docs):
        """ Extract LDA topic models """
        cvectorizer = CountVectorizer(min_df=min_occ, stop_words="english")
        cvz = cvectorizer.fit_transform(docs)
        lda_model = lda.LDA(n_topics=num_topics, n_iter=num_iter)
        X_topics = lda_model.fit_transform(cvz)
        _lda_keys = []
        for i in xrange(X_topics.shape[0]):
            _lda_keys.append(X_topics[i].argmax())
        topic_summaries = []
        topic_word = lda_model.topic_word_  # all topic words
        n_top_words = 5
        vocab = cvectorizer.get_feature_names()
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
            topic_summaries.append(' '.join(topic_words))
        return topic_summaries

    def get_top_num(self, coll, num):
        """ Extract top 10 ranked items"""
        top, counts = zip(*Counter(coll).most_common(num))
        return list(top)

    def get_wikipedia_vote(self, query):
        """ Wikipedia lookup binary rank"""
        pages = wikipedia.search(query)
        for pageName in pages:
            try:
                page = wikipedia.page(pageName)
                content = page.content.lower()
                for keyword in self.brands_keywords_google:
                    if keyword in content:
                        return 1
            except:
                return 0
        return 0

    def get_google_search_vote(self, query):
        """ Google search lookup binary rank"""
        try:
            response = GoogleSearch().search(query)
            for result in response.results:
                text = result.getText().lower()
                title  = result.title.lower()
                for keyword in self.brands_keywords_google:
                    if keyword in text or keyword in title:
                        return 1
        except:
            return 0
        return 0
