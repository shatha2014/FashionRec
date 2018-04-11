import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import pyspark
import json
import operator
from nltk.stem import WordNetLemmatizer
from snorkel import SnorkelSession
from snorkel.learning.gen_learning import GenerativeModel
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

"""
Script for preprocessing data to make it suitable for training the classifier 
"""

tknzr= TweetTokenizer(strip_handles=True, reduce_len=True)
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

path = "./data/small2.csv"
vectors = "./data/vectors2.vec"

labels_to_int = {}
labels_to_int["tops_and_tshirts"] = 1
labels_to_int["topsTshirts"] = 1
labels_to_int["bags"] = 2
labels_to_int["all_accessories"] = 3
labels_to_int["accessories"] = 3
labels_to_int["shoes"] = 4
labels_to_int["jeans"] = 5
labels_to_int["skirts"] = 6
labels_to_int["tichts_and_socks"] = 7
labels_to_int["tightsAndSocks"] = 7
labels_to_int["dresses"] = 8
labels_to_int["jackets"] = 9
labels_to_int["blouses_and_tunics"] = 10
labels_to_int["blouseAndTunics"] = 10
labels_to_int["trouser_and_shorts"] = 11
labels_to_int["trouserAndShorts"] = 11
labels_to_int["coats"] = 12
labels_to_int["jumpers_and_cardigans"] = 0
labels_to_int["jumperAndCartigens"] = 0

int_to_label = {}
int_to_label[1] = "tops_and_tshirts"
int_to_label[2] = "bags"
int_to_label[3] = "all_accessories"
int_to_label[4] = "shoes"
int_to_label[5] = "jeans"
int_to_label[6] = "skirts"
int_to_label[7] = "tichts_and_socks"
int_to_label[8] = "dresses"
int_to_label[9] = "jackets"
int_to_label[10] = "blouses_and_tunics"
int_to_label[11] = "trouser_and_shorts"
int_to_label[12] = "coats"
int_to_label[0] = "jumpers_and_cardigans"

wordnet_lemmatizer = WordNetLemmatizer()

def make_binary_labels(labels):
    """ Make binary labels from continous labels """
    binary_labels = {}
    for k, v in labels.iteritems():
        binary_labels[k] = 0
    maxK = 1
    maxScore = -1
    sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)
    for k in range(1, 5):
        top_k = sorted_labels[:k]
        rest = sorted_labels[k:len(sorted_labels)]
        # score = float(sum([pair[1] for pair in top_k])) / float(sum([pair[1] for pair in rest]))
        score = (
            sum([pair[1] for pair in top_k]) / float(len(top_k)) - sum([pair[1] for pair in rest]) / float(
                len(rest)))
        if score > maxScore:
            maxK = k
            maxScore = score
    for i in range(0, len(labels)):
        label = sorted_labels[i][0]
        if i <= maxK:
            binary_labels[label] = 1
        else:
            binary_labels[label] = 0
    return binary_labels

def testlabels_to_onehot(labels):
    """ One-hot encode labels """
    onehot = []
    for i in range(0, 13):
        onehot.append(0)
    for i in range(0, 13):
        if i in labels or str(i) in labels:
            onehot[i] = 1
        else:
            onehot[i] = 0
    return onehot

def dict_to_onehot(dict_labels):
    """ Convert dict to onehot encoding """
    lbls = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    for k,v in dict_labels.iteritems():
        lbls[labels_to_int[k]] = v
    return lbls


def votes_to_onehot(labels):
    """ One-hot encode labels """
    onehot = []
    for i in range(0, 13):
        onehot.append(0)
    for k,v in labels.iteritems():
        onehot[labels_to_int[k]] = v
    return onehot

def sparkConf():
    "Setup spark configuration, change this to run on cluster"
    # conf = pyspark.SparkConf().setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false")
    # return conf
    conf = pyspark.SparkConf().setAppName("fashion_rec_data_setup").setMaster("local[*]") \
        .setAppName("fashion_rec_data_setup") \
        .set("spark.hadoop.validateOutputSpecs", "false").set("spark.driver.memory", "5g")
    return conf


def pre_process_features(inputPaths, outputPath, multichannel):
    """ Process input data to correct feature data. Remove suplerflous data and save to csv"""
    ids = set()
    x_text = []
    for path in inputPaths:
        with open(path, 'r') as csvfile:
            for line in csvfile:
                line = line.strip("\n")
                parts = line.split(",")
                if len(parts) == 5:
                    comment = parts[2]
                    caption = parts[3]
                    tags = parts[4]
                    if parts[0] not in ids:
                        if(multichannel):
                            features = comment + "," + caption + " " + tags
                            x_text.append((parts[0], features))
                        else:
                            features = tags + " " + comment + " " + caption
                            x_text.append((parts[0], features))
                        ids.add(parts[0])
                if len(parts) == 4:
                    comment = parts[1]
                    caption = parts[2]
                    tags = parts[3]
                    if parts[0] not in ids:
                        if(multichannel):
                            features = comment + "," + caption + " " + tags
                            x_text.append((parts[0], features))
                        else:
                            features = tags + " " + comment + " " + caption
                            x_text.append((parts[0], features))
                        ids.add(parts[0])
    print("cleaned features len: {}".format(len(x_text)))
    with open(outputPath, 'w') as outfile:
        for entry in x_text:
            outfile.write(entry[0] + ",")
            outfile.write(entry[1] + "\n")

def combine_labels_features(featuresPath, labelsPath, testFeaturesPath, testLabelsPath, vectorsPath,
                            vectorDim, max_document_length, multichannel, generative):
    """
    Combine labels and features into train-set in correct format to feed into CNN, can use pre-trained embeddings or not and converts votes to
    noisy labels using generative model or majority voting
    """

    # Read in features
    features = {}
    with open(featuresPath, 'r') as csvfile:
        for line in csvfile:
            line = line.replace("\n", " ")
            parts = line.split(",")
            id = parts[0]
            if not multichannel:
                if len(parts) == 2:
                    text = parts[1]
                if len(parts) < 2:
                    text = ""
                features[id] = text
            if multichannel:
                comments = parts[1]
                caption = parts[2]
                features[id] = [comments, caption]

    # Read in labels
    labels = json.load(open(labelsPath))

    # Combine features and labels
    x_data = []
    y_data = []
    for k,v in labels.iteritems():
        x_data.append(features[k])
        y_data.append(labels[k])
    y_data = np.array(y_data)

    test_features = {}
    test_labels = {}
    test_ids = set()

    # Read in test features
    with open(testFeaturesPath, 'r') as csvfile:
        for line in csvfile:
            line = line.replace("\n", " ")
            parts = line.split(",")
            id = parts[0]
            if not multichannel:
                if len(parts) > 1:
                    text = parts[1]
                else:
                    text = ""
                test_features[id] = text
            if multichannel:
                comments = parts[1]
                caption = parts[2]
                test_features[id] = [comments, caption]

    # Read in test labels
    with open(testLabelsPath, 'r') as csvfile:
        for line in csvfile:
            parts = line.split(",")
            id = parts[0]
            indices = parts[1]
            test_labels[id] = indices.split(" ")
            test_ids.add(id)

    # Combine test labels and features
    x_test_data = []
    y_test_data = []
    for id in test_ids:
        x_test_data.append(test_features[id])
        y_test_data.append(testlabels_to_onehot(test_labels[id]))
    y_test_data = np.array(y_test_data)
    concat_x = np.array(x_data + x_test_data)

    # Setup vocabulary to convert text into list of indices
    if(multichannel):
        comment_x = concat_x[:,0]
        caption_x = concat_x[:,1]

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        concat_fit_comment = np.array(list(vocab_processor.fit_transform(comment_x)))
        concat_fit_caption = np.array(list(vocab_processor.fit_transform(caption_x)))
        x_comment = concat_fit_comment[0:len(x_data)]
        x_comment_test = concat_fit_comment[len(x_data):len(x_data) + len(x_test_data)]
        x_caption = concat_fit_caption[0:len(x_data)]
        x_caption_test = concat_fit_caption[len(x_data):len(x_data) + len(x_test_data)]
    else:
        # Convert text to numeric idx format
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        concat_fit = np.array(list(vocab_processor.fit_transform(concat_x)))
        x = concat_fit[0:len(x_data)]
        x_test = concat_fit[len(x_data):len(x_data) + len(x_test_data)]

    if multichannel:
        x = []
        for i in range(0, len(x_comment)):
            x.append([x_comment[i], x_caption[i]])
        x_test = []
        for i in range(0, len(x_comment_test)):
            x_test.append([x_comment_test[i], x_caption_test[i]])
        x = np.array(x)
        x_test = np.array(x_test)

    # If using pre-trained embeddings, load them and initialize an embedding matrix for all words in the vocabulary
    if(vectorsPath != ""):
        # Load pre-trained embeddings (depending on which gensim version you use)
        #model = gensim.models.KeyedVectors.load_word2vec_format(vectorsPath, binary=False)
        model = gensim.models.Word2Vec.load_word2vec_format(vectorsPath, binary=False)
        vocab_size = len(vocab_processor.vocabulary_)
        emb_size = 300
        embeddings = np.zeros((vocab_size, emb_size))

        # Lookup words in pretrained embeddings
        words = []
        if(multichannel):
            for entry in comment_x:
                words = words + entry.split(" ")
            for entry in caption_x:
                words = words + entry.split(" ")
        else:
            for entry in concat_x:
                words = words + entry.split(" ")
        # Initialize embeddings
        for w in set(words):
            if w in model.vocab:
                embeddings[vocab_processor.vocabulary_.get(w)] = model[w]
            else:
                embeddings[vocab_processor.vocabulary_.get(w)] = np.random.uniform(-0.25, 0.25, vectorDim)
        embeddings = embeddings.astype(np.float32)
        W = tf.constant(embeddings, name="W", dtype=tf.float32)

        if generative:
            # Convert vote-labels to noisy labels using generative model
            y_data_marginals = learn_generative(y_data)
        else:
            # Convert vote-labels to noisy labels using majority voting
            y_data_marginals = majority_vote(y_data)
        return x, vocab_processor, y_data_marginals, x_test, y_test_data, W
    else:
        if generative:
            # Convert vote-labels to noisy labels using generative model
            y_data_marginals = learn_generative(y_data)
        else:
            # Convert vote-labels to noisy labels using majority voting
            y_data_marginals = majority_vote(y_data)
        return x, vocab_processor, y_data_marginals, x_test, y_test_data

def majority_vote(y_data):
    """Take the majority vote of the votes as the truth label"""
    vote_data = []
    for ex in y_data:
        acc = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        for votes in ex:
            acc = np.add(acc, votes)
        for i in range(0, len(acc)):
            if(acc[i] < 0):
                acc[i] = 0
            else:
                acc[i] = 1
        vote_data.append(acc)
    return np.array(vote_data)

def learn_generative(y_data):
    """
    Uses Snorkel to learn a generative model of the relative accuracies of LFs.
    It learns one generative model for each class, and combines them into a set of noisy labels
    """
    labels = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for ex in y_data:
        for i in range(0,13):
            label_i = []
            for vote in ex:
                label_i.append(int(vote[i]))
            labels[i].append(np.array(label_i))
    labels = map(lambda x: np.array(x), labels)
    labels = np.array(labels)
    n_labels = []
    n_stats = []
    for i, class_lbl in enumerate(labels):
        print("learning generative model for label: {}".format(i))
        session = SnorkelSession()
        gen_model = GenerativeModel()
        gen_model.train(
            class_lbl,
            epochs=100, decay=0.95, step_size=0.1 / class_lbl.shape[0], reg_param=1e-6, cardinality=2
        )
        train_marginals = gen_model.marginals(csr_matrix(class_lbl))
        n_labels.append(train_marginals)
        n_stats.append(gen_model.learned_lf_stats())
    for i, stats in enumerate(n_stats):
        stats.to_csv("./results/lf_stats/" + int_to_label[i], sep=',', encoding='utf-8')
    return np.array(n_labels).T

def normalize_labels(labels):
    """ Normalize labels to probabilities"""
    s = sum(labels.values())
    normalized = {}
    for k, v in labels.iteritems():
        if s > 0:
            normalized[k] = v / s
        else:
            normalized[k] = v
    return normalized

def make_binary_labels2(labels):
    """ Make binary labels from continous labels """
    binary_labels = {}
    for k, v in labels.iteritems():
        binary_labels[k] = 0
    sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)
    top5labels = map(lambda x: x[0], sorted_labels)[0:4]
    total = sum([pair[1] for pair in sorted_labels])
    avg = (total)/float(len(sorted_labels))
    outp = {}
    ids = set()
    for category in labels_to_int.keys():
        if labels_to_int[category] not in ids:
            if category in labels:
                if labels[category] > avg and category in top5labels:
                    outp[category] = 1
                else:
                    outp[category] = 0
                ids.add(labels_to_int[category])
    for category in labels_to_int.keys():
        if labels_to_int[category] not in ids:
            outp[category] = 0
    return outp

def filter_dict(dict):
    """ Filter dicts based positive values """
    return {k: v for k, v in dict.iteritems() if v > 0}


def sum_dicts(dicts):
    """ Sum values in dicts"""
    summed = {}
    for d in dicts:
        for k, v in d.iteritems():
            if k in summed:
                summed[k] = summed[k] + v
            else:
                summed[k] = v
    return summed

def pre_process_labels(inputPath, outputPath):
    """
    Pre-process labels by normalizing and combining them into lists of votes
    """
    labels = json.load(open(inputPath))
    cleaned_labels = {}
    for item in labels:
        try:
            item = json.loads(item)
        except:
            item = item
        clarifai = normalize_labels(item["clarifai_classification"]["items"])
        deep_detect = normalize_labels(item["deep_detect_classification"]["items"])
        deepomatic = normalize_labels(item["deepomatic_classification"]["items"])
        google = normalize_labels(item["google_vision_classification"]["items"])
        text_clustering = normalize_labels(item["text_clustering"]["items"])
        lf1 = item["LF1"]
        lf2 = item["LF2"]
        combined = [
            votes_to_onehot(make_binary_labels(clarifai)),
            votes_to_onehot(make_binary_labels(deep_detect)),
            votes_to_onehot(make_binary_labels(deepomatic)),
            votes_to_onehot(make_binary_labels(google)),
            votes_to_onehot(make_binary_labels(text_clustering)),
            votes_to_onehot(lf1),
            votes_to_onehot(lf2),
        ]
        cleaned_labels[item["id"]] = combined
    with open(outputPath, 'w') as outfile:
        json.dump(cleaned_labels, outfile)

def combine_labels(inputPaths, outputPath):
    """ Combine labels stored in spark parallelized format into single file """
    sc = pyspark.SparkContext(conf=sparkConf())
    totalLabels = []
    for path in inputPaths:
        text_file = sc.textFile(path)
        text_file = text_file.repartition(1)
        labels = text_file.collect()
        totalLabels = totalLabels + labels
    with open(outputPath, 'w') as outfile:
        json.dump(totalLabels, outfile)


def combine_labels2(inputPaths, outputPath, textAnalysisPath):
    """ Combine labels stored in spark parallelized format into single file """
    text_analysis = json.load(open(textAnalysisPath))
    id_to_text = {}
    for t in text_analysis:
        text_clustering = {}
        text_clustering["items"] = t["item_category"]
        id_to_text[t["id"]] = text_clustering
    sc = pyspark.SparkContext(conf=sparkConf())
    totalLabels = []
    for path in inputPaths:
        text_file = sc.textFile(path)
        text_file = text_file.repartition(1)
        labels = text_file.collect()
        labels = map(lambda x: json.loads(x), labels)
        totalLabels = totalLabels + labels

    totalLabels = filter(lambda x: x["clothing"] == True, totalLabels)
    totalLabels2 = []
    for lbl in totalLabels:
        nlbl = {}
        nlbl["text_clustering"] = id_to_text[lbl["id"]]
        nlbl["id"] = lbl["id"]
        nlbl["clarifai_classification"] = lbl["clarifai_classification"]
        nlbl["deep_detect_classification"] = lbl["deep_detect_classification"]
        nlbl["deepomatic_classification"] = lbl["deepomatic_classification"]
        nlbl["google_vision_classification"] = lbl["google_vision_classification"]
        totalLabels2.append(nlbl)

    with open(outputPath, 'w') as outfile:
        jsonStr = json.dumps(totalLabels2, outfile)
        outfile.write(jsonStr)

def merge_totals(path1, path2, outputPath):
    """ Merge combined labels"""
    p1 = json.load(open(path1))
    p2 = json.load(open(path2))
    total = p1 + p2
    with open(outputPath, 'w') as outfile:
        jsonStr = json.dumps(total, outfile)
        outfile.write(jsonStr)

def split(x, y, sample_percentage, vocab):
    """ Split train data into train/dev"""

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]

    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, x_dev, y_train, y_dev

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def test_labels_to_csv(inputPath, outputPath):
    """ Convert json labels to CSV """
    labels = json.load(open(inputPath))
    cleaned_labels = []
    for item in labels:
        if (item["annotatorusername"] == "kim" or item["annotatorusername"] == "mallu" or item["annotatorusername"] == "shatha" or item["annotatorusername"] == "ummul") and item["imageinfo"][
            "annotated"] == True:
            id = item["imageid"]
            cat_labels = item["imageinfo"]["annotateddatajson"]
            cat_items = []
            for label in cat_labels:
                if label["ItemCategory"] != "Non Fashion  item":
                    cat_items.append(label["ItemCategory"])
            cleaned_labels.append((id, cat_items))
    with open(outputPath, 'w') as outfile:
        for label in cleaned_labels:
            if len(label[1]) > 0:
                outfile.write(label[0] + ",")
                for lbl in label[1]:
                    outfile.write(str(labels_to_int[lbl]) + " ")
                outfile.write("\n")

def keyword_labeling_funs(distant_supervision_labels_path, featuresPath, vectorsPath, outputPath):
    """ Apply basic LFs based on keywords matching on the train set """
    labels = json.load(open(distant_supervision_labels_path))
    features = {}
    #depending on which version of gensim you use
    #model = gensim.models.KeyedVectors.load_word2vec_format(vectorsPath, binary=False)
    model = gensim.models.Word2Vec.load_word2vec_format(vectorsPath, binary=False)
    idx = 0
    synsets_jumper = ["jumper", "cardigan", "hoodie", "sweatshirt", "jersey", "sweater", "pullover", "cardi"]
    synsets_tops = ["top", "tshirt", "polo", "vest", "sleeve", "shirt", "tee", "tank"]
    synsets_coats = ["coat","parka","trench"]
    synsets_dresses = ["dress"]
    synsets_jackets = ["jacket", "blazer", "cape"]
    synsets_jeans = ["jean", "jeans", "skinny", "denim"]
    synsets_skirts = ["skirt"]
    synsets_tights = ["tights", "sock"]
    synsets_trouser = ["trouser", "legging", "chino", "jogger", "sweat", "playsuit", "jumpsuit", "shorts"]
    synsets_shoes = ["shoe", "boot", "heel", "sandal", "slipper", "sneaker", "loafer", "espadrilles", "moccasin"]
    synsets_bags = ["bags", "bag", "handbag", "rucksack", "totebag", "shoulderbag", "carryall"]
    synsets_accessories = ["accessories", "belt", "glove", "hat", "cap", "jewellery", "watch", "purse", "scarve", "shawl", "sunglass", "beanie"]
    synsets_blouse = ["blouses", "blouse", "tunics", "tunic", "shirt"]
    synsets = [synsets_jumper, synsets_tops, synsets_coats, synsets_dresses, synsets_jackets, synsets_jeans, synsets_skirts, synsets_tights, synsets_trouser, synsets_shoes, synsets_bags, synsets_accessories, synsets_blouse]
    with open(featuresPath, 'r') as featuresFile:
        for line in featuresFile:
            if idx % 100 == 0:
                print("processing post: {}".format(idx))
            idx += 1
            parts = line.split(",")
            id = parts[0]
            text = parts[1]
            tokens = tknzr.tokenize(text)
            tokens = map(lambda x: wordnet_lemmatizer.lemmatize(x).encode("utf-8"), tokens)
            tokens = filter(lambda x: x not in stop_words, tokens)

            LF1_jumper = -1
            LF1_tops = -1
            LF1_coat = -1
            LF1_dress = -1
            LF1_jacket = -1
            LF1_jeans = -1
            LF1_skirt = -1
            LF1_tights = -1
            LF1_trouser = -1
            LF1_shoe = -1
            LF1_bag = -1
            LF1_accessories = -1
            LF1_blouse = -1
            LF2_jumper = -1
            LF2_tops = -1
            LF2_coat = -1
            LF2_dress = -1
            LF2_jacket = -1
            LF2_jeans = -1
            LF2_skirt = -1
            LF2_tights = -1
            LF2_trouser = -1
            LF2_shoe = -1
            LF2_bag = -1
            LF2_accessories = -1
            LF2_blouse = -1
            lfs1 = [LF1_jumper, LF1_tops, LF1_coat, LF1_dress, LF1_jacket, LF1_jeans, LF1_skirt, LF1_tights, LF1_trouser, LF1_shoe, LF1_bag, LF1_accessories, LF1_blouse]
            lfs2 = [LF2_jumper, LF2_tops, LF2_coat, LF2_dress, LF2_jacket, LF2_jeans, LF2_skirt, LF2_tights, LF2_trouser, LF2_shoe, LF2_bag, LF2_accessories, LF2_blouse]

            for i in range(0,len(synsets)):
                for w in synsets[i]:
                    if w in tokens:
                        lfs1[i] = 1

            for i in range(0,len(synsets)):
                for w in synsets[i]:
                    for w2 in tokens:
                        if w in model.wv.vocab and w2 in model.wv.vocab:
                            similarity = model.similarity(w, w2)
                        if similarity > 0.6:
                            lfs2[i] = 1
            res = {}
            res["LF1"] = {}
            res["LF2"] = {}

            res["LF1"]["jumpers_and_cardigans"] = lfs1[0]
            res["LF1"]["tops_and_tshirts"] = lfs1[1]
            res["LF1"]["coats"] = lfs1[2]
            res["LF1"]["dresses"] = lfs1[3]
            res["LF1"]["jackets"] = lfs1[4]
            res["LF1"]["jeans"] = lfs1[5]
            res["LF1"]["skirts"] = lfs1[6]
            res["LF1"]["tichts_and_socks"] = lfs1[7]
            res["LF1"]["trouser_and_shorts"] = lfs1[8]
            res["LF1"]["shoes"] = lfs1[9]
            res["LF1"]["bags"] = lfs1[10]
            res["LF1"]["all_accessories"] = lfs1[11]
            res["LF1"]["blouses_and_tunics"] = lfs1[12]

            res["LF2"]["jumpers_and_cardigans"] = lfs2[0]
            res["LF2"]["tops_and_tshirts"] = lfs2[1]
            res["LF2"]["coats"] = lfs2[2]
            res["LF2"]["dresses"] = lfs2[3]
            res["LF2"]["jackets"] = lfs2[4]
            res["LF2"]["jeans"] = lfs2[5]
            res["LF2"]["skirts"] = lfs2[6]
            res["LF2"]["tichts_and_socks"] = lfs2[7]
            res["LF2"]["trouser_and_shorts"] = lfs2[8]
            res["LF2"]["shoes"] = lfs2[9]
            res["LF2"]["bags"] = lfs2[10]
            res["LF2"]["all_accessories"] = lfs2[11]
            res["LF2"]["blouses_and_tunics"] = lfs2[12]
            features[id] = res
    new_labels = []
    print("len features: {}".format(len(features)))
    for item in labels:
        try:
            item = json.loads(item)
        except:
            item = item
            lfs = features[item["id"]]
            item["LF1"] = lfs["LF1"]
            item["LF2"] = lfs["LF2"]
            new_labels.append(item)

    with open(outputPath, 'w') as outfile:
        json.dump(new_labels, outfile)