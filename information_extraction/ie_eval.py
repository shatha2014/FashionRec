import json
from InformationExtraction import InformationExtractor
from wordsegment import load, segment
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import rankings_helper
import argparse
import numpy as np
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import logging
from scipy import stats

"""
Script for evaluating the IE system using NDGC@K, P@K, R@K, and MAP, also computes p-values. 

Requires ground-truth labels for the evaluation.  
"""

tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
tknzr_strip_users = TweetTokenizer(strip_handles=True, reduce_len=True)
load()
# seg_ig = Segmenter(corpus="ig_corpus")
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

# For compatibility between different formats
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


def parse_args():
    """Parses the commandline arguments with argparse"""
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-i", "--input", help="folder where input documents are", default="data/sample_user.csv")
    parser.add_argument("-im", "--imagepath", help="path to images, necessary if using google vision API", default="")
    parser.add_argument("-o", "--output", help="folder where output documents are", default="output")
    parser.add_argument("-c", "--conf", help="path to confFile", default="./conf/conf.json")
    parser.add_argument("-ma", "--materials", help="path to file with clothing materials/fabrics",
                        default="./domain_data/material.csv")
    parser.add_argument("-it", "--items", help="path to file with sub-categories of clothing items",
                        default="./domain_data/items.csv")
    parser.add_argument("-st", "--styles", help="path to file with clothing styles",
                        default="./domain_data/styles.csv")
    parser.add_argument("-br", "--brands", help="path to file with clothing brands",
                        default="./domain_data/companies.csv")
    parser.add_argument("-pat", "--patterns", help="path to file with clothing patterns",
                        default="./domain_data/patterns.csv")
    parser.add_argument("-itc", "--itemtopcategory", help="path to file with top-categories of items",
                        default="./domain_data/item_top_category.csv")
    parser.add_argument("-pbr", "--probasebrands", help="path to file with probase categories to match with brands",
                        default="./domain_data/probase_brands.csv")
    parser.add_argument("-pma", "--probasematerials",
                        help="path to file with probase categories to match with materials/fabrics",
                        default="./domain_data/probase_materials.csv")
    parser.add_argument("-vec", "--vectors", help="path to file with word vectors",
                        default="./vectors/vectors.vec")
    parser.add_argument("-testvectors", "--testvectors", help="conf file with list of vectors to evaluate",
                        default="./conf/vectors.json")
    parser.add_argument("-lab", "--labels", help="file location of labels",
                        default="./eval/annotations.json")

    args = parser.parse_args()
    return args


def annotations_to_csv(inputPath, outputPath):
    """Convert the annotations exported in JSON to CSV that is suitable for processing"""
    labels = json.load(open(inputPath))
    cleaned_labels = []
    for item in labels:
        if item["annotatorusername"] == "kim" and item["imageinfo"]["annotated"] == True:
            # item = json.loads(item)
            id = item["imageid"]
            styles = item["imageinfo"]["styles"]
            cat_labels = item["imageinfo"]["annotateddatajson"]
            cat_items = []
            subcat_items = []
            patterns = []
            materials = []
            brands = []
            for label in cat_labels:
                if label["ItemCategory"] != "Non Fashion  item":
                    cat_items.append(label["ItemCategory"])
                    if "ItemSubCategory" in label:
                        subcat_items.append(label["ItemSubCategory"])
                    if "FinalizeAnnotatedAttributes" in label:
                        label_details = label["FinalizeAnnotatedAttributes"]
                        if "Pattern" in label_details:
                            for p in label_details["Pattern"]:
                                patterns.append(p)
                        if "Material" in label_details:
                            for m in label_details["Material"]:
                                materials.append(m)
                        if "Brand" in label_details:
                            for b in label_details["Brand"]:
                                brands.append(b)
            cleaned_labels.append((id, cat_items, subcat_items, patterns, materials, styles, brands))
    with open(outputPath, 'w') as outfile:
        outfile.write("id,item-category,item-sub-category,pattern,material,style,brand")
        for label in cleaned_labels:
            if len(label[1]) > 0:
                outfile.write(label[0] + ",")
                for lbl in label[1]:
                    outfile.write(lbl + " ")
                outfile.write(",")
                for lbl in label[2]:
                    outfile.write(lbl + " ")
                outfile.write(",")
                for lbl in label[3]:
                    outfile.write(lbl + " ")
                outfile.write(",")
                for lbl in label[4]:
                    outfile.write(lbl + " ")
                outfile.write(",")
                for lbl in label[5]:
                    outfile.write(lbl + " ")
                outfile.write(",")
                for lbl in label[6]:
                    outfile.write(lbl + " ")
                outfile.write("\n")


def get_hashtags(tokens):
    """Extract hashtags from a set of tokens"""
    hashtags = [x for x in tokens if x.startswith("#")]
    return hashtags


def get_userhandles(tokens):
    """ Extract userhandles from a set of tokens """
    userhandles = [x for x in tokens if x.startswith("@")]
    return userhandles


def create_vocab(docs):
    """Create a vocabulary for a given set of documents"""
    words = set()
    for d in docs:
        for w in d:
            words.add(w)
    vocab = {}
    for i, w in enumerate(list(words)):
        vocab[w] = i
    return vocab


def create_tf_idf(x, numX):
    """
    Compute the TF-IDF scores based on the entire corpus
    but only return the first numX documents, as those have labels to be used for evaluation
    """
    dct = Dictionary(x)
    corpus = [dct.doc2bow(line) for line in x]
    model = TfidfModel(corpus)
    tfidf_factors = {}
    for i, doc in enumerate(corpus):
        temp = {}
        for id, value in model[doc]:
            word = dct.get(id)
            temp[word] = value
        tfidf_factors[i] = temp
    tfidf_factors_subset = {}
    for i in range(0, numX):
        tfidf_factors_subset[i] = tfidf_factors[i]
    return tfidf_factors_subset


def clean_text(text):
    """ Clean the text, lowercase, remove noise, lemmatize """
    text = text.decode('utf-8', 'ignore').encode("utf-8")
    list_of_words = [i.lower() for i in tknzr.tokenize(text.decode("utf-8")) if
                     i.lower() not in stop_words]  # remove stopwords & tokenize (preserve hashtags)
    list_of_words = filter(lambda x: "pic.twitter" not in x and not "twitter.com" in x, list_of_words)
    list_of_words = filter(lambda x: "http" not in x, list_of_words)
    list_of_words = map(lambda x: wordnet_lemmatizer.lemmatize(x), list_of_words)  # lemmatize
    list_of_words = map(lambda x: x.encode("utf-8"), list_of_words)
    return list_of_words


def extract_features(labelsPath, featuresPath):
    """
    Match features from the larger corpus to the corresponding
    labels in the smaller annotated dataset
     """
    features = {}
    labels = {}
    ids = set()
    with open(featuresPath, 'r') as csvfile:
        for line in csvfile:
            line = line.replace("\n", " ")
            parts = line.split(",")
            id = parts[0]
            comments = parts[1]
            if comments is not None:
                comments = clean_text(comments)
            else:
                comments = []
            caption = parts[2]
            if caption is not None:
                caption = clean_text(caption)
            else:
                caption = []
            tags = parts[3]
            if tags is not None:
                tags = clean_text(tags)
            else:
                tags = []
            features[id] = (comments, caption, tags, id)

    with open(labelsPath, 'r') as csvfile:
        firstLine = False
        for line in csvfile:
            if firstLine:
                parts = line.split(",")
                id = parts[0]
                itemcat = parts[1]
                itemsubcat = parts[2]
                pattern = parts[3]
                material = parts[4]
                style = parts[5]
                brand = parts[6]
                labels[id] = (id, itemcat, itemsubcat, pattern, material, style, brand)
                ids.add(id)
            else:
                firstLine = True
    x_data = []
    x_data_all = []
    y_data = []
    # for id in list(ids)[0:2]:
    for id in ids:
        e = features[id]
        comments = e[0]
        caption = e[1]
        tags = e[2]
        id = e[3]
        hashtags = get_hashtags(comments + caption + tags)
        userhandles = get_userhandles(comments + caption + tags)
        segmented_hashtags = []
        for token in hashtags:
            if len(token) < 20:
                seg = segment(token.strip("#"))
                # seg = seg_ig.segment(token.strip("#"))
                segmented_hashtags.extend(seg)
        comments = tknzr_strip_users.tokenize(" ".join(comments))
        caption = tknzr_strip_users.tokenize(" ".join(caption))
        tags = tknzr_strip_users.tokenize(" ".join(tags))
        x_data.append((list(set(comments)), list(set(caption)), list(set(tags)), list(set(segmented_hashtags)),
                       list(set(hashtags)), list(set(userhandles)), id))
        x_data_all.append(comments + caption + tags + segmented_hashtags + hashtags)
        y_data.append(labels[id])
    for k, v in features.iteritems():
        if k not in ids:
            comments = v[0]
            caption = v[1]
            tags = v[2]
            comments = tknzr_strip_users.tokenize(" ".join(comments))
            caption = tknzr_strip_users.tokenize(" ".join(caption))
            tags = tknzr_strip_users.tokenize(" ".join(tags))
            hashtags = get_hashtags(comments + caption + tags)
            segmented_hashtags = []
            for token in hashtags:
                if len(token) < 20:
                    seg = segment(token.strip("#"))
                    # seg = seg_ig.segment(token.strip("#"))
                    segmented_hashtags.extend(seg)
            x_data_all.append(comments + caption + tags + segmented_hashtags + hashtags)
    return x_data, y_data, x_data_all


def semantic_clustering_predict(features, information_extractor):
    """ Do predictions for the evaluation using semantic clustering """
    predictions = []
    predictions_wo_probase = []
    for i, post in enumerate(features):
        print("Processing post: {}".format(i))
        comments = post[0]
        caption = post[1]
        tags = post[2]
        segmented_hashtags = post[3]
        hashtags = post[4]
        userhandles = post[5]
        id = post[6]
        preds, pred_wo_probase = text_clustering_LF(caption, comments, segmented_hashtags, tags, hashtags, userhandles,
                                                    information_extractor, i)
        preds["caption"] = caption
        preds["comments"] = comments
        preds["segmented_hashtags"] = segmented_hashtags
        preds["tags"] = tags
        preds["hashtags"] = hashtags
        preds["userhandles"] = userhandles
        pred_wo_probase["caption"] = caption
        pred_wo_probase["comments"] = comments
        pred_wo_probase["segmented_hashtags"] = segmented_hashtags
        pred_wo_probase["tags"] = tags
        pred_wo_probase["hashtags"] = hashtags
        pred_wo_probase["userhandles"] = userhandles
        predictions.append(preds)
        predictions_wo_probase.append(pred_wo_probase)

    return predictions, predictions_wo_probase


def syntactic_clustering_predict(features, information_extractor):
    """ Do predictions for the evaluation using syntactic clustering """
    predictions = []
    for i, post in enumerate(features):
        print("Processing post: {}".format(i))
        comments = post[0]
        caption = post[1]
        tags = post[2]
        segmented_hashtags = post[3]
        hashtags = post[4]
        userhandles = post[5]
        id = post[6]
        preds = text_clustering_LF_syntactic(caption, comments, segmented_hashtags, tags, hashtags, userhandles,
                                             information_extractor, i)
        predictions.append(preds)
    return predictions


def eval(predictions, labels):
    """"
    Evaluate results with precision@K, MAP, NDGC@K etc..
    """
    total_item_avg_precision = []
    total_item_precision_at_k = []
    total_item_ndgc_score_at_k1 = []
    total_item_dgc_score_at_k1 = []
    total_item_ndgc_score_at_k2 = []
    total_item_dgc_score_at_k2 = []
    total_item_precision_at_k2 = []
    total_item_map_score = []
    total_item_r_precision = []
    total_item_mean_reciprocal_rank = []

    total_style_avg_precision = []
    total_style_precision_at_k = []
    total_style_ndgc_score_at_k1 = []
    total_style_dgc_score_at_k1 = []
    total_style_ndgc_score_at_k2 = []
    total_style_dgc_score_at_k2 = []
    total_style_precision_at_k2 = []
    total_style_map_score = []
    total_style_r_precision = []
    total_style_mean_reciprocal_rank = []

    total_material_avg_precision = []
    total_material_precision_at_k = []
    total_material_ndgc_score_at_k1 = []
    total_material_dgc_score_at_k1 = []
    total_material_ndgc_score_at_k2 = []
    total_material_dgc_score_at_k2 = []
    total_material_precision_at_k2 = []
    total_material_map_score = []
    total_material_r_precision = []
    total_material_mean_reciprocal_rank = []

    total_pattern_avg_precision = []
    total_pattern_precision_at_k = []
    total_pattern_ndgc_score_at_k1 = []
    total_pattern_dgc_score_at_k1 = []
    total_pattern_ndgc_score_at_k2 = []
    total_pattern_dgc_score_at_k2 = []
    total_pattern_precision_at_k2 = []
    total_pattern_map_score = []
    total_pattern_r_precision = []
    total_pattern_mean_reciprocal_rank = []

    total_brand_avg_precision = []
    total_brand_precision_at_k = []
    total_brand_ndgc_score_at_k1 = []
    total_brand_dgc_score_at_k1 = []
    total_brand_ndgc_score_at_k2 = []
    total_brand_dgc_score_at_k2 = []
    total_brand_precision_at_k2 = []
    total_brand_map_score = []
    total_brand_r_precision = []
    total_brand_mean_reciprocal_rank = []

    for i in range(0, len(labels)):
        label = labels[i]
        pred = predictions[i]

        ## Items
        predictions_ranks_items = []
        labels_ranks_items = []
        pred_items = pred["item-category"]
        pred_items_sum = 0
        pred_items_int = {}
        for k, v in pred_items.iteritems():
            pred_items_int[labels_to_int[k]] = v
            pred_items_sum += v
        lbl_items = label[1].split(" ")
        lbl_items = filter(lambda x: x != " " and x != "", lbl_items)
        lbl_items = map(lambda x: labels_to_int[x], lbl_items)
        lbl_items = list(set(lbl_items))
        lbl_items_size = len(lbl_items)
        lbl_ranks_items_binary = []
        for entry in sorted(pred_items_int.items(), reverse=True, key=lambda x: x[1]):
            predictions_ranks_items.append(entry[1] / float(pred_items_sum))
            if entry[0] in lbl_items:
                labels_ranks_items.append(1.0 / float(lbl_items_size))
                lbl_ranks_items_binary.append(1)
                lbl_items.remove(entry[0])
            else:
                labels_ranks_items.append(0.0)
                lbl_ranks_items_binary.append(0)
        for lbl in lbl_items:
            labels_ranks_items.append(1.0 / float(lbl_items_size))
            predictions_ranks_items.append(0.0)

        if not (len(set(labels_ranks_items)) < 2 and (labels_ranks_items[0] == 0 or labels_ranks_items[0] == 0.0)):
            total_item_avg_precision.append(rankings_helper.average_precision_score(labels_ranks_items,
                                                                                    predictions_ranks_items,
                                                                                    len(predictions_ranks_items)))
            total_item_avg_precision.append(rankings_helper.average_precision(lbl_ranks_items_binary))
            total_item_map_score.append(rankings_helper.mean_average_precision([lbl_ranks_items_binary]))
            total_item_mean_reciprocal_rank.append(rankings_helper.mean_reciprocal_rank(lbl_ranks_items_binary))
            total_item_r_precision.append(rankings_helper.r_precision(lbl_ranks_items_binary))

            items_precision_at_k = []
            items_precision_at_k2 = []
            ndgc_at_k1 = []
            ndgc_at_k2 = []
            dgc_at_k1 = []
            dgc_at_k2 = []
            for i in range(1, 11):
                items_precision_at_k.append(
                    rankings_helper.ranking_precision_score(labels_ranks_items, predictions_ranks_items, i))
                items_precision_at_k2.append(rankings_helper.precision_at_k(lbl_ranks_items_binary, i))
                ndgc_at_k1.append(
                    rankings_helper.ndcg_score(labels_ranks_items, predictions_ranks_items, i, "exponential"))
                dgc_at_k1.append(
                    rankings_helper.dcg_score(labels_ranks_items, predictions_ranks_items, i, "exponential"))
                ndgc_at_k2.append(rankings_helper.ndcg_at_k(labels_ranks_items, i))
                dgc_at_k2.append(rankings_helper.dcg_at_k(labels_ranks_items, i))

            total_item_precision_at_k.append(items_precision_at_k)
            total_item_precision_at_k2.append(items_precision_at_k2)
            total_item_ndgc_score_at_k1.append(ndgc_at_k1)
            total_item_ndgc_score_at_k2.append(ndgc_at_k2)
            total_item_dgc_score_at_k1.append(dgc_at_k1)
            total_item_dgc_score_at_k2.append(dgc_at_k2)

        ## Styles
        predictions_ranks_styles = []
        labels_ranks_styles = []
        pred_styles = pred["styles"]
        pred_styles_sum = 0
        pred_styles_temp = {}
        for k, v in pred_styles.iteritems():
            pred_styles_sum += v
            pred_styles_temp[k.lower()] = v
        pred_styles = pred_styles_temp
        lbl_styles = [label[5].lower().strip()]
        lbl_styles = list(set(lbl_styles))
        lbl_styles_size = len(lbl_styles)
        lbl_ranks_styles_binary = []
        for entry in sorted(pred_styles.items(), reverse=True, key=lambda x: x[1]):
            predictions_ranks_styles.append(entry[1] / float(pred_styles_sum))
            if entry[0] in lbl_styles:
                labels_ranks_styles.append(1.0 / float(lbl_styles_size))
                lbl_ranks_styles_binary.append(1)
                lbl_styles.remove(entry[0])
            else:
                labels_ranks_styles.append(0.0)
                lbl_ranks_styles_binary.append(0)
        for lbl in lbl_styles:
            labels_ranks_styles.append(1.0 / float(lbl_styles_size))
            predictions_ranks_styles.append(0.0)
            lbl_ranks_styles_binary.append(1)

        if not (len(set(labels_ranks_styles)) < 2 and (labels_ranks_styles[0] == 0 or labels_ranks_styles[0] == 0.0)):
            total_style_avg_precision.append(rankings_helper.average_precision_score(labels_ranks_styles,
                                                                                     predictions_ranks_styles,
                                                                                     len(predictions_ranks_styles)))
            total_style_avg_precision.append(rankings_helper.average_precision(lbl_ranks_styles_binary))
            total_style_map_score.append(rankings_helper.mean_average_precision([lbl_ranks_styles_binary]))
            total_style_mean_reciprocal_rank.append(rankings_helper.mean_reciprocal_rank(lbl_ranks_styles_binary))
            total_style_r_precision.append(rankings_helper.r_precision(lbl_ranks_styles_binary))

            styles_precision_at_k = []
            styles_precision_at_k2 = []
            ndgc_at_k1 = []
            ndgc_at_k2 = []
            dgc_at_k1 = []
            dgc_at_k2 = []
            for i in range(1, 11):
                styles_precision_at_k.append(
                    rankings_helper.ranking_precision_score(labels_ranks_styles, predictions_ranks_styles, i))
                styles_precision_at_k2.append(rankings_helper.precision_at_k(lbl_ranks_styles_binary, i))
                ndgc_at_k1.append(
                    rankings_helper.ndcg_score(labels_ranks_styles, predictions_ranks_styles, i, "exponential"))
                dgc_at_k1.append(
                    rankings_helper.dcg_score(labels_ranks_styles, predictions_ranks_styles, i, "exponential"))
                ndgc_at_k2.append(rankings_helper.ndcg_at_k(labels_ranks_styles, i))
                dgc_at_k2.append(rankings_helper.dcg_at_k(labels_ranks_styles, i))

            total_style_precision_at_k.append(styles_precision_at_k)
            total_style_precision_at_k2.append(styles_precision_at_k2)
            total_style_ndgc_score_at_k1.append(ndgc_at_k1)
            total_style_ndgc_score_at_k2.append(ndgc_at_k2)
            total_style_dgc_score_at_k1.append(dgc_at_k1)
            total_style_dgc_score_at_k2.append(dgc_at_k2)

        ## Materials
        predictions_ranks_materials = []
        labels_ranks_materials = []
        pred_materials = pred["materials"]
        pred_materials_sum = 0
        pred_materials_temp = {}
        for k, v in pred_materials.iteritems():
            pred_materials_sum += v
            pred_materials_temp[k.lower()] = v
        pred_materials = pred_materials_temp
        lbl_materials = label[4].lower().strip().split(" ")
        lbl_materials = filter(lambda x: x != " " and x != "" and x != "\n", lbl_materials)
        # lbl_materials = map(lambda x: wordnet_lemmatizer.lemmatize(x))
        lbl_materials = list(set(lbl_materials))
        lbl_materials_size = len(lbl_materials)
        lbl_ranks_materials_binary = []
        for entry in sorted(pred_materials.items(), reverse=True, key=lambda x: x[1]):
            predictions_ranks_materials.append(entry[1] / float(pred_materials_sum))
            if entry[0] in lbl_materials:
                labels_ranks_materials.append(1.0 / float(lbl_materials_size))
                lbl_ranks_materials_binary.append(1)
                lbl_materials.remove(entry[0])
            else:
                labels_ranks_materials.append(0.0)
                lbl_ranks_materials_binary.append(0)
        for lbl in lbl_materials:
            labels_ranks_materials.append(1.0 / float(lbl_materials_size))
            predictions_ranks_materials.append(0.0)
            lbl_ranks_materials_binary.append(1)

        if not (len(set(labels_ranks_materials)) < 2 and (
                labels_ranks_materials[0] == 0 or labels_ranks_materials[0] == 0.0)):
            total_material_avg_precision.append(rankings_helper.average_precision_score(labels_ranks_materials,
                                                                                        predictions_ranks_materials,
                                                                                        len(
                                                                                            predictions_ranks_materials)))
            total_material_avg_precision.append(rankings_helper.average_precision(lbl_ranks_materials_binary))
            total_material_map_score.append(rankings_helper.mean_average_precision([lbl_ranks_materials_binary]))
            total_material_mean_reciprocal_rank.append(rankings_helper.mean_reciprocal_rank(lbl_ranks_materials_binary))
            total_material_r_precision.append(rankings_helper.r_precision(lbl_ranks_materials_binary))

            materials_precision_at_k = []
            materials_precision_at_k2 = []
            ndgc_at_k1 = []
            ndgc_at_k2 = []
            dgc_at_k1 = []
            dgc_at_k2 = []
            for i in range(1, 11):
                materials_precision_at_k.append(
                    rankings_helper.ranking_precision_score(labels_ranks_materials, predictions_ranks_materials, i))
                materials_precision_at_k2.append(rankings_helper.precision_at_k(lbl_ranks_materials_binary, i))
                ndgc_at_k1.append(
                    rankings_helper.ndcg_score(labels_ranks_materials, predictions_ranks_materials, i, "exponential"))
                dgc_at_k1.append(
                    rankings_helper.dcg_score(labels_ranks_materials, predictions_ranks_materials, i, "exponential"))
                ndgc_at_k2.append(rankings_helper.ndcg_at_k(labels_ranks_materials, i))
                dgc_at_k2.append(rankings_helper.dcg_at_k(labels_ranks_materials, i))

            total_material_precision_at_k.append(materials_precision_at_k)
            total_material_precision_at_k2.append(materials_precision_at_k2)
            total_material_ndgc_score_at_k1.append(ndgc_at_k1)
            total_material_ndgc_score_at_k2.append(ndgc_at_k2)
            total_material_dgc_score_at_k1.append(dgc_at_k1)
            total_material_dgc_score_at_k2.append(dgc_at_k2)

        ## Patterns
        predictions_ranks_patterns = []
        labels_ranks_patterns = []
        pred_patterns = pred["patterns"]
        pred_patterns_sum = 0
        pred_patterns_temp = {}
        for k, v in pred_patterns.iteritems():
            pred_patterns_sum += v
            pred_patterns_temp[
                k.lower().replace("animal print", "animal-print").replace("colour gradient", "colour-gradient").replace(
                    "polka dot", "polka-dot")] = v
        pred_patterns = pred_patterns_temp
        lbl_patterns = label[3].lower().strip()
        lbl_patterns = lbl_patterns.replace("animal print", "animal-print")
        lbl_patterns = lbl_patterns.replace("colour gradient", "colour-gradient")
        lbl_patterns = lbl_patterns.replace("polka dot", "polka-dot")
        lbl_patterns = lbl_patterns.split(" ")
        lbl_patterns = filter(lambda x: x != " " and x != "" and x != "\n", lbl_patterns)
        lbl_patterns_size = len(lbl_patterns)
        lbl_ranks_patterns_binary = []
        lbl_patterns = list(set(lbl_patterns))
        for entry in sorted(pred_patterns.items(), reverse=True, key=lambda x: x[1]):
            predictions_ranks_patterns.append(entry[1] / float(pred_patterns_sum))
            if entry[0] in lbl_patterns:
                labels_ranks_patterns.append(1.0 / float(lbl_patterns_size))
                lbl_ranks_patterns_binary.append(1)
                lbl_patterns.remove(entry[0])
            else:
                labels_ranks_patterns.append(0.0)
                lbl_ranks_patterns_binary.append(0)
        for lbl in lbl_patterns:
            labels_ranks_patterns.append(1.0 / float(lbl_patterns_size))
            predictions_ranks_patterns.append(0.0)
            lbl_ranks_patterns_binary.append(1)

        if not (len(set(labels_ranks_patterns)) < 2 and (
                labels_ranks_patterns[0] == 0 or labels_ranks_patterns[0] == 0.0)):
            total_pattern_avg_precision.append(rankings_helper.average_precision_score(labels_ranks_patterns,
                                                                                       predictions_ranks_patterns,
                                                                                       len(predictions_ranks_patterns)))
            total_pattern_avg_precision.append(rankings_helper.average_precision(lbl_ranks_patterns_binary))
            total_pattern_map_score.append(rankings_helper.mean_average_precision([lbl_ranks_patterns_binary]))
            total_pattern_mean_reciprocal_rank.append(rankings_helper.mean_reciprocal_rank(lbl_ranks_patterns_binary))
            total_pattern_r_precision.append(rankings_helper.r_precision(lbl_ranks_patterns_binary))

            patterns_precision_at_k = []
            patterns_precision_at_k2 = []
            ndgc_at_k1 = []
            ndgc_at_k2 = []
            dgc_at_k1 = []
            dgc_at_k2 = []
            for i in range(1, 11):
                patterns_precision_at_k.append(
                    rankings_helper.ranking_precision_score(labels_ranks_patterns, predictions_ranks_patterns, i))
                patterns_precision_at_k2.append(rankings_helper.precision_at_k(lbl_ranks_patterns_binary, i))
                ndgc_at_k1.append(
                    rankings_helper.ndcg_score(labels_ranks_patterns, predictions_ranks_patterns, i, "exponential"))
                dgc_at_k1.append(
                    rankings_helper.dcg_score(labels_ranks_patterns, predictions_ranks_patterns, i, "exponential"))
                ndgc_at_k2.append(rankings_helper.ndcg_at_k(labels_ranks_patterns, i))
                dgc_at_k2.append(rankings_helper.dcg_at_k(labels_ranks_patterns, i))

            total_pattern_precision_at_k.append(patterns_precision_at_k)
            total_pattern_precision_at_k2.append(patterns_precision_at_k2)
            total_pattern_ndgc_score_at_k1.append(ndgc_at_k1)
            total_pattern_ndgc_score_at_k2.append(ndgc_at_k2)
            total_pattern_dgc_score_at_k1.append(dgc_at_k1)
            total_pattern_dgc_score_at_k2.append(dgc_at_k2)

        ## Brands
        predictions_ranks_brands = []
        labels_ranks_brands = []
        pred_brands = pred["brands"]
        pred_brands_sum = 0
        pred_brands_temp = {}
        for k, v in pred_brands.iteritems():
            pred_brands_sum += v
            pred_brands_temp[
                k.lower().replace("animal print", "animal-print").replace("colour gradient", "colour-gradient").replace(
                    "polka dot", "polka-dot")] = v
        pred_brands = pred_brands_temp
        lbl_brands = label[6].lower().strip()
        lbl_brands = lbl_brands.replace("animal print", "animal-print")
        lbl_brands = lbl_brands.replace("colour gradient", "colour-gradient")
        lbl_brands = lbl_brands.replace("polka dot", "polka-dot")
        lbl_brands = lbl_brands.split(" ")
        lbl_brands = filter(lambda x: x != " " and x != "" and x != "\n", lbl_brands)
        lbl_brands = list(set(lbl_brands))
        lbl_brands_size = len(lbl_brands)
        lbl_ranks_brands_binary = []
        for entry in sorted(pred_brands.items(), reverse=True, key=lambda x: x[1]):
            predictions_ranks_brands.append(entry[1] / float(pred_brands_sum))
            if entry[0] in lbl_brands:
                labels_ranks_brands.append(1.0 / float(lbl_brands_size))
                lbl_ranks_brands_binary.append(1)
                lbl_brands.remove(entry[0])
            else:
                labels_ranks_brands.append(0.0)
                lbl_ranks_brands_binary.append(0)
        for lbl in lbl_brands:
            labels_ranks_brands.append(1.0 / float(lbl_brands_size))
            predictions_ranks_brands.append(0.0)
            lbl_ranks_brands_binary.append(1)

        if not (len(set(labels_ranks_brands)) < 2 and (labels_ranks_brands[0] == 0 or labels_ranks_brands[0] == 0.0)):
            total_brand_avg_precision.append(rankings_helper.average_precision_score(labels_ranks_brands,
                                                                                     predictions_ranks_brands,
                                                                                     len(predictions_ranks_brands)))
            total_brand_avg_precision.append(rankings_helper.average_precision(lbl_ranks_brands_binary))
            total_brand_map_score.append(rankings_helper.mean_average_precision([lbl_ranks_brands_binary]))
            total_brand_mean_reciprocal_rank.append(rankings_helper.mean_reciprocal_rank(lbl_ranks_brands_binary))
            total_brand_r_precision.append(rankings_helper.r_precision(lbl_ranks_brands_binary))

            brands_precision_at_k = []
            brands_precision_at_k2 = []
            ndgc_at_k1 = []
            ndgc_at_k2 = []
            dgc_at_k1 = []
            dgc_at_k2 = []
            for i in range(1, 11):
                brands_precision_at_k.append(
                    rankings_helper.ranking_precision_score(labels_ranks_brands, predictions_ranks_brands, i))
                brands_precision_at_k2.append(rankings_helper.precision_at_k(lbl_ranks_brands_binary, i))
                ndgc_at_k1.append(
                    rankings_helper.ndcg_score(labels_ranks_brands, predictions_ranks_brands, i, "exponential"))
                dgc_at_k1.append(
                    rankings_helper.dcg_score(labels_ranks_brands, predictions_ranks_brands, i, "exponential"))
                ndgc_at_k2.append(rankings_helper.ndcg_at_k(labels_ranks_brands, i))
                dgc_at_k2.append(rankings_helper.dcg_at_k(labels_ranks_brands, i))

            total_brand_precision_at_k.append(brands_precision_at_k)
            total_brand_precision_at_k2.append(brands_precision_at_k2)
            total_brand_ndgc_score_at_k1.append(ndgc_at_k1)
            total_brand_ndgc_score_at_k2.append(ndgc_at_k2)
            total_brand_dgc_score_at_k1.append(dgc_at_k1)
            total_brand_dgc_score_at_k2.append(dgc_at_k2)

    result = {}
    result["sample_item_avg_precision"] = total_item_avg_precision
    result["sample_item_precision_at_k"] = total_item_precision_at_k
    result["sample_item_ndgc_score_at_k1"] = total_item_ndgc_score_at_k1
    result["sample_item_dgc_score_at_k1"] = total_item_dgc_score_at_k1
    result["sample_item_ndgc_score_at_k2"] = total_item_ndgc_score_at_k2
    result["sample_item_dgc_score_at_k2"] = total_item_dgc_score_at_k2
    result["sample_item_precision_at_k2"] = total_item_precision_at_k2
    result["sample_item_map_score"] = total_item_map_score
    result["sample_item_r_precision"] = total_item_r_precision
    result["sample_item_mean_reciprocal_rank"] = total_item_mean_reciprocal_rank

    total_item_avg_precision = np.mean(total_item_avg_precision).tolist()
    total_item_precision_at_k = np.mean(total_item_precision_at_k, axis=0).tolist()
    total_item_ndgc_score_at_k1 = np.mean(total_item_ndgc_score_at_k1, axis=0).tolist()
    total_item_dgc_score_at_k1 = np.mean(total_item_dgc_score_at_k1, axis=0).tolist()
    total_item_ndgc_score_at_k2 = np.mean(total_item_ndgc_score_at_k2, axis=0).tolist()
    total_item_dgc_score_at_k2 = np.mean(total_item_dgc_score_at_k2, axis=0).tolist()
    total_item_precision_at_k2 = np.mean(total_item_precision_at_k2, axis=0).tolist()
    total_item_map_score = np.mean(total_item_map_score).tolist()
    total_item_r_precision = np.mean(total_item_r_precision).tolist()
    total_item_mean_reciprocal_rank = np.mean(total_item_mean_reciprocal_rank).tolist()

    result["sample_style_avg_precision"] = total_style_avg_precision
    result["sample_style_precision_at_k"] = total_style_precision_at_k
    result["sample_style_ndgc_score_at_k1"] = total_style_ndgc_score_at_k1
    result["sample_style_dgc_score_at_k1"] = total_style_dgc_score_at_k1
    result["sample_style_ndgc_score_at_k2"] = total_style_ndgc_score_at_k2
    result["sample_style_dgc_score_at_k2"] = total_style_dgc_score_at_k2
    result["sample_style_precision_at_k2"] = total_style_precision_at_k2
    result["sample_style_map_score"] = total_style_map_score
    result["sample_style_r_precision"] = total_style_r_precision
    result["sample_style_mean_reciprocal_rank"] = total_style_mean_reciprocal_rank

    total_style_avg_precision = np.mean(total_style_avg_precision).tolist()
    total_style_precision_at_k = np.mean(total_style_precision_at_k, axis=0).tolist()
    total_style_ndgc_score_at_k1 = np.mean(total_style_ndgc_score_at_k1, axis=0).tolist()
    total_style_dgc_score_at_k1 = np.mean(total_style_dgc_score_at_k1, axis=0).tolist()
    total_style_ndgc_score_at_k2 = np.mean(total_style_ndgc_score_at_k2, axis=0).tolist()
    total_style_dgc_score_at_k2 = np.mean(total_style_dgc_score_at_k2, axis=0).tolist()
    total_style_precision_at_k2 = np.mean(total_style_precision_at_k2, axis=0).tolist()
    total_style_map_score = np.mean(total_style_map_score).tolist()
    total_style_r_precision = np.mean(total_style_r_precision).tolist()
    total_style_mean_reciprocal_rank = np.mean(total_style_mean_reciprocal_rank).tolist()

    result["sample_material_avg_precision"] = total_material_avg_precision
    result["sample_material_precision_at_k"] = total_material_precision_at_k
    result["sample_material_ndgc_score_at_k1"] = total_material_ndgc_score_at_k1
    result["sample_material_dgc_score_at_k1"] = total_material_dgc_score_at_k1
    result["sample_material_ndgc_score_at_k2"] = total_material_ndgc_score_at_k2
    result["sample_material_dgc_score_at_k2"] = total_material_dgc_score_at_k2
    result["sample_material_precision_at_k2"] = total_material_precision_at_k2
    result["sample_material_map_score"] = total_material_map_score
    result["sample_material_r_precision"] = total_material_r_precision
    result["sample_material_mean_reciprocal_rank"] = total_material_mean_reciprocal_rank

    total_material_avg_precision = np.mean(total_material_avg_precision).tolist()
    total_material_precision_at_k = np.mean(total_material_precision_at_k, axis=0).tolist()
    total_material_ndgc_score_at_k1 = np.mean(total_material_ndgc_score_at_k1, axis=0).tolist()
    total_material_dgc_score_at_k1 = np.mean(total_material_dgc_score_at_k1, axis=0).tolist()
    total_material_ndgc_score_at_k2 = np.mean(total_material_ndgc_score_at_k2, axis=0).tolist()
    total_material_dgc_score_at_k2 = np.mean(total_material_dgc_score_at_k2, axis=0).tolist()
    total_material_precision_at_k2 = np.mean(total_material_precision_at_k2, axis=0).tolist()
    total_material_map_score = np.mean(total_material_map_score).tolist()
    total_material_r_precision = np.mean(total_material_r_precision).tolist()
    total_material_mean_reciprocal_rank = np.mean(total_material_mean_reciprocal_rank).tolist()

    result["sample_pattern_avg_precision"] = total_pattern_avg_precision
    result["sample_pattern_precision_at_k"] = total_pattern_precision_at_k
    result["sample_pattern_ndgc_score_at_k1"] = total_pattern_ndgc_score_at_k1
    result["sample_pattern_dgc_score_at_k1"] = total_pattern_dgc_score_at_k1
    result["sample_pattern_ndgc_score_at_k2"] = total_pattern_ndgc_score_at_k2
    result["sample_pattern_dgc_score_at_k2"] = total_pattern_dgc_score_at_k2
    result["sample_pattern_precision_at_k2"] = total_pattern_precision_at_k2
    result["sample_pattern_map_score"] = total_pattern_map_score
    result["sample_pattern_r_precision"] = total_pattern_r_precision
    result["sample_pattern_mean_reciprocal_rank"] = total_pattern_mean_reciprocal_rank

    total_pattern_avg_precision = np.mean(total_pattern_avg_precision).tolist()
    total_pattern_precision_at_k = np.mean(total_pattern_precision_at_k, axis=0).tolist()
    total_pattern_ndgc_score_at_k1 = np.mean(total_pattern_ndgc_score_at_k1, axis=0).tolist()
    total_pattern_dgc_score_at_k1 = np.mean(total_pattern_dgc_score_at_k1, axis=0).tolist()
    total_pattern_ndgc_score_at_k2 = np.mean(total_pattern_ndgc_score_at_k2, axis=0).tolist()
    total_pattern_dgc_score_at_k2 = np.mean(total_pattern_dgc_score_at_k2, axis=0).tolist()
    total_pattern_precision_at_k2 = np.mean(total_pattern_precision_at_k2, axis=0).tolist()
    total_pattern_map_score = np.mean(total_pattern_map_score).tolist()
    total_pattern_r_precision = np.mean(total_pattern_r_precision).tolist()
    total_pattern_mean_reciprocal_rank = np.mean(total_pattern_mean_reciprocal_rank).tolist()

    result["sample_brand_avg_precision"] = total_brand_avg_precision
    result["sample_brand_precision_at_k"] = total_brand_precision_at_k
    result["sample_brand_ndgc_score_at_k1"] = total_brand_ndgc_score_at_k1
    result["sample_brand_dgc_score_at_k1"] = total_brand_dgc_score_at_k1
    result["sample_brand_ndgc_score_at_k2"] = total_brand_ndgc_score_at_k2
    result["sample_brand_dgc_score_at_k2"] = total_brand_dgc_score_at_k2
    result["sample_brand_precision_at_k2"] = total_brand_precision_at_k2
    result["sample_brand_map_score"] = total_brand_map_score
    result["sample_brand_r_precision"] = total_brand_r_precision
    result["sample_brand_mean_reciprocal_rank"] = total_brand_mean_reciprocal_rank

    total_brand_avg_precision = np.mean(total_brand_avg_precision).tolist()
    total_brand_precision_at_k = np.mean(total_brand_precision_at_k, axis=0).tolist()
    total_brand_ndgc_score_at_k1 = np.mean(total_brand_ndgc_score_at_k1, axis=0).tolist()
    total_brand_dgc_score_at_k1 = np.mean(total_brand_dgc_score_at_k1, axis=0).tolist()
    total_brand_ndgc_score_at_k2 = np.mean(total_brand_ndgc_score_at_k2, axis=0).tolist()
    total_brand_dgc_score_at_k2 = np.mean(total_brand_dgc_score_at_k2, axis=0).tolist()
    total_brand_precision_at_k2 = np.mean(total_brand_precision_at_k2, axis=0).tolist()
    total_brand_map_score = np.mean(total_brand_map_score).tolist()
    total_brand_r_precision = np.mean(total_brand_r_precision).tolist()
    total_brand_mean_reciprocal_rank = np.mean(total_brand_mean_reciprocal_rank).tolist()

    result["item_avg_precision"] = total_item_avg_precision
    result["item_avg_precision_at_k"] = total_item_precision_at_k
    result["item_avg_precision_at_k2"] = total_item_precision_at_k2
    result["item_avg_ndgc_at_k"] = total_item_ndgc_score_at_k1
    result["item_avg_ndgc_at_k2"] = total_item_ndgc_score_at_k2
    result["item_avg_dgc_at_k"] = total_item_dgc_score_at_k1
    result["item_avg_dgc_at_k2"] = total_item_dgc_score_at_k2
    result["item_avg_map"] = total_item_map_score
    result["item_avg_r_precision"] = total_item_r_precision
    result["item_avg_mean_reciprocal_rank"] = total_item_mean_reciprocal_rank

    result["style_avg_precision"] = total_style_avg_precision
    result["style_avg_precision_at_k"] = total_style_precision_at_k
    result["style_avg_precision_at_k2"] = total_style_precision_at_k2
    result["style_avg_ndgc_at_k"] = total_style_ndgc_score_at_k1
    result["style_avg_ndgc_at_k2"] = total_style_ndgc_score_at_k2
    result["style_avg_dgc_at_k"] = total_style_dgc_score_at_k1
    result["style_avg_dgc_at_k2"] = total_style_dgc_score_at_k2
    result["style_avg_map"] = total_style_map_score
    result["style_avg_r_precision"] = total_style_r_precision
    result["style_avg_mean_reciprocal_rank"] = total_style_mean_reciprocal_rank

    result["material_avg_precision"] = total_material_avg_precision
    result["material_avg_precision_at_k"] = total_material_precision_at_k
    result["material_avg_precision_at_k2"] = total_material_precision_at_k2
    result["material_avg_ndgc_at_k"] = total_material_ndgc_score_at_k1
    result["material_avg_ndgc_at_k2"] = total_material_ndgc_score_at_k2
    result["material_avg_dgc_at_k"] = total_material_dgc_score_at_k1
    result["material_avg_dgc_at_k2"] = total_material_dgc_score_at_k2
    result["material_avg_map"] = total_material_map_score
    result["material_avg_r_precision"] = total_material_r_precision
    result["material_avg_mean_reciprocal_rank"] = total_material_mean_reciprocal_rank

    result["pattern_avg_precision"] = total_pattern_avg_precision
    result["pattern_avg_precision_at_k"] = total_pattern_precision_at_k
    result["pattern_avg_precision_at_k2"] = total_pattern_precision_at_k2
    result["pattern_avg_ndgc_at_k"] = total_pattern_ndgc_score_at_k1
    result["pattern_avg_ndgc_at_k2"] = total_pattern_ndgc_score_at_k2
    result["pattern_avg_dgc_at_k"] = total_pattern_dgc_score_at_k1
    result["pattern_avg_dgc_at_k2"] = total_pattern_dgc_score_at_k2
    result["pattern_avg_map"] = total_pattern_map_score
    result["pattern_avg_r_precision"] = total_pattern_r_precision
    result["pattern_avg_mean_reciprocal_rank"] = total_pattern_mean_reciprocal_rank

    result["brand_avg_precision"] = total_brand_avg_precision
    result["brand_avg_precision_at_k"] = total_brand_precision_at_k
    result["brand_avg_precision_at_k2"] = total_brand_precision_at_k2
    result["brand_avg_ndgc_at_k"] = total_brand_ndgc_score_at_k1
    result["brand_avg_ndgc_at_k2"] = total_brand_ndgc_score_at_k2
    result["brand_avg_dgc_at_k"] = total_brand_dgc_score_at_k1
    result["brand_avg_dgc_at_k2"] = total_brand_dgc_score_at_k2
    result["brand_avg_map"] = total_brand_map_score
    result["brand_avg_r_precision"] = total_brand_r_precision
    result["brand_avg_mean_reciprocal_rank"] = total_brand_mean_reciprocal_rank

    return result


def text_clustering_LF(caption, comments, segmented_hashtags, tags, hashtags, userhandles, information_extractor, i):
    """ Do semantic text clustering to predict a given Instagram post"""
    top_items = information_extractor.find_closest_semantic_hierarchy(caption, comments, tags,
                                                                       hashtags + segmented_hashtags,
                                                                       information_extractor.hieararchy, i, 10)
    styles = information_extractor.find_closest_semantic_hierarchy(caption, comments, tags,
                                                                    hashtags + segmented_hashtags,
                                                                    information_extractor.styles, i, 10)
    sub_items = sorted(
        list(
            set(
                information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                            information_extractor.items_lemmas.keys(), i))
        ), reverse=True, key=lambda x: x[1]
    )
    materials = sorted(
        list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                             information_extractor.materials_lemmas.keys(), i))),
        reverse=True, key=lambda x: x[1])
    brands = sorted(
        list(set(information_extractor.find_closest_semantic(caption + userhandles, comments, tags, hashtags,
                                                             segmented_hashtags, 10,
                                                             information_extractor.companies, i))),
        reverse=True, key=lambda x: x[1])
    patterns = sorted(
        list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                             information_extractor.patterns, i))),
        reverse=True, key=lambda x: x[1])
    ranked_materials = sorted(re_rank_materials(information_extractor, materials), reverse=True, key=lambda x: x[1])
    ranked_brands = sorted(re_rank_brands(information_extractor, brands), reverse=True, key=lambda x: x[1])
    text_clustering_LF = {}
    text_clustering_LF["brands"] = dict(ranked_brands)
    text_clustering_LF["patterns"] = dict(patterns)
    text_clustering_LF["materials"] = dict(ranked_materials)
    text_clustering_LF["item-category"] = dict(top_items)
    text_clustering_LF["item-sub-category"] = dict(sub_items)
    text_clustering_LF["styles"] = dict(styles)

    text_clustering_LF2 = {}
    text_clustering_LF2["brands"] = dict(brands)
    text_clustering_LF2["patterns"] = dict(patterns)
    text_clustering_LF2["materials"] = dict(materials)
    text_clustering_LF2["item-category"] = dict(top_items)
    text_clustering_LF2["item-sub-category"] = dict(sub_items)
    text_clustering_LF2["styles"] = dict(styles)
    return text_clustering_LF, text_clustering_LF2


def re_rank_materials(information_extractor, materials):
    """ Re-rank materials based on probase lookup"""
    ranked_materials = map(lambda material: material_rank_mapper(information_extractor, material), materials)
    return ranked_materials


def material_rank_mapper(information_extractor, material_):
    """ Re-rank materials based on probase lookup"""
    material, rank = material_
    factor_probase = lookup_material_probase(information_extractor, material, 10)
    return (material, factor_probase * rank)


def lookup_material_probase(information_extractor, query, num):
    """Lookup material in Probase"""
    material_params = {
        'instance': query,
        'topK': num
    }
    result = information_extractor.lookup_probase(material_params)
    rank = information_extractor.rank_probase_result_material(result)
    return rank


def re_rank_brands(information_extractor, brands):
    """ Re-rank brands based on probase lookup"""
    ranked_brands = map(lambda brand: brand_rank_mapper(information_extractor, brand), brands)
    return ranked_brands


def brand_rank_mapper(information_extractor, brand_):
    """ Re-rank brands based on probase lookup"""
    brand, rank = brand_
    factor_probase = lookup_company_probase(information_extractor, brand, 10)
    return (brand, rank * factor_probase)


def lookup_company_probase(information_extractor, query, num):
    """ Lookup company in probase"""
    company_params = {
        'instance': query,
        'topK': num
    }
    result = information_extractor.lookup_probase(company_params)
    rank = information_extractor.rank_probase_result_company(result)
    return rank


def read_gazetter(file_path):
    """ Read domain list"""
    words = []
    with open(file_path, "r") as file:
        for line in file:
            words.append(line.replace("\n", "").lower())
    return words


def text_clustering_LF_syntactic(caption, comments, segmented_hashtags, tags, hashtags, userhandles,
                                 information_extractor, i):
    """ Do semantic text clustering to predict a given Instagram post"""
    top_items = information_extractor.find_closest_syntactic_hierarchy(caption, comments, tags,
                                                                        hashtags + segmented_hashtags,
                                                                        information_extractor.hieararchy, i, 10)
    styles = information_extractor.find_closest_syntactic_hierarchy(caption, comments, tags,
                                                                     hashtags + segmented_hashtags,
                                                                     information_extractor.styles, i, 10)
    sub_items = sorted(
        list(
            set(
                information_extractor.find_closest_syntactic2(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                              information_extractor.items_lemmas.keys(), i))
        ), reverse=True, key=lambda x: x[1]
    )
    materials = sorted(
        list(
            set(information_extractor.find_closest_syntactic2(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                              information_extractor.materials_lemmas.keys(), i))),
        reverse=True, key=lambda x: x[1])
    brands = sorted(
        list(
            set(information_extractor.find_closest_syntactic2(caption + userhandles, comments, tags, hashtags,
                                                              segmented_hashtags, 10,
                                                              information_extractor.companies, i))),
        reverse=True, key=lambda x: x[1])
    patterns = sorted(
        list(
            set(information_extractor.find_closest_syntactic2(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                              information_extractor.patterns, tfidf))),
        reverse=True, key=lambda x: x[1])
    ranked_materials = sorted(re_rank_materials(information_extractor, materials), reverse=True, key=lambda x: x[1])
    ranked_brands = sorted(re_rank_brands(information_extractor, brands), reverse=True, key=lambda x: x[1])
    text_clustering_LF = {}
    text_clustering_LF["brands"] = dict(ranked_brands)
    text_clustering_LF["patterns"] = dict(patterns)
    text_clustering_LF["materials"] = dict(ranked_materials)
    text_clustering_LF["item-category"] = dict(top_items)
    text_clustering_LF["item-sub-category"] = dict(sub_items)
    text_clustering_LF["styles"] = dict(styles)
    return text_clustering_LF


def calculate_p_values_vectors(vectors_res_path, focus_index, outputPath):
    """ Calculate p-values that some set of vectors are significantly superior than the rest"""
    vectors_evals = json.load(open(vectors_res_path))
    focus_res = vectors_evals[focus_index]
    results = []
    for i in range(0, len(vectors_evals)):
        if not i == focus_index:
            current = vectors_evals[i]
            item_map_current_sample = current["semantic"]["sample_item_map_score"]
            item_map_focus_sample = focus_res["semantic"]["sample_item_map_score"]
            p_values = {}

            p_values["focus_current_map_item_pvalue"] = t_test_p_value(item_map_focus_sample, item_map_current_sample)
            for i in range(1, 11):
                precision_1_i_current_sample = column(current["semantic"]["sample_item_precision_at_k"], i - 1)
                precision_2_i_current_sample = column(current["semantic"]["sample_item_precision_at_k2"], i - 1)
                ndgc_1_i_current_sample = column(current["semantic"]["sample_item_ndgc_score_at_k1"], i - 1)
                ndgc_2_i_current_sample = column(current["semantic"]["sample_item_ndgc_score_at_k2"], i - 1)

                precision_1_i_focus_sample = column(focus_res["semantic"]["sample_item_precision_at_k"], i - 1)
                precision_2_i_focus_sample = column(focus_res["semantic"]["sample_item_precision_at_k2"], i - 1)
                ndgc_1_i_focus_sample = column(focus_res["semantic"]["sample_item_ndgc_score_at_k1"], i - 1)
                ndgc_2_i_focus_sample = column(focus_res["semantic"]["sample_item_ndgc_score_at_k2"], i - 1)

                p_values["focus_current_precision1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(
                    precision_1_i_focus_sample, precision_1_i_current_sample)
                p_values["focus_current_precision2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(
                    precision_2_i_focus_sample, precision_2_i_current_sample)

                p_values["focus_current_ndgc1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_1_i_focus_sample,
                                                                                               ndgc_1_i_current_sample)
                p_values["focus_current_ndgc2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_2_i_focus_sample,
                                                                                               ndgc_2_i_current_sample)

            p_values["focus"] = focus_res["vectors"]
            p_values["current"] = current["vectors"]
            results.append(p_values)
    with open(outputPath, 'w') as fp:
        json.dump(results, fp)
    return results


def column(matrix, i):
    """Helper to get column of matrix"""
    return [row[i] for row in matrix]


def calculate_p_values_sem_syn_probase(res):
    """ Calculates p-values between SemCluster and the baselines"""
    item_map_semantic_sample = res["semantic"]["sample_item_map_score"]
    item_map_syntactic_sample = res["syntactic"]["sample_item_map_score"]
    item_map_semantic_wo_probase_sample = res["semantic_wo_probase"]["sample_item_map_score"]
    p_values = {}
    p_values["sem_syn_map_item_pvalue"] = t_test_p_value(item_map_semantic_sample, item_map_syntactic_sample)
    p_values["sem_semwoprobase_map_item_pvalue"] = t_test_p_value(item_map_semantic_sample,
                                                                  item_map_semantic_wo_probase_sample)
    for i in range(1, 11):
        precision_1_i_semantic_sample = column(res["semantic"]["sample_item_precision_at_k"], i - 1)
        precision_2_i_semantic_sample = column(res["semantic"]["sample_item_precision_at_k2"], i - 1)
        ndgc_1_i_semantic_sample = column(res["semantic"]["sample_item_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_sample = column(res["semantic"]["sample_item_ndgc_score_at_k2"], i - 1)
        precision_1_i_syntactic_sample = column(res["syntactic"]["sample_item_precision_at_k"], i - 1)
        precision_2_i_syntactic_sample = column(res["syntactic"]["sample_item_precision_at_k2"], i - 1)
        ndgc_1_i_syntactic_sample = column(res["syntactic"]["sample_item_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_syntactic_sample = column(res["syntactic"]["sample_item_ndgc_score_at_k2"], i - 1)
        precision_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_item_precision_at_k"],
                                                          i - 1)
        precision_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_item_precision_at_k2"],
                                                          i - 1)
        ndgc_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_item_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_item_ndgc_score_at_k2"], i - 1)

        p_values["sem_syn_precision1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(precision_1_i_semantic_sample,
                                                                                      precision_1_i_syntactic_sample)
        p_values["sem_semwoprobase_precision1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(
            precision_1_i_semantic_sample, precision_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_precision2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(precision_2_i_semantic_sample,
                                                                                      precision_2_i_syntactic_sample)
        p_values["sem_semwoprobase_precision2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(
            precision_2_i_semantic_sample, precision_2_i_semantic_wo_probase_sample)

        p_values["sem_syn_ndgc1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                 ndgc_1_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc1_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                          ndgc_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_ndgc2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                 ndgc_2_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc2_at_" + str(i) + "_item_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                          ndgc_2_i_semantic_wo_probase_sample)

    style_map_semantic_sample = res["semantic"]["sample_style_map_score"]
    style_map_syntactic_sample = res["syntactic"]["sample_style_map_score"]
    style_map_semantic_wo_probase_sample = res["semantic_wo_probase"]["sample_style_map_score"]
    p_values["sem_syn_map_style_pvalue"] = t_test_p_value(style_map_semantic_sample, style_map_syntactic_sample)
    p_values["sem_semwoprobase_map_style_pvalue"] = t_test_p_value(style_map_semantic_sample,
                                                                   style_map_semantic_wo_probase_sample)
    for i in range(1, 11):
        precision_1_i_semantic_sample = column(res["semantic"]["sample_style_precision_at_k"], i - 1)
        precision_2_i_semantic_sample = column(res["semantic"]["sample_style_precision_at_k2"], i - 1)
        ndgc_1_i_semantic_sample = column(res["semantic"]["sample_style_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_sample = column(res["semantic"]["sample_style_ndgc_score_at_k2"], i - 1)
        precision_1_i_syntactic_sample = column(res["syntactic"]["sample_style_precision_at_k"], i - 1)
        precision_2_i_syntactic_sample = column(res["syntactic"]["sample_style_precision_at_k2"], i - 1)
        ndgc_1_i_syntactic_sample = column(res["syntactic"]["sample_style_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_syntactic_sample = column(res["syntactic"]["sample_style_ndgc_score_at_k2"], i - 1)
        precision_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_style_precision_at_k"],
                                                          i - 1)
        precision_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_style_precision_at_k2"],
                                                          i - 1)
        ndgc_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_style_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_style_ndgc_score_at_k2"], i - 1)

        p_values["sem_syn_precision1_at_" + str(i) + "_style_pvalue"] = t_test_p_value(precision_1_i_semantic_sample,
                                                                                       precision_1_i_syntactic_sample)
        p_values["sem_semwoprobase_precision1_at_" + str(i) + "_style_pvalue"] = t_test_p_value(
            precision_1_i_semantic_sample, precision_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_precision2_at_" + str(i) + "_style_pvalue"] = t_test_p_value(precision_2_i_semantic_sample,
                                                                                       precision_2_i_syntactic_sample)
        p_values["sem_semwoprobase_precision2_at_" + str(i) + "_style_pvalue"] = t_test_p_value(
            precision_2_i_semantic_sample, precision_2_i_semantic_wo_probase_sample)

        p_values["sem_syn_ndgc1_at_" + str(i) + "_style_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                  ndgc_1_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc1_at_" + str(i) + "_style_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                           ndgc_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_ndgc2_at_" + str(i) + "_style_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                  ndgc_2_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc2_at_" + str(i) + "_style_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                           ndgc_2_i_semantic_wo_probase_sample)

    material_map_semantic_sample = res["semantic"]["sample_material_map_score"]
    material_map_syntactic_sample = res["syntactic"]["sample_material_map_score"]
    material_map_semantic_wo_probase_sample = res["semantic_wo_probase"]["sample_material_map_score"]
    p_values["sem_syn_map_material_pvalue"] = t_test_p_value(material_map_semantic_sample,
                                                             material_map_syntactic_sample)
    p_values["sem_semwoprobase_map_material_pvalue"] = t_test_p_value(material_map_semantic_sample,
                                                                      material_map_semantic_wo_probase_sample)
    for i in range(1, 11):
        precision_1_i_semantic_sample = column(res["semantic"]["sample_material_precision_at_k"], i - 1)
        precision_2_i_semantic_sample = column(res["semantic"]["sample_material_precision_at_k2"], i - 1)
        ndgc_1_i_semantic_sample = column(res["semantic"]["sample_material_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_sample = column(res["semantic"]["sample_material_ndgc_score_at_k2"], i - 1)
        precision_1_i_syntactic_sample = column(res["syntactic"]["sample_material_precision_at_k"], i - 1)
        precision_2_i_syntactic_sample = column(res["syntactic"]["sample_material_precision_at_k2"], i - 1)
        ndgc_1_i_syntactic_sample = column(res["syntactic"]["sample_material_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_syntactic_sample = column(res["syntactic"]["sample_material_ndgc_score_at_k2"], i - 1)
        precision_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_material_precision_at_k"],
                                                          i - 1)
        precision_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_material_precision_at_k2"],
                                                          i - 1)
        ndgc_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_material_ndgc_score_at_k1"],
                                                     i - 1)
        ndgc_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_material_ndgc_score_at_k2"],
                                                     i - 1)

        p_values["sem_syn_precision1_at_" + str(i) + "_material_pvalue"] = t_test_p_value(precision_1_i_semantic_sample,
                                                                                          precision_1_i_syntactic_sample)
        p_values["sem_semwoprobase_precision1_at_" + str(i) + "_material_pvalue"] = t_test_p_value(
            precision_1_i_semantic_sample, precision_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_precision2_at_" + str(i) + "_material_pvalue"] = t_test_p_value(precision_2_i_semantic_sample,
                                                                                          precision_2_i_syntactic_sample)
        p_values["sem_semwoprobase_precision2_at_" + str(i) + "_material_pvalue"] = t_test_p_value(
            precision_2_i_semantic_sample, precision_2_i_semantic_wo_probase_sample)

        p_values["sem_syn_ndgc1_at_" + str(i) + "_material_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                     ndgc_1_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc1_at_" + str(i) + "_material_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                              ndgc_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_ndgc2_at_" + str(i) + "_material_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                     ndgc_2_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc2_at_" + str(i) + "_material_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                              ndgc_2_i_semantic_wo_probase_sample)

    pattern_map_semantic_sample = res["semantic"]["sample_pattern_map_score"]
    pattern_map_syntactic_sample = res["syntactic"]["sample_pattern_map_score"]
    pattern_map_semantic_wo_probase_sample = res["semantic_wo_probase"]["sample_pattern_map_score"]
    p_values["sem_syn_map_pattern_pvalue"] = t_test_p_value(pattern_map_semantic_sample, pattern_map_syntactic_sample)
    p_values["sem_semwoprobase_map_pattern_pvalue"] = t_test_p_value(pattern_map_semantic_sample,
                                                                     pattern_map_semantic_wo_probase_sample)
    for i in range(1, 11):
        precision_1_i_semantic_sample = column(res["semantic"]["sample_pattern_precision_at_k"], i - 1)
        precision_2_i_semantic_sample = column(res["semantic"]["sample_pattern_precision_at_k2"], i - 1)
        ndgc_1_i_semantic_sample = column(res["semantic"]["sample_pattern_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_sample = column(res["semantic"]["sample_pattern_ndgc_score_at_k2"], i - 1)
        precision_1_i_syntactic_sample = column(res["syntactic"]["sample_pattern_precision_at_k"], i - 1)
        precision_2_i_syntactic_sample = column(res["syntactic"]["sample_pattern_precision_at_k2"], i - 1)
        ndgc_1_i_syntactic_sample = column(res["syntactic"]["sample_pattern_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_syntactic_sample = column(res["syntactic"]["sample_pattern_ndgc_score_at_k2"], i - 1)
        precision_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_pattern_precision_at_k"],
                                                          i - 1)
        precision_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_pattern_precision_at_k2"],
                                                          i - 1)
        ndgc_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_pattern_ndgc_score_at_k1"],
                                                     i - 1)
        ndgc_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_pattern_ndgc_score_at_k2"],
                                                     i - 1)

        p_values["sem_syn_precision1_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(precision_1_i_semantic_sample,
                                                                                         precision_1_i_syntactic_sample)
        p_values["sem_semwoprobase_precision1_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(
            precision_1_i_semantic_sample, precision_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_precision2_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(precision_2_i_semantic_sample,
                                                                                         precision_2_i_syntactic_sample)
        p_values["sem_semwoprobase_precision2_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(
            precision_2_i_semantic_sample, precision_2_i_semantic_wo_probase_sample)

        p_values["sem_syn_ndgc1_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                    ndgc_1_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc1_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                             ndgc_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_ndgc2_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                    ndgc_2_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc2_at_" + str(i) + "_pattern_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                             ndgc_2_i_semantic_wo_probase_sample)

    brand_map_semantic_sample = res["semantic"]["sample_brand_map_score"]
    brand_map_syntactic_sample = res["syntactic"]["sample_brand_map_score"]
    brand_map_semantic_wo_probase_sample = res["semantic_wo_probase"]["sample_brand_map_score"]
    p_values["sem_syn_map_brand_pvalue"] = t_test_p_value(brand_map_semantic_sample, brand_map_syntactic_sample)
    p_values["sem_semwoprobase_map_brand_pvalue"] = t_test_p_value(brand_map_semantic_sample,
                                                                   brand_map_semantic_wo_probase_sample)
    for i in range(1, 11):
        precision_1_i_semantic_sample = column(res["semantic"]["sample_brand_precision_at_k"], i - 1)
        precision_2_i_semantic_sample = column(res["semantic"]["sample_brand_precision_at_k2"], i - 1)
        ndgc_1_i_semantic_sample = column(res["semantic"]["sample_brand_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_sample = column(res["semantic"]["sample_brand_ndgc_score_at_k2"], i - 1)
        precision_1_i_syntactic_sample = column(res["syntactic"]["sample_brand_precision_at_k"], i - 1)
        precision_2_i_syntactic_sample = column(res["syntactic"]["sample_brand_precision_at_k2"], i - 1)
        ndgc_1_i_syntactic_sample = column(res["syntactic"]["sample_brand_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_syntactic_sample = column(res["syntactic"]["sample_brand_ndgc_score_at_k2"], i - 1)
        precision_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_brand_precision_at_k"],
                                                          i - 1)
        precision_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_brand_precision_at_k2"],
                                                          i - 1)
        ndgc_1_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_brand_ndgc_score_at_k1"], i - 1)
        ndgc_2_i_semantic_wo_probase_sample = column(res["semantic_wo_probase"]["sample_brand_ndgc_score_at_k2"], i - 1)

        p_values["sem_syn_precision1_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(precision_1_i_semantic_sample,
                                                                                       precision_1_i_syntactic_sample)
        p_values["sem_semwoprobase_precision1_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(
            precision_1_i_semantic_sample, precision_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_precision2_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(precision_2_i_semantic_sample,
                                                                                       precision_2_i_syntactic_sample)
        p_values["sem_semwoprobase_precision2_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(
            precision_2_i_semantic_sample, precision_2_i_semantic_wo_probase_sample)

        p_values["sem_syn_ndgc1_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                  ndgc_1_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc1_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(ndgc_1_i_semantic_sample,
                                                                                           ndgc_1_i_semantic_wo_probase_sample)
        p_values["sem_syn_ndgc2_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                  ndgc_2_i_syntactic_sample)
        p_values["sem_semwoprobase_ndgc2_at_" + str(i) + "_brand_pvalue"] = t_test_p_value(ndgc_2_i_semantic_sample,
                                                                                           ndgc_2_i_semantic_wo_probase_sample)
    return p_values


def t_test_p_value(sample1, sample2):
    """ Performs pairwise t-test for computing the p-value given two samples"""
    if(len(sample1) > 1 and len(sample2 > 1)):
        res = stats.ttest_rel(sample1, sample2)
        return res.pvalue
    else:
        return 1.0


def eval_vectors(args):
    """ Method for evaluating a list of vectors"""
    vectors = json.load(open(args.testvectors))
    annotations_to_csv(args.labels, "./eval/annotations.csv")
    materials = read_gazetter(args.materials)
    items = read_gazetter(args.items)
    styles = read_gazetter(args.styles)
    companies = read_gazetter(args.brands)
    probase_brands = read_gazetter(args.probasebrands)
    probase_materials = read_gazetter(args.probasematerials)
    patterns = read_gazetter(args.patterns)
    item_top_categories = read_gazetter(args.itemtopcategory)
    evals = []
    x, y, x_all = extract_features("./eval/annotations.csv", args.input)
    tfidf = create_tf_idf(x_all, len(x))
    for vec in vectors:
        information_extractor = InformationExtractor(vec, companies, styles, materials, items, probase_brands,
                                                     probase_materials, patterns, item_top_categories, False, args.conf, tfidf)
        semantic_predictions, semantic_wo_probase_predictions = semantic_clustering_predict(x, information_extractor)
        semantic_res = eval(semantic_predictions, y)
        semantic_wo_probase_res = eval(semantic_wo_probase_predictions, y)
        res = {}
        res["semantic"] = semantic_res
        res["semantic_wo_probase"] = semantic_wo_probase_res
        res["vectors"] = vec
        evals.append(res)

    with open(args.output + 'vectors_results.json', 'w') as fp:
        json.dump(evals, fp)


def eval_syn_vs_sem(args):
    """ Method for SemCluster against baselines"""
    annotations_to_csv(args.labels, "./eval/annotations.csv")
    materials = read_gazetter(args.materials)
    items = read_gazetter(args.items)
    styles = read_gazetter(args.styles)
    companies = read_gazetter(args.brands)
    probase_brands = read_gazetter(args.probasebrands)
    probase_materials = read_gazetter(args.probasematerials)
    patterns = read_gazetter(args.patterns)
    item_top_categories = read_gazetter(args.itemtopcategory)
    x, y, x_all = extract_features("./eval/annotations.csv", args.input)
    tfidf = create_tf_idf(x_all, len(x))
    information_extractor = InformationExtractor(args.vectors, companies, styles,
                                                 materials, items, probase_brands, probase_materials, patterns,
                                                 item_top_categories, False, args.conf, tfidf)
    semantic_predictions, semantic_wo_probase_predictions = semantic_clustering_predict(x, information_extractor)
    syntactic_predictions = syntactic_clustering_predict(x, information_extractor)
    semantic_res = eval(semantic_predictions, y)
    syntactic_res = eval(syntactic_predictions, y)
    semantic_wo_probase_res = eval(semantic_wo_probase_predictions, y)
    res = {}
    res["semantic"] = semantic_res
    res["syntactic"] = syntactic_res
    res["semantic_wo_probase"] = semantic_wo_probase_res
    p_values = calculate_p_values_sem_syn_probase(res)
    res["p-values"] = p_values
    with open(args.output + '/syn_sem_results.json', 'w') as fp:
        json.dump(res, fp)


def main():
    """ Program entrypoint, orchestrates the pipeline"""
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = parse_args()
    #eval_vectors(args)
    eval_syn_vs_sem(args)
    #calculate_p_values_vectors(args.output + "/vectors_results.json", 0, args.output + "./vectors_pvalues.json")


if __name__ == '__main__':
    main()
