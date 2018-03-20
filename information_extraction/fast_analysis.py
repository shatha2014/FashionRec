#!/usr/bin/env python
# coding=utf-8
from InformationExtraction import InformationExtractor
import os
import re
import emoji
import pyspark
import pyspark.sql
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from collections import Counter
from wordsegment import load, segment
import argparse
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

"""
Script for analyzing users, uses Preprocessor.py and InformationExtraction.py
Uses Spark for distribution.
Uses a plethora of external APIs as distant supervision, and uses local semantic clustering of text 
"""

# Initialize schema, tokenizer, stopwords, and load word segmenter
schema = (StructType().add('id', IntegerType(), True)
          .add('comments', StringType(), True)
          .add('caption', StringType(), True)
          .add('tags', DoubleType(), True))
tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
tknzr_strip_users = TweetTokenizer(strip_handles=True, reduce_len=True)
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
load()


def clean_text(text, tokenizer):
    """ Clean the text, lowercase, remove noise, lemmatize """
    text = text.decode('utf-8', 'ignore').encode("utf-8")
    list_of_words = [i.lower() for i in tokenizer.tokenize(text.decode("utf-8")) if
                     i.lower() not in stop_words]  # remove stopwords & tokenize (preserve hashtags)
    list_of_words = filter(lambda x: "pic.twitter" not in x and not "twitter.com" in x, list_of_words)
    list_of_words = filter(lambda x: "http" not in x, list_of_words)
    list_of_words = map(lambda x: wordnet_lemmatizer.lemmatize(x), list_of_words)  # lemmatize
    list_of_words = map(lambda x: x.encode("utf-8"), list_of_words)
    return list_of_words


def create_tf_idf(corporaPath):
    """
    Compute the TF-IDF scores based on the entire corpus
    """
    docs = []
    ids = []
    with open(corporaPath, "r") as csvfile:
        for line in csvfile:
            line = line.replace("\n", " ")
            parts = line.split(",")
            if (len(parts) == 5):
                id = parts[0]
                url = parts[1]
                comments = parts[2]
                if comments is not None:
                    comments = clean_text(comments, tknzr_strip_users)
                else:
                    comments = []
                caption = parts[3]
                if caption is not None:
                    caption = clean_text(caption, tknzr_strip_users)
                else:
                    caption = []
                tags = parts[4]
                if tags is not None:
                    tags = clean_text(tags, tknzr_strip_users)
                else:
                    tags = []
            docs.append(comments + caption + tags)
            ids.append(id)
    idx_to_id = {}
    for i, id in enumerate(ids):
        idx_to_id[i] = id

    dct = Dictionary(docs)
    corpus = [dct.doc2bow(line) for line in docs]
    model = TfidfModel(corpus)
    tfidf_factors = {}
    for i, doc in enumerate(corpus):
        temp = {}
        for word_id, value in model[doc]:
            word = dct.get(word_id)
            temp[word] = value
        tfidf_factors[idx_to_id[i]] = temp
    return tfidf_factors


def filterOccurenceCount(iter):
    """Filter occurrences"""
    A = Counter(iter)
    return {x: A[x] for x in A if A[x] >= 3}.keys()


def filterEmojis(emojis, tokens):
    """Filter emojis"""
    return filter(lambda x: x not in emojis, tokens)


def read_gazetter(file_path):
    """ Read domain list"""
    words = []
    with open(file_path, "r") as file:
        for line in file:
            words.append(line.replace("\n", "").lower())
    return words


def parse_args():
    """Parses the commandline arguments with argparse"""
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-i", "--input", help="folder where input documents are", default="data/sample_user.csv")
    parser.add_argument("-c", "--conf", help="path to confFile", default="./conf/conf.json")
    parser.add_argument("-im", "--imagepath", help="path to images, necessary if using google vision API", default="")
    parser.add_argument("-o", "--output", help="folder where output documents are", default="output")
    parser.add_argument("-g", "--google", help="flag whether to use google vision api",
                        action="store_true")
    parser.add_argument("-ta", "--textanalysis", help="flag whether to use text analysis pipeline",
                        action="store_true")
    parser.add_argument("-dd", "--deepdetect", help="flag whether to use deepdetect API",
                        action="store_true")
    parser.add_argument("-ca", "--clarifai", help="flag whether to use clarifai API",
                        action="store_true")
    parser.add_argument("-dm", "--deepomatic", help="flag whether to use deepomatic API",
                        action="store_true")
    parser.add_argument("-lk", "--liketkit", help="flag whether to scrape liketkit links in the classification",
                        action="store_true")
    parser.add_argument("-pa", "--partitions", help="number of spark partitions",
                        default=1)
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

    args = parser.parse_args()
    return args


def sparkConf():
    "Setup spark configuration, change this to run on cluster"
    conf = pyspark.SparkConf()
    return conf.setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false").set(
        "spark.executor.heartbeatInterval", "20s").set("spark.rpc.message.maxSize", "512").set(
        "spark.kryoserializer.buffer.max", "1024")
    # return conf \
    #    .setMaster("local[8]") \
    #    .setAppName("fashion_rec_data_setup") \
    #    .set("spark.hadoop.validateOutputSpecs", "false")


def parse_raw(sql, filePath):
    """ Read corpora with spark"""
    df = sql.read.csv(
        filePath, header=False, mode="DROPMALFORMED"
    )
    return df


def map_post(row, information_extractor, index, args):
    """Process post, semantic + syntactic similarity classification"""
    if index % 10 == 0:
        print ("Processing post with index {0}".format(index))
    text_clustering_res = {}
    liketkit_classes = {}
    google_vision_classes = {}
    deep_detect_classes = {}
    clarifai_classes = {}
    deepomatic_classes = {}
    if (args.textanalysis):
        text_clustering_res = text_clustering_LF(row, information_extractor)
    if (args.liketkit):
        liketkit_classes = liktekit_LF(row, information_extractor)
    if (args.google):
        google_vision_classes = google_vision_LF(row, information_extractor)
    if (args.deepdetect):
        deep_detect_classes = deep_detect_lookup(row, information_extractor)
    if (args.clarifai):
        clarifai_classes = clarifai_lookup(row, information_extractor)
    if (args.deepomatic):
        deepomatic_classes = deepomatic_lookup(row, information_extractor)
    row = pyspark.sql.Row(id=row.id, hashtags=row.hashtags, links=row.links, text_clustering=text_clustering_res,
                          liketkit_classification=liketkit_classes,
                          google_vision_classification=google_vision_classes,
                          deep_detect_classification=deep_detect_classes,
                          clarifai_classification=clarifai_classes,
                          deepomatic_classification=deepomatic_classes,
                          url=row.url)
    return row


def google_vision_LF(row, information_extractor):
    """ Analyze image with Google vision API """
    item_candidates = information_extractor.google_vision_lookup(row.image_path)
    items = information_extractor.map_candidates_to_ontology(item_candidates)
    google_vision_classes = {}
    google_vision_classes["items"] = dict(items)
    return google_vision_classes


def deep_detect_lookup(row, information_extractor):
    """ Analyze image with deepdetect """
    items_and_fabrics = information_extractor.deep_detect_lookup(row.url)
    items = information_extractor.map_candidates_to_ontology(items_and_fabrics["items"])
    deep_detect_classes = {}
    deep_detect_classes["items"] = dict(items)
    return deep_detect_classes


def deepomatic_lookup(row, information_extractor):
    """ Analyze image with deepomatic """
    candidates = information_extractor.deepomatic_lookup(row.url)
    items = information_extractor.map_candidates_to_ontology(candidates)
    deepomatic_classes = {}
    deepomatic_classes["items"] = dict(items)
    return deepomatic_classes


def clarifai_lookup(row, information_extractor):
    """ Analyze image with clarifai"""
    candidates = information_extractor.clarifai_lookup(row.url)
    items = information_extractor.map_candidates_to_ontology(candidates)
    clarifai_classes = {}
    clarifai_classes["items"] = dict(items)
    return clarifai_classes


def emoji_LF(row, information_extractor):
    """ Analyze image based on emojis"""

    emoji_classes = information_extractor.emoji_classification(row.emojis, 10)
    return dict(emoji_classes)


def liktekit_LF(row, information_extractor):
    """ Analyze image based on liketkit link scraping"""
    text = []
    for link in row.links:
        text.extend(information_extractor.liketkit_classification(link))
    liktekit_classes = {}
    liktekit_classes["items"] = dict(
        information_extractor.find_closest_semantic_hierarchy(text, [], [], [], information_extractor.hieararchy))
    return liktekit_classes


def text_clustering_LF(row, information_extractor):
    """ Analyze image based on semantic text clustering """
    caption = row.caption
    comments = row.comments
    hashtags = row.hashtags
    segmented_hashtags = row.segmented_hashtags
    userhandles = row.userhandles
    id = row.id
    tags = row.tags

    top_items = information_extractor.find_closest_semantic_hierarchy(caption, comments, tags,
                                                                      hashtags + segmented_hashtags,
                                                                      information_extractor.top_category_items, id, 10)
    styles = information_extractor.find_closest_semantic_hierarchy(caption, comments, tags,
                                                                   hashtags + segmented_hashtags,
                                                                   information_extractor.styles, id, 10)
    sub_items = sorted(
        list(
            set(
                information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                            information_extractor.items_lemmas.keys(), id))
        ), reverse=True, key=lambda x: x[1]
    )
    materials = sorted(
        list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                             information_extractor.materials_lemmas.keys(), id))),
        reverse=True, key=lambda x: x[1])
    brands = sorted(
        list(set(information_extractor.find_closest_semantic(caption + userhandles, comments, tags, hashtags,
                                                             segmented_hashtags, 10,
                                                             information_extractor.companies, id))),
        reverse=True, key=lambda x: x[1])
    patterns = sorted(
        list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
                                                             information_extractor.patterns, id))),
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


def premap_post(row, args):
    """Sloppy/Quick text normalization"""
    if row._c0 is not None:
        id = row._c0.encode('utf-8', 'ignore').decode("utf-8")
    else:
        id = ""
    if row._c1 is not None:
        url = row._c1.encode('utf-8', 'ignore').decode("utf-8")
    else:
        url = ""
    if row._c2 is not None:
        comments = row._c2.encode('utf-8', 'ignore').decode("utf-8")
    else:
        comments = ""
    if row._c3 is not None:
        caption = row._c3.encode('utf-8', 'ignore').decode("utf-8")
    else:
        caption = ""
    if row._c4 is not None:
        tags = row._c4.encode('utf-8', 'ignore').decode("utf-8")
    else:
        tags = ""
    text = comments + caption + tags
    tokens = tknzr.tokenize(text)
    hashtags = [x for x in tokens if x.startswith("#")]
    segmented_hashtags = []
    styles = []
    for token in hashtags:
        if len(token) < 20:
            seg = segment(token.strip("#"))
            if "style" in seg or "styles" in seg:
                for w in seg:
                    if not w == "style" and not w == "styles":
                        styles.append((w, 100.0))
            segmented_hashtags.extend(seg)
    if styles > 8:
        styles = styles[0:8]
    links = []
    emojis = []
    userhandles = []
    for token in tokens:
        match = re.search("http://liketk.it/([^\s]+)", token)
        if match is not None:
            link = match.group(0)
            links.append(link)
        if token in emoji.UNICODE_EMOJI:
            emojis.append(token)
        if token.startswith("@"):
            userhandles.append(token)
    text = tokens
    comments = tknzr_strip_users.tokenize(comments)
    caption = tknzr_strip_users.tokenize(caption)
    tags = tknzr_strip_users.tokenize(tags)
    img_name = url.rsplit('/', 1)[-1]
    image_path = args.imagepath + img_name
    return pyspark.sql.Row(id=id, text=text, hashtags=hashtags, links=links,
                           comments=comments, segmented_hashtags=segmented_hashtags,
                           caption=caption, tags=tags, styles=styles, userhandles=userhandles, emojis=emojis, url=url,
                           image_path=image_path)


def analyze_user(information_extractor, sql, args):
    """ Analyzes a given user with semantic/syntactic similarities"""
    df = parse_raw(sql, args.input)
    count = df.count()
    print("number of posts: {0}".format(count))
    rdd = df.rdd.repartition(args.partitions)
    rdd = rdd.map(lambda x: premap_post(x, args))
    rdd = rdd.repartition(args.partitions)
    rdd = rdd.zipWithIndex()
    rdd = rdd.repartition(args.partitions)
    rdd = rdd.map(lambda (x, index): map_post(x, information_extractor, index, args))
    try:
        rdd.toDF().toJSON().saveAsTextFile(args.output)
    except:
        rdd.saveAsTextFile(args.output)


def main():
    """ Program entrypoint, orchestrates the pipeline"""
    args = parse_args()
    tfidf = create_tf_idf(args.input)
    information_extractor = InformationExtractor(args.vectors, read_gazetter(args.brands), read_gazetter(args.styles),
                                                 read_gazetter(args.materials), read_gazetter(args.items),
                                                 read_gazetter(args.probasebrands),
                                                 read_gazetter(args.probasematerials), read_gazetter(args.patterns),
                                                 read_gazetter(args.itemtopcategory), args.deepdetect, args.conf, tfidf)
    sc = pyspark.SparkContext(conf=sparkConf())
    sql = pyspark.SQLContext(sc)
    analyze_user(information_extractor, sql, args)


if __name__ == '__main__':
    main()
