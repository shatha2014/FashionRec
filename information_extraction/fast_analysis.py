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

"""
A faster, distributed version of TextAnalysis.py

Uses Spark for distribution. 
"""

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


def sparkConf():
    "Setup spark configuration, change this to run on cluster"
    conf = pyspark.SparkConf()
    return conf.setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false").set("spark.executor.heartbeatInterval", "20s").set("spark.rpc.message.maxSize", "512").set("spark.kryoserializer.buffer.max", "1024")
    # return conf \
    #    .setMaster("local[8]") \
    #    .setAppName("fashion_rec_data_setup") \
    #    .set("spark.hadoop.validateOutputSpecs", "false")


def parse_raw(sql, user):
    """ Read corpora with spark"""
    df = sql.read.csv(
        "data/concat.csv", header=False, mode="DROPMALFORMED"
    )
    return df


def map_post(row, information_extractor, index):
    """Process post, semantic + syntactic similarity classification"""
    print("processing post {}".format(index))
    if index % 10 == 0:
        print ("Processing post with index {0}".format(index))
    text_clustering_res = text_clustering_LF(row, information_extractor)
    liketkit_classes = liktekit_LF(row,information_extractor)
    google_vision_classes = google_vision_LF(row, information_extractor)
    deep_detect_classes = deep_detect_lookup(row, information_extractor)
    clarifai_classes = clarifai_lookup(row, information_extractor)
    deepomatic_classes = deepomatic_lookup(row, information_extractor)
    row = pyspark.sql.Row(id=row.id, hashtags=row.hashtags, links=row.links, text_clustering=text_clustering_res,
                          liketkit_classification=liketkit_classes,
                          google_vision_classification=google_vision_classes,
                          deep_detect_classification=deep_detect_classes,
                          clarifai_classification=clarifai_classes,
                          deepomatic_classification=deepomatic_classes,
                          url=row.url)
    #print("row construction done, returning {}".format(row))
    return row

def google_vision_LF(row, information_extractor):
    """ Analyze image with Google vision API """
    item_candidates = information_extractor.google_vision_lookup(row.image_path)
    #print("google mapping candidates: {}".format(item_candidates))
    items = information_extractor.map_candidates_to_ontology(item_candidates)
    #materials = information_extractor.map_candidates_to_ontology(item_candidates,information_extractor.materials_lemmas.keys(),10)
    #patterns = information_extractor.map_candidates_to_ontology(item_candidates,information_extractor.patterns,10)
    #print("result: {}".format(items))
    google_vision_classes = {}
    google_vision_classes["items"] = dict(items)
    #google_vision_classes["materials"] = dict(materials)
    #google_vision_classes["patterns"] = dict(patterns)
    return google_vision_classes

def deep_detect_lookup(row, information_extractor):
    """ Analyze image with deepdetect """
    items_and_fabrics = information_extractor.deep_detect_lookup(row.url)
    #print("deep_detect mapping candidates: {}".format(items_and_fabrics))
    items = information_extractor.map_candidates_to_ontology(items_and_fabrics["items"])
    #materials = information_extractor.map_candidates_to_ontology(items_and_fabrics["fabrics"],information_extractor.materials_lemmas.keys(),10)
    #print("result: {}".format(items))
    #print("result: {}".format(materials))
    deep_detect_classes = {}
    deep_detect_classes["items"] = dict(items)
    #deep_detect_classes["materials"] = dict(materials)
    return deep_detect_classes

def deepomatic_lookup(row, information_extractor):
    """ Analyze image with deepomatic """
    candidates = information_extractor.deepomatic_lookup(row.url)
    #print("deepomatic mapping candidates: {}".format(candidates))
    items = information_extractor.map_candidates_to_ontology(candidates)
    #materials = information_extractor.map_candidates_to_ontology(candidates,information_extractor.materials_lemmas.keys(),10)
    #patterns = information_extractor.map_candidates_to_ontology(candidates,information_extractor.patterns,10)
    #print("result: {}".format(items))
    deepomatic_classes = {}
    deepomatic_classes["items"] = dict(items)
    #deepomatic_classes["materials"] = dict(materials)
    #deepomatic_classes["patterns"] = dict(patterns)
    return deepomatic_classes

def clarifai_lookup(row, information_extractor):
    """ Analyze image with clarifai"""
    candidates = information_extractor.clarifai_lookup(row.url)
    #print("clarifai mapping candidates: {}".format(candidates))
    items = information_extractor.map_candidates_to_ontology(candidates)
    #materials = information_extractor.map_candidates_to_ontology(candidates,information_extractor.materials_lemmas.keys(),10)
    #patterns = information_extractor.map_candidates_to_ontology(candidates,information_extractor.patterns,10)
    #print("result: {}".format(items))
    clarifai_classes = {}
    clarifai_classes["items"] = dict(items)
    #clarifai_classes["materials"] = dict(materials)
    #clarifai_classes["patterns"] = dict(patterns)
    return clarifai_classes

def emoji_LF(row,information_extractor):
    """ Analyze image based on emojis"""

    emoji_classes = information_extractor.emoji_classification(row.emojis, 10)
    #print("emoji classes: {}".format(emoji_classes))
    return dict(emoji_classes)

def liktekit_LF(row, information_extractor):
    """ Analyze image based on liketkit link scraping"""
    text = []
    for link in row.links:
        text.extend(information_extractor.liketkit_classification(link))
    #print("liketkit text: {}".format(text))
    liktekit_classes = {}
    # liktekit_classes["styles"] = dict(sorted(list(set(
    #     information_extractor.find_closest_semantic([], text, [], [], [], 10,
    #                                                 information_extractor.styles_lemmas.keys()))),
    #     reverse=True, key=lambda x: x[1]))
    liktekit_classes["items"] = dict(information_extractor.find_closest_semantic_hieararchy(text, [], [], []))
    # liktekit_classes["materials"] = dict(sorted(
    #     list(set(information_extractor.find_closest_semantic([], text, [], [], [], 10,
    #                                                          information_extractor.materials_lemmas.keys()))),
    #     reverse=True, key=lambda x: x[1]))
    # liktekit_classes["brands"] = dict(sorted(
    #     list(set(information_extractor.find_closest_semantic([], text, [], [], [], 10,
    #                                                          information_extractor.companies))),
    #     reverse=True, key=lambda x: x[1]))
    # liktekit_classes["patterns"] = dict(sorted(
    #     list(set(information_extractor.find_closest_semantic([], text, [], [], [], 10,
    #                                                          information_extractor.patterns))),
    #     reverse=True, key=lambda x: x[1]))
    return liktekit_classes

def text_clustering_LF(row, information_extractor):
    """ Analyze image based on semantic text clustering """
    caption = row.caption
    comments = row.comments
    hashtags = row.hashtags
    segmented_hashtags = row.segmented_hashtags
    # hashtags.extend(row.segmented_hashtags)
    tags = row.tags
    #styles1 = row.styles
    #numStyles = 10 - len(styles1)
    # styles = sorted(list(set(
    #     information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, numStyles,
    #                                                 information_extractor.styles_lemmas.keys()))),
    #     reverse=True, key=lambda x: x[1])
    # styles.extend(styles1)
    items = information_extractor.find_closest_semantic_hieararchy(caption, comments, tags, hashtags)
    # materials = sorted(
    #     list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
    #                                                          information_extractor.materials_lemmas.keys()))),
    #     reverse=True, key=lambda x: x[1])
    # brands = sorted(
    #     list(set(information_extractor.find_closest_semantic(caption+row.userhandles, comments, tags, hashtags, segmented_hashtags, 10,
    #                                                          information_extractor.companies))),
    #     reverse=True, key=lambda x: x[1])
    # patterns = sorted(
    #     list(set(information_extractor.find_closest_semantic(caption, comments, tags, hashtags, segmented_hashtags, 10,
    #                                                          information_extractor.patterns))),
    #     reverse=True, key=lambda x: x[1])
    # ranked_materials = sorted(re_rank_materials(information_extractor, materials), reverse=True, key=lambda x: x[1])
    # ranked_brands = sorted(re_rank_brands(information_extractor, brands), reverse=True, key=lambda x: x[1])
    # ranked_brands_2 = []
    # items_temp = map(lambda x: x[0], items[:3])
    # styles_temp = map(lambda x: x[0], styles[:3])
    # materials_temp = map(lambda x: x[0], ranked_materials[:3])
    # styles_temp.extend(materials_temp)
    # extra_brands = sorted(information_extractor.find_closest_semantic(items_temp, styles_temp, [], [], [], 10,information_extractor.companies),reverse=True, key=lambda x: x[1])
    # for brand in ranked_brands:
    #     if brand[1] == 0:
    #         newbrand = extra_brands.pop(0)
    #         ranked_brands_2.append(newbrand)
    #     else:
    #         ranked_brands_2.append(brand)
    text_clustering_LF = {}
    # text_clustering_LF["brands"] = dict(ranked_brands_2)
    # text_clustering_LF["patterns"] = dict(patterns)
    # text_clustering_LF["materials"] = dict(materials)
    text_clustering_LF["items"] = dict(items)
    #text_clustering_LF["styles"] = dict(styles)
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


def premap_post(row):
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
    #print("styles in hashtags: {}".format(styles))
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
    image_path = "/media/limmen/HDD/Dropbox/wordvec/ie/data/concat/" + img_name
    return pyspark.sql.Row(id=id, text=text, hashtags=hashtags, links=links,
                           comments=comments, segmented_hashtags=segmented_hashtags,
                           caption=caption, tags=tags, styles=styles, userhandles=userhandles, emojis=emojis, url=url, image_path = image_path)


def analyze_user(user, information_extractor, sql):
    """ Analyzes a given user with semantic/syntactic similarities"""
    df = parse_raw(sql, user)
    count = df.count()
    print("number of posts: {0}".format(count))
    rdd = df.rdd.repartition(30)
    rdd = rdd.map(lambda x: premap_post(x))
    print("first map done: {0}".format(count))
    rdd = rdd.repartition(30)
    rdd = rdd.zipWithIndex()
    rdd = rdd.repartition(30)
    print("zip done: {0}".format(count))
    rdd = rdd.repartition(30).map(lambda (x, index): map_post(x, information_extractor, index))
    print("second map done: {0}".format(count))
    rdd.toDF().toJSON().saveAsTextFile("data/rdd/test")


def main():
    """ Program entrypoint, orchestrates the pipeline"""
    materials = read_gazetter("./domain_data/material.csv")
    items = read_gazetter("./domain_data/items.csv")
    styles = read_gazetter("./domain_data/general_style2.csv")
    companies = read_gazetter("./domain_data/companies.csv")
    probase_brands = read_gazetter("./domain_data/probase_brands.csv")
    probase_materials = read_gazetter("./domain_data/probase_materials.csv")
    brands_keywords_google_search = read_gazetter("./domain_data/brands_keywords_google_search.csv")
    materials_keywords_google_search = read_gazetter("./domain_data/materials_keywords_google_search.csv")
    colors = read_gazetter("./domain_data/colors.csv")
    patterns = read_gazetter("./domain_data/patterns.csv")
    hierarchy = read_gazetter("./domain_data/hierarchical/hieararchical.csv")
    information_extractor = InformationExtractor("./vectors/vectors.vec", companies, styles, materials, items,
                                                 brands_keywords_google_search, materials_keywords_google_search,
                                                 probase_brands, probase_materials, colors, patterns,hierarchy)
    sc = pyspark.SparkContext(conf=sparkConf())
    sql = pyspark.SQLContext(sc)
    analyze_user("test", information_extractor, sql)


if __name__ == '__main__':
    main()
