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

"""
A faster, distributed version of TextAnalysis.py

Uses Spark for distribution. 
"""

schema = (StructType().add('id', IntegerType(), True)
          .add('comments', StringType(), True)
          .add('caption', StringType(), True)
          .add('tags', DoubleType(), True))

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

def filterOccurenceCount(iter):
    """Filter occurrences"""
    A = Counter(iter)
    return {x : A[x] for x in A if A[x] >= 3}.keys()

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
    return conf.setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false")
    #return conf \
    #    .setMaster("local[8]") \
    #    .setAppName("fashion_rec_data_setup") \
    #    .set("spark.hadoop.validateOutputSpecs", "false")

def parse_raw(sql, user):
    """ Read corpora with spark"""
    df = sql.read.csv(
        "data/corpora.csv", header=False, mode="DROPMALFORMED"
    )
    return df

def map_post(row, information_extractor, index):
    """Process post, semantic + syntactic similarity classification"""
    if index % 10 == 0:
        print ("Processing post with index {0}".format(index))
    tokens = row.text.split(" ")
    styles = sorted(list(set(information_extractor.find_closest_semantic(tokens, 10, information_extractor.styles_lemmas.keys()))), reverse=True, key=lambda x: x[1])
    items = sorted(list(set(information_extractor.find_closest_semantic(tokens, 10, information_extractor.items_lemmas.keys()))), reverse=True, key=lambda x: x[1])
    materials = sorted(list(set(information_extractor.find_closest_semantic(tokens, 10, information_extractor.materials_lemmas.keys()))), reverse=True, key=lambda x: x[1])
    brands = sorted(list(set(information_extractor.find_closest_semantic(tokens, 10, information_extractor.companies))), reverse=True, key=lambda x: x[1])
    patterns = sorted(list(set(information_extractor.find_closest_semantic(tokens, 10, information_extractor.patterns))), reverse=True, key=lambda x: x[1])
    ranked_brands = sorted(re_rank_brands(information_extractor, brands),reverse=True, key=lambda x: x[1])
    ranked_materials = sorted(re_rank_materials(information_extractor, materials),reverse=True, key=lambda x: x[1])
    return pyspark.sql.Row(id = row.id, brands = dict(ranked_brands), hashtags=row.hashtags, links = row.links,
                           materials = dict(ranked_materials), patterns =dict(patterns),
                           items = dict(items), styles= dict(styles))

def re_rank_materials(information_extractor, materials):
    """ Re-rank materials based on probase lookup"""
    ranked_materials = map(lambda material: material_rank_mapper(information_extractor, material), materials)
    return ranked_materials

def material_rank_mapper(information_extractor, material_):
    """ Re-rank materials based on probase lookup"""
    material, rank = material_
    factor_probase = lookup_material_probase(information_extractor, material, 10)
    return (material, factor_probase*rank)

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
    return (brand, rank*factor_probase)

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
        id = row._c0.encode('utf-8','ignore').decode("utf-8")
    else:
        id = ""
    if row._c1 is not None:
        comments = row._c1.encode('utf-8','ignore').decode("utf-8")
    else:
        comments = ""
    if row._c2 is not None:
        caption = row._c2.encode('utf-8','ignore').decode("utf-8")
    else:
        caption = ""
    if row._c3 is not None:
        tags = row._c3.encode('utf-8','ignore').decode("utf-8")
    else:
        tags = ""
    text = comments + caption + tags
    tokens = tknzr.tokenize(text)
    hashtags = [x for x in tokens if x.startswith("#")]
    links = []
    for token in tokens:
        match = re.search("http://liketk.it/([^\s]+)", token)
        if match is not None:
            link = match.group(0)
            links.append(link)
    tokens = filter(lambda x: x not in emoji.UNICODE_EMOJI, tokens)
    text = " ".join(tokens)
    return pyspark.sql.Row(id = id, text=text, hashtags=hashtags, links = links)

def analyze_user(user, information_extractor, sql):
    """ Analyzes a given user with semantic/syntactic similarities"""
    df = parse_raw(sql, user)
    rdd = df.rdd.map(lambda x: premap_post(x))
    rdd = rdd.zipWithIndex()
    rdd = rdd.map(lambda (x, index): map_post(x, information_extractor, index))
    rdd.toDF().toJSON().saveAsTextFile("data/rdd/test")

def main():
    """ Program entrypoint, orchestrates the pipeline"""
    materials = read_gazetter("./domain_data/material.csv")
    items = read_gazetter("./domain_data/items.csv")
    styles = read_gazetter("./domain_data/general_style.csv")
    companies = read_gazetter("./domain_data/companies.csv")
    probase_brands = read_gazetter("./domain_data/probase_brands.csv")
    probase_materials = read_gazetter("./domain_data/probase_materials.csv")
    brands_keywords_google_search = read_gazetter("./domain_data/brands_keywords_google_search.csv")
    materials_keywords_google_search = read_gazetter("./domain_data/materials_keywords_google_search.csv")
    colors = read_gazetter("./domain_data/colors.csv")
    patterns = read_gazetter("./domain_data/patterns.csv")
    information_extractor = InformationExtractor("./vectors/vectors.vec", companies, styles, materials, items,
                                                 brands_keywords_google_search, materials_keywords_google_search,
                                                 probase_brands, probase_materials, colors, patterns)
    users = os.listdir("data/ig/")
    print users
    sc = pyspark.SparkContext(conf=sparkConf())
    sql = pyspark.SQLContext(sc)
    analyze_user("test", information_extractor, sql)

if __name__ == '__main__':
    main()