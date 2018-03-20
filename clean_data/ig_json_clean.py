#!/usr/bin/env python

"""
Python script for parsing IG-users in JSON formats with spark and
write comments and  captions to CSV or TSV files, one per user.
Also outputs corpus files with raw text that later can be used for training word embeddings (still normalization required)

Example commands to run it:

python ig_json_clean.py --input data --format tsv --p --output cleaned
python ig_json_clean.py --input data --format tsv --output cleaned -d
python ig_json_clean.py --input data --format csv --output cleaned -d
python ig_json_clean.py --input data --format csv --output cleaned
"""

import pyspark
import pyspark.sql
import os
import argparse
import numpy as np
import shutil

def sparkConf():
    "Setup spark configuration, change this to run on cluster"
    #conf = pyspark.SparkConf().setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false")
    return pyspark.SparkConf().setMaster("local[*]") \
        .setAppName("fashion_rec_data_setup") \
        .set("spark.hadoop.validateOutputSpecs", "false").set("spark.driver.maxResultSize", "8g")


def parse_raw(sqlContext, input, user):
    """Parses the raw json for a user"""
    df = sqlContext.read.json(input + "/" + user + "/" + user + ".json", multiLine=True)
    return df

def mapRow(row):
    """Mapper to convert json row into a row with only comments and caption in string formats"""
    commentsRow = row.comments
    captionRow = row.caption
    comments = commentsRow.data  # select comments
    textComments = " ".join([x.text for x in comments])  # remove metadata from comments
    if hasattr(captionRow, "edges"):
        captions = captionRow.edges
        textCaptions = " ".join([x.node.text for x in captions])
    if hasattr(captionRow, "text"):
        textCaptions = captionRow.text
    if not row.tags is None:
        tags = " ".join([x for x in row.tags])
    else:
        tags = ""
    textComments = textComments.replace("\n", " ")
    textComments = textComments.replace("\t", " ")
    textComments = textComments.replace(",", " ")
    textCaptions = textCaptions.replace("\n", " ")
    textCaptions = textCaptions.replace("\t", " ")
    textCaptions = textCaptions.replace(",", " ")
    tags = tags.replace("\n", " ")
    tags = tags.replace("\t", " ")
    tags = tags.replace(",", " ")
    if len(row.urls) > 0:
        url = row.urls[0]
    else:
        url = "missing-url"
    id = row.id
    return pyspark.sql.Row(comments=textComments, caption=textCaptions, tags=tags, id=id, url=url)


def parse_args():
    """Parses the commandline arguments with argparse"""
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-f", "--format", help="output format: (csv|tsv|json)", choices=["csv", "tsv", "json"],
                        default="tsv")
    parser.add_argument("-p", "--parallelized", help="save output in parallelized or single file format",
                        action="store_true")
    parser.add_argument("-i", "--input", help="folder where input documents are", default="data")
    parser.add_argument("-o", "--output", help="folder where output documents are", default="cleaned")
    parser.add_argument("-d", "--documentformat", help="combine all features into a single text per post",
                        action="store_true")
    parser.add_argument("-pa", "--partitions", help="number of spark partitions",
                    default=1)
    args = parser.parse_args()
    return args


def writeToFile(rdd, parallelized, output, user, format, features):
    """Writes the processed RDD to output file"""
    fileEnd = "." + format
    output_path = output + "/ig/" + user + "/" + user + fileEnd
    if parallelized:
        rdd.saveAsTextFile(output_path)
    else:
        arr = np.array(rdd.collect())
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'w+') as tsvfile:
            for row in arr:
                if format == "json":
                    tsvfile.write(row.encode("utf-8", errors='ignore') + "\n")
                else:
                    tsvfile.write(row + "\n")
        if not format == "json":
            output_path = output + "/ig/" + user + "/" + user + ".txt"
            saveCorpusFile(output_path, arr, format, features)


def format(df, format, features):
    """Formats the RDD pre-output"""

    def mapDocFormat(row):
        if format == "csv":
            return (row.id.encode("utf-8", errors='ignore') + "," + " ".join(
                [x.encode("utf-8", errors='ignore') for x in [row.comments, row.caption, row.tags]])).lower()
        if format == "tsv":
            return (row.id.encode("utf-8") + "\t" + " ".join(
                [x.encode("utf-8", errors='ignore') for x in [row.comments, row.caption, row.tags]])).lower()
        if format == "json":
            return pyspark.sql.Row(doc=row.id.encode("utf-8", errors='ignore').lower(),
                                   text=(''.join([x.encode("utf-8", errors='ignore') for x in row]))).lower()

    if features:
        df = df.rdd.map(mapDocFormat)
        if format == "json":
            return df.toDF().toJSON()
        else:
            return df
    else:
        if format == "csv":
            return df.rdd.map(lambda row: (','.join(
                [x.encode("utf-8", errors='ignore') for x in [row.id, row.url, row.comments, row.caption, row.tags]])).lower())
        if format == "tsv":
            return df.rdd.map(lambda row: ('\t'.join(
                [x.encode("utf-8", errors='ignore') for x in [row.id, row.url, row.comments, row.caption, row.tags]])).lower())
        if format == "json":
            return df.toJSON()

def saveCorpusFile(output_path, arr, format, features):
    """ Saves the corpus file to the given output path"""
    def rowMap(x):
        if format == "csv":
            if features:
                x = x.split(",")[1]
            else:
                parts = x.split(",")
                parts.pop(0)
                x = " ".join(parts)
            return x.replace(",", " ")
        if format == "tsv":
            if features:
                x = x.split("\t")[1]
            else:
                parts = x.split("\t")
                parts.pop(0)
                x = " ".join(parts)
            return x.replace("\t", " ")

    arr_corpus = map(lambda x: rowMap(x), arr)
    with open(output_path, 'w+') as corpusfile:
        for row in arr_corpus:
            corpusfile.write(row + "\n")


def select(df):
    """Selects the columns from the raw JSON RDD"""
    commentsCols = ["comments"]
    captionCols = ["edge_media_to_caption", "caption"]
    tagsCols = ["tags"]
    idCol = "id"
    urlsCol = "urls"
    commentCol = filter(lambda commentCol: commentCol in df.columns, commentsCols)[0]
    captionCol = filter(lambda captionCol: captionCol in df.columns, captionCols)[0]
    tagCol = filter(lambda tagCol: tagCol in df.columns, tagsCols)[0]
    df = df.select([commentCol, captionCol, tagCol, idCol, urlsCol])
    df = df.selectExpr(commentCol + " as comments", captionCol + " as caption", tagCol + " as tags", idCol + " as id",
                       urlsCol + " as urls")
    return df

def parseUser(user, args, sql, numPartitions):
    """Parses a single user JSON file and writes it as CSV or TSV"""
    print("parsing user {}".format(user))
    df = parse_raw(sql, args.input, user)
    df.repartition(numPartitions)
    df = select(df)
    df = df.rdd.map(lambda x: mapRow(x)).toDF()
    df = format(df, args.format, args.documentformat)
    writeToFile(df, args.parallelized, args.output, user, args.format, args.documentformat)


def cleanOutputDir(output):
    """Cleans the output directory to make room for new outputs"""
    if os.path.exists(output) and os.path.isdir(output):
        shutil.rmtree(output)

def append_corpus(output):
    """ Appends corpus files for all users to a single corpus file"""
    files = []
    output_path = output + "/ig/" + "ig_corpus.txt"
    for root, directories, filenames in os.walk(output + "/ig/"):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    corpusfiles = filter(lambda x: ".txt" in x, files)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w+") as corpusFile:
        for file in corpusfiles:
            fileH = open(file, "r")
            corpusFile.write(fileH.read())

def corpora_stats(output):
    """ Computes some basic statistics about the output corpus"""
    igFiles = []
    for root, directories, filenames in os.walk(output + "/ig/"):
        for filename in filenames:
            igFiles.append(os.path.join(root, filename))
    igFiles = filter(lambda x: ".txt" in x, igFiles)
    words = []
    for file in igFiles:
        fileH = open(file, "r")
        words = words + fileH.read().split(" ")
    print("Number of words in IG corpus: {}".format(len(words)))
    print("Vocabulary size of IG corpus: {}".format(len(set(words))))


def main():
    """
    Main function, orchestrates the pipeline.
    Creates the spark context, parses arguments, and parses all users.
    """
    sc = pyspark.SparkContext(conf=sparkConf())
    sql = pyspark.SQLContext(sc)
    args = parse_args()
    cleanOutputDir(args.output)
    users = os.listdir(args.input)
    map(lambda user: parseUser(user, args, sql, args.partitions), users)
    corpora_stats(args.output)
    append_corpus(args.output)


if __name__ == '__main__':
    main()