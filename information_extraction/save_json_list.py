import pyspark
import pyspark.sql
import ast
import json

def parse_raw(sql):
    df = sql.read.json("data/rdd_backup_new_70k/test2/total.json").toDF()
    return df

def sparkConf():
    "Setup spark configuration, change this to run on cluster"
    conf = pyspark.SparkConf()
    return conf.setAppName("fashion_rec_data_setup").set("spark.hadoop.validateOutputSpecs", "false")
    #return conf \
    #    .setMaster("local[8]") \
    #    .setAppName("fashion_rec_data_setup") \
    #    .set("spark.hadoop.validateOutputSpecs", "false")

def main():
    sc = pyspark.SparkContext(conf=sparkConf())
    sql = pyspark.SQLContext(sc)
    df = parse_raw(sql)
    print "take 0 : {0}".format(df.take(1))
    jsonList = df.toJSON().collect()
    print "jsonList: {0}".format(jsonList[0])
    print type(jsonList)
    print len(jsonList)
    #analyze_user("test", information_extractor, sql)
    ids = set()
    jsonList2 = []
    for x in jsonList:
        x2 = ast.literal_eval(x)
        id = x2["id"]
        if not id in ids:
            jsonList2.append(x2)
        ids.add(id)
    print len(jsonList2)
    with open('data/json/data.json', 'w') as outfile:
        json.dump(jsonList2, outfile)


if __name__ == '__main__':
    main()