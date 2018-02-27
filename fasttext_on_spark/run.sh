#!/bin/bash

$SPARK_HOME/bin/spark-submit \
    --master spark://limmen-MS-7823:7077 \
    --class "limmen.fasttext_on_spark.Main" \
    --conf spark.cores.max=8 \
    --conf spark.task.cpus=7 \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    --conf spark.rpc.message.maxSize=2000 \
    --executor-memory 8g \
    --driver-memory 8g \
/media/limmen/HDD/workspace/scala/fasttext_on_spark/target/scala-2.11/fasttext_on_spark-assembly-0.1.0-SNAPSHOT.jar --input "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/clean2_corpus.txt" --output "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/output" --cluster --partitions 5 --iterations 5 --saveparallel --dim 100 --windowsize 5 --algorithm "fasttext" --minn 3 --maxn 6 --norm

#/media/limmen/HDD/workspace/scala/fasttext_on_spark/target/scala-2.11/fasttext_on_spark-assembly-0.1.0-SNAPSHOT.jar --input "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/clean2_corpus.txt" --output "/media/limmen/HDD/workspace/scala/fasttext_on_spark/data/output" --cluster --partitions 20 --iterations 2 --saveparallel --dim 100 --windowsize 5 --algorithm "fasttext" --minn 3 --maxn 6 --norm
