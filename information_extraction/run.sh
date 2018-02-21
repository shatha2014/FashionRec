#!/bin/bash

export PYSPARK_PYTHON=/home/limmen/anaconda2/envs/data_exp/bin/python
export PYSPARK_DRIVER_PYTHON=/home/limmen/anaconda2/envs/data_exp/bin/python

$SPARK_HOME/sbin/start-master.sh

export MASTER=spark://limmen-MS-7823
export SPARK_WORKER_INSTANCES=7
export CORES_PER_WORKER=1

$SPARK_HOME/sbin/start-slave.sh $MASTER:7077 --cores 1 --memory 2g

firefox http://127.0.0.1:8080 &

$SPARK_HOME/bin/spark-submit \
--master spark://limmen-MS-7823:7077 \
--py-files /media/limmen/HDD/Dropbox/wordvec/ie/fast_analysis.py,/media/limmen/HDD/Dropbox/wordvec/ie/InformationExtraction.py \
--conf spark.cores.max=8 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--executor-memory 2g \
--driver-memory 2g \
/media/limmen/HDD/Dropbox/wordvec/ie/fast_analysis.py
