#!/bin/bash

export PYSPARK_PYTHON=/home/limmen/anaconda2/envs/data_exp/bin/python
export PYSPARK_DRIVER_PYTHON=/home/limmen/anaconda2/envs/data_exp/bin/python

$SPARK_HOME/sbin/start-master.sh

export MASTER=spark://limmen-MS-7823
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=7

$SPARK_HOME/sbin/start-slave.sh $MASTER:7077 --cores 1 --memory 10g

firefox http://127.0.0.1:8080 &

