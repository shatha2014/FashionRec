#!/bin/bash

$SPARK_HOME/sbin/start-master.sh

export MASTER=spark://limmen-MS-7823
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=3

$SPARK_HOME/sbin/start-slave.sh $MASTER:7077 --cores 7 --memory 8g

firefox http://127.0.0.1:8080 &
