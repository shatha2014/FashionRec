#!/bin/bash

/home/limmen/programs/spark-2.2.1-bin-hadoop2.7/bin/spark-submit \
--master spark://limmen-MS-7823:7077 \
--py-files /media/limmen/HDD/Dropbox/wordvec/ie/save_json_list.py \
--conf spark.cores.max=8 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--executor-memory 2g \
--driver-memory 2g \
/media/limmen/HDD/Dropbox/wordvec/ie/save_json_list.py
