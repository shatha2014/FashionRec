#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--master spark://limmen-MS-7823:7077 \
--py-files /media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/fast_analysis.py,/media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/InformationExtraction.py,/media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/deepomatic.py,/media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/dd_client.py,/media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/dd_bench.py \
--conf spark.cores.max=8 \
--conf spark.task.cpus=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--executor-memory 10g \
--driver-memory 5g \
/media/limmen/HDD/workspace/fashion_rec/FashionRec/information_extraction/fast_analysis.py --input data/sample_user.csv --output output/sample_user --conf ./conf/conf.json --textanalysis