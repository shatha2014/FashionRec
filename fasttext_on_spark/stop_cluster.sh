#!/bin/bash

$SPARK_HOME/sbin/stop-all.sh

$SPARK_HOME/sbin/stop-slaves.sh

kill $(ps aux | grep "spark" | awk "{print $2}")
