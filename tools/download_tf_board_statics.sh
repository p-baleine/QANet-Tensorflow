#!/bin/bash

RUN="$1"
OUT=${2:-.}
URL="http://localhost:6006/data/plugin/scalars/scalars"

curl -o ${OUT}/run_${RUN}_train_loss.csv \
     "${URL}?run=${RUN}&tag=train_loss&format=csv"
curl -o ${OUT}/run_${RUN}_train_acc_p1.csv \
     "${URL}?run=${RUN}&tag=train_acc_p1&format=csv"
curl -o ${OUT}/run_${RUN}_train_acc_p2.csv \
     "${URL}?run=${RUN}&tag=train_acc_p2&format=csv"

curl -o ${OUT}/run_${RUN}_dev_loss.csv \
     "${URL}?run=${RUN}&tag=dev_loss&format=csv"
curl -o ${OUT}/run_${RUN}_dev_acc_p1.csv \
     "${URL}?run=${RUN}&tag=dev_acc_p1&format=csv"
curl -o ${OUT}/run_${RUN}_dev_acc_p2.csv \
     "${URL}?run=${RUN}&tag=dev_acc_p2&format=csv"
