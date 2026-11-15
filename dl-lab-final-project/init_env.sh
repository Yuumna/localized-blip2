#!/bin/bash

USER=$(whoami)
TF_SHARED_FOLDER_HOST=hoos
REPO_NAME=dl-lab-final-project

# check if we are running on TFPool or GoogleCloud
if [ -d /project/dl2023s ]; then
    echo "Running on TFPool"
    
    # relocate cache directory
    CACHE_DIR=/project/dl2023s/$USER/$REPO_NAME/cache
    echo "New cache dir: $CACHE_DIR"
    if [ ! -d $CACHE_DIR ]; then
        mkdir -p $CACHE_DIR
    fi
    export TRANSFORMERS_CACHE=$CACHE_DIR
    export TORCH_HOME=$CACHE_DIR
    export FIFTYONE_DATABASE_DIR=$CACHE_DIR
    export FIFTYONE_DATASET_ZOO_DIR=$CACHE_DIR

    export DATASET_DIR=/project/dl2023s/$TF_SHARED_FOLDER_HOST/shared/datasets
    export BATCH_SIZE=8

elif [ -d /home/shared/ ]; then
    echo "Running on GoogleCloud"
    export DATASET_DIR=/home/shared/datasets
    export BATCH_SIZE=16

else
    echo "Running on local machine / unknown environment"
fi