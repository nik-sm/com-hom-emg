#!/usr/bin/env bash
set -euo pipefail

script_path=$(dirname "$(readlink -f $0)")
data_dir="$script_path/../data"
mkdir -p $data_dir
cd $data_dir

dataset_url="https://zenodo.org/records/10359729/files/combination-gesture-dataset.zip?download=1"

echo "Downloading dataset from: $dataset_url to: $data_dir"
curl $dataset_url --output combination-gesture-dataset.zip 

echo "Unzip"
unzip combination-gesture-dataset.zip
