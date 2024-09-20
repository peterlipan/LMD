#!/bin/bash
read -p "Please input the data dir: " data_dir

kvasir_dir="$data_dir/kvasir"
mkdir -p "$kvasir_dir"

wget "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"

unzip hyper-kvasir-labeled-images.zip -d "$kvasir_dir"

rm hyper-kvasir-labeled-images.zip
