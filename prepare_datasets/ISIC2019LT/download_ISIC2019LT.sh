#!/bin/bash
read -p "Please input the data dir: " data_dir

isic_dir="$data_dir/ISIC2019"
mkdir -p "$isic_dir"

wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

unzip -jn ISIC_2019_Training_Input.zip -d "$isic_dir"

mv ISIC_2019_Training_GroundTruth.csv "$isic_dir"

rm ISIC_2019*.zip
