#!/bin/bash
read -p "Please input the data dir: " data_dir

isic_dir="$data_dir/ISIC_Archive"
mkdir -p "$isic_dir"

# NV - nevus 12875
./isic image download --search 'diagnosis:"nevus"' --limit 12875 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/NV.csv"

# MEL - melanoma 4522
./isic image download --search 'diagnosis:"melanoma"' --limit 4522 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/MEL.csv"

# BCC - basal cell carcinoma 3393
./isic image download --search 'diagnosis:"basal cell carcinoma"' --limit 3393 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/BCC.csv"

# SK - seborrheic keratosis 1464
./isic image download --search 'diagnosis:"seborrheic keratosis"' --limit 1464 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/SK.csv"

# AK - actinic keratosis 869
./isic image download --search 'diagnosis:"actinic keratosis"' --limit 869 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/AK.csv"

# SCC - squamous cell carcinoma 656
./isic image download --search 'diagnosis:"squamous cell carcinoma"' --limit 656 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/SCC.csv"

# BKL - pigmented benign keratosis 384
./isic image download --search 'diagnosis:"pigmented benign keratosis"' --limit 384 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/BKL.csv"

# SL - solar lentigo 270
./isic image download --search 'diagnosis:"solar lentigo"' --limit 270 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/SL.csv"

# VASC - vascular lesion 253
./isic image download --search 'diagnosis:"vascular lesion"' --limit 253 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/VASC.csv"

# DF - dermatofibroma 246
./isic image download --search 'diagnosis:"dermatofibroma"' --limit 246 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/DF.csv"

# LK - lichenoid keratosis 32
./isic image download --search 'diagnosis:"lichenoid keratosis"' --limit 32 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/LK.csv"

# LS - lentigo simplex 27
./isic image download --search 'diagnosis:"lentigo simplex"' --limit 27 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/LS.csv"

# AN - angioma 15
./isic image download --search 'diagnosis:"angioma"' --limit 15 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/AN.csv"

# AMP - atypical melanocytic proliferation 14
./isic image download --search 'diagnosis:"atypical melanocytic proliferation"' --limit 14 "$isic_dir"
mv "$isic_dir/metadata.csv" "$isic_dir/AMP.csv"
