#!/bin/bash
wget "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"

mkdir /mnt/ssd/li/kvasir

unzip hyper-kvasir-labeled-images.zip -d /mnt/ssd/li/kvasir
rm hyper-kvasir-labeled-images.zip
