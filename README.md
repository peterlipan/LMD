# LMD
This is the official repository of our paper [**Long-tailed Medical Diagnosis with Relation-aware Representation Learning and Iterative Classifier Calibration**](https://arxiv.org/abs/2502.03238)

![framework](/assets/framework.png)

We improved the decoupling method for balanced medical image classification on long-tailed datasets as follows:

![framework](/assets/preliminary.png)

## Pipeline
1. Create a conda environment using the requirements file.
```bash
conda env create -n env_name -f environment.yaml
conda activate env_name
```

2. Download the ISIC2019LT, ISIC Archive, and Hyper-Kvasir by running the following scripts:
```bash
bash prepare_datasets/ISIC2019LT/download_ISIC2019LT.sh
bash prepare_datasets/ISIC_Archive/download_isic_archive.sh
bash prepare_datasets/hyper-kvasir/download_hyper_kvasir.sh
```

3. Modify the parameters in the yaml files under the folder [config](config).

4. Run the first and second stage of the LMD:
```bash
python stage1.py --config config/isic2019.yaml
python stage2.py --config config/isic2019.yaml
```

## Illustrations
![tsne1](/assets/Figure_4c.png)
![tsne2](/assets/Figure_5c.png)
As shown above, the LMD framework is capable of generating rich and balanced representations for long-tailed medical image classification.

## Citation
