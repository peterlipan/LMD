import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed)
target_root = './split/hyper-kvasir'
df = pd.read_csv("/mnt/ssd/li/kvasir/labeled-images/image-labels.csv")
# preserve the order of the classes and assign a consistent label to each class
classes = sorted(set(df['Finding'].values))
class2label = {find: label for label, find in enumerate(classes)}
df['labels'] = [class2label[item] for item in df['Finding']]
# shuffle the dataframe as it's stored uniformly regarding the findings
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

temp_df, test_df = train_test_split(df, test_size=0.1, random_state=seed, stratify=df['labels'])
train_df, val_df = train_test_split(temp_df, test_size=1 / 9, random_state=seed, stratify=temp_df['labels'])

if not os.path.exists(target_root):
    os.makedirs(target_root)

df.to_csv(os.path.join(target_root, "overall.csv"), index=False)
train_df.to_csv(os.path.join(target_root, "training.csv"), index=False)
test_df.to_csv(os.path.join(target_root, "testing.csv"), index=False)
val_df.to_csv(os.path.join(target_root, "validation.csv"), index=False)
