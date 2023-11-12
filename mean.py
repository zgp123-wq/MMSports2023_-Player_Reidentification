import csv
import pandas as pd
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

from clipreid.transforms import get_transforms
from clipreid.dataset import ChallengeDataset
from clipreid.evaluator import write_mat_csv
from clipreid.utils import print_line


csv1="/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/98.21.csv"
csv2="/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/vit_es_98.26.csv"

csv_files = [csv1, csv2]
dist_matrix_rerank_list = []

img_size = (224,224)
mean=(0.48145466, 0.4578275, 0.40821073)
std=(0.26862954, 0.26130258, 0.27577711)
    
val_transforms, train_transforms = get_transforms(img_size, mean, std)
df_challenge = pd.read_csv("./data/data_reid/challenge_df.csv")
challenge_dataset = ChallengeDataset(df=df_challenge,
                                         image_transforms=val_transforms)
challenge_loader = DataLoader(challenge_dataset,
                                  batch_size=64,
                                  num_workers=0,
                                  shuffle=False,
                                  pin_memory=True)
for csv_file in csv_files:
    # 读取 CSV 文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        # 提取距离矩阵数据（去除第一行和第一列），并转换为浮点型
        dist_matrix = [[float(value) for value in row[1:]] for row in rows[1:]]

        # 将距离矩阵转换为二维数组（array）
        dist_matrix_array = np.array(dist_matrix)
        dist_matrix_rerank_list.append(dist_matrix_array)

if len(dist_matrix_rerank_list) > 1:
    
    print_line(name="Ensemble", length=80)
    
   
    # with re-ranking
    dist_matrix_rerank_ensemble = np.stack(dist_matrix_rerank_list, axis=0).mean(0)
    save_path = '/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/final/challenge_dmat_rerank_ensemble.csv'
    print("write distance matrix:", save_path)
    write_mat_csv(save_path,
                  dist_matrix_rerank_ensemble,
                  challenge_dataset.query,
                  challenge_dataset.gallery)

