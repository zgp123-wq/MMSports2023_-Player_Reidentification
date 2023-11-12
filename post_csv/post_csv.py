#/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/model/ViT-L-14_openai_colorjitter/fold-1_seed_1/challenge_dmat_rerank.csv

import csv
from collections import defaultdict

csv_file2 = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/98.21.csv"  # CSV文件路径
csv_file1 = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/vit_es_98.26.csv"  # CSV文件路径
csv_file3 = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/eva_es_97.66.csv"
csv_file4 = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/eva_97.59.csv"
csv_file5 = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/maegr/eva02_all_97.43.csv"

csv_files = [csv_file1, csv_file2, csv_file3,csv_file4,csv_file5]

out_files = "/home/data1/zgp/mmsport/2022-winners-player-reidentification-challenge-master/final/challenge_dmat_rerank.csv"
sequence_counts = defaultdict(int)

csv_counts = defaultdict(int)

with open(csv_files[0], 'r') as infile:
    reader = csv.reader(infile)
    header = next(reader)

with open(out_files, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

for row_number in range(1, 618):  

    for csv_file in csv_files:
        with open(csv_file, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)  

            for _ in range(row_number):
                next(reader)

            row = next(reader)  
          
            gallery_distances = [float(distance) for distance in row]
            gallery_indices = sorted(range(len(gallery_distances)), key=lambda i: gallery_distances[i], reverse=True)[:21]
            i=20
            for index in gallery_indices:
                sequence_counts[index] += 1+i
                i=i-1
    max_count_index = max(sequence_counts, key=lambda i: sequence_counts[i])

    
    max_count_csv = csv_files[max_count_index]
    csv_counts[max_count_index] +=1
    print(f"选择了 {max_count_csv} 的第 {row_number} 行作为新的CSV行")
    with open(max_count_csv, 'r') as infile, open(out_files, 'a', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for _ in range(row_number):
            next(reader)
        writer.writerow(next(reader))
    # 清零序列频数字典
    sequence_counts.clear()

print("已完成从三个CSV文件中选择频数最大的行的操作。")
print(f"选择了 {csv_counts[0]}个第一个csv作为的CSV的行")
print(f"选择了 {csv_counts[1]}个第一个csv作为的CSV的行")
print(f"选择了 {csv_counts[2]}个第一个csv作为的CSV的行")
print(f"选择了 {csv_counts[3]}个第一个csv作为的CSV的行")
print(f"选择了 {csv_counts[4]}个第一个csv作为的CSV的行")
