import csv
import shutil
import os

csv_file = '/home/data1/zgp/code/maegr/es_all_98.70.csv'

query_folder = '/home/data1/zgp/code/data/data_reid/reid_challenge/query/'
gallery_folder = '/home/data1/zgp/code/data/data_reid/reid_challenge/gallery/'
targe_folder = '/home/data1/zgp/code/data/data_reid/query_gallery/'


os.makedirs(targe_folder,exist_ok=True)

with open (csv_file,'r') as file:
    reader = csv.reader(file)
    header=next(reader)
    for query_index,row in enumerate(reader):
        query_index = row[0]
        query_filename =  str(query_index).zfill(5)+'.jpeg'
        
        closest_gallery_filenames=[]
        gallery_indices = sorted(range(len(row[1:])), key=lambda i: float(row[i+1]))
        closest_gallery_indices=gallery_indices[:20]   
            
        for gallery_index in closest_gallery_indices:
            gallery_index = header[gallery_index+1]
            gallery_filename = str(gallery_index).zfill(5)+'.jpeg'
            closest_gallery_filenames.append(gallery_filename)  
        
        os.makedirs(targe_folder+query_index,exist_ok=True)
        shutil.copy(query_folder+query_filename,targe_folder+query_index)
        
        for  gallery_filename in closest_gallery_filenames:
            shutil.copy(gallery_folder+gallery_filename,targe_folder+query_index)     
        
        