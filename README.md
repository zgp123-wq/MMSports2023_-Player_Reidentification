
# Top 2 solution for MMSports2023_-Player_Reidentification
Top 2  solution for MMSports2023_-Player_Reidentification

### Requirements
* Python 3.8
* cuda 11.7
* timm 0.8.17
* pytorch 1.13.1
* torchaudio 0.13.1
* torchvision 0.14.1

### About the Code

#### 1. Prepare Data
Download the competition data 
```
python download_data.py 
```

#### 2. Creating Data Frames
```
python preprocess_data.py
```

#### 3. Train the Model
```
python train.py 
```

#### 4. Feature Extraction
Here the feature extraction is performed on our trained model for query and gallery.
```
python  fea.py 
```

#### 5. Feature Enhancement
Feature enhancement is performed separately for the trained model.We use the pyretri repository（https://github.com/PyRetri/PyRetri/tree/master/pyretri） ，feature enhancement is performed on the features that have been extracted.
```
python post_process.py 
```


#### 6. Model Fusion
Fusion of the above trained models.
```
python model_fuse.py
```



