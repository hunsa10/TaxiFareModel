### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-577-hunt'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

# For storing model
STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'
