# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn import set_config; set_config(display='diagram')
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from google.cloud import storage
import joblib


### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-577-hunt'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

# For storing model
STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # create distance pipeline
        dist_pipe = Pipeline([
                    ('dist_trans', DistanceTransformer()),
                    ('stdscaler', StandardScaler())])

        # create time pipeline
        time_pipe = Pipeline([
                    ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
                        ('distance',
                            dist_pipe,
                            ["pickup_latitude",
                             "pickup_longitude",
                             'dropoff_latitude',
                             'dropoff_longitude']),
                        ('time', time_pipe,
                         ['pickup_datetime'])],
                        remainder="drop")

        # Add linear regression model
        self.pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('linear_model', LinearRegression())])

    def train(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk necessary to upload it to storage
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # Initialise
    trainer = Trainer(X_train, y_train)

    # train the pipeline
    trainer.train()

    # evaluate the pipeline
    rmse = trainer.evaluate(X_val, y_val)

    print(f'RMSE is: {rmse}')

    #Saving model
    trainer.save_model()
