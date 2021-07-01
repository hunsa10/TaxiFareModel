# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import set_config; set_config(display='diagram')
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


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

        # Add linear regression model to complete pipeline
        # Add the model of your choice to the pipeline
        self.pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('linear_model', LinearRegression())])

    def train(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


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

    # build pipeline
    trainer.set_pipeline()

    # train the pipeline
    trainer.train()

    # evaluate the pipeline
    rmse = trainer.evaluate(X_val, y_val)

    print(f'RMSE is: {rmse}')
