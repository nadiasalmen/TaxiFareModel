# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
import joblib
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y
        self.pipeline = None

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
              "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
              'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())])
        return pipe

    # def holdout(self):
    #     """
    #       Split the dataset in train and test
    #     """
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        # joblib.dump(self.pipeline, 'model.joblib')

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
