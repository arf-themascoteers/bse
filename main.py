from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import os

TEST = False
results_file = "results.txt"
file = "lucas.csv"
if TEST:
    file = "lucas_min.csv"
data = pd.read_csv(f"../data/{file}")


def get_algorithm(name):
    if name == "lr":
        return LinearRegression()
    if name == "svr":
        return SVR(C=100, kernel='rbf', gamma=1)
    if name == "nn":
        return MLPRegressor()
    if name == "rf":
        return RandomForestRegressor()


def get_results():
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return pd.DataFrame(columns=["skip","n_bands","case_name","algorithm","train_size","R^2","bands"])


def run_case(algorithm,train_size,skip=0,case_name=None,bands=None):
    model = get_algorithm(algorithm)
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=42)
    if bands is None:
        bands = [i for i in range(0,len(data.columns)-1,1)]
    bands = sorted(bands)
    if case_name is None:
        case_name = f"{train_size}"
        if skip > 0:
            case_name += f"_{skip}"
        case_name += f"_{algorithm}"
    bands_str = "|".join(bands)

    X_train = train_data[:,bands]
    X_test = test_data[:,bands]
    y_train = train_data[:,0]
    y_test = test_data[:,0]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results = get_results()
    results.loc[len(results)] = [skip,len(bands),case_name,algorithm,train_size,r2,bands_str]
    results.to_csv(results_file, index=False)


if __name__ == "__main__":
    algorithm = ["lr"]
    skip = [10,50,90,130,170,210,250,290,330,370,410,450,490,530]
    train_size = [1,11,21,31,41,51,61,71,81,91]
    for s in skip:
        for t in train_size:
            run_case(algorithm,t,s)