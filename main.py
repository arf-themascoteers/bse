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
results_file = "results.csv"
file = "lucas.csv"
if TEST:
    file = "lucas_min.csv"
data = pd.read_csv(f"{file}").to_numpy()


def get_algorithm(name):
    if name == "lr":
        return LinearRegression()
    if name == "svr":
        return SVR(C=100, kernel='rbf', gamma=1)
    if name == "mlp":
        return MLPRegressor()
    if name == "rf":
        return RandomForestRegressor()


def get_results(result_output):
    if os.path.exists(result_output):
        return pd.read_csv(result_output)
    return pd.DataFrame(columns=["skip","n_bands","case_name","algorithm","train_size","R^2","bands"])


def run_case(algorithm,train_size,skip=0,case_name=None,bands=None,result_output=None):
    if result_output is None:
        result_output = results_file
    model = get_algorithm(algorithm)
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=42)
    if bands is None:
        bands = [i for i in range(0,data.shape[1]-1,skip)]
    bands = sorted(bands)
    if case_name is None:
        case_name = f"{train_size}"
        if skip > 0:
            case_name += f"_{skip}"
        case_name += f"_{algorithm}"
    bands_str = "|".join([str(b) for b in bands])

    X_train = train_data[:,bands]
    X_test = test_data[:,bands]
    y_train = train_data[:,-1]
    y_test = test_data[:,-1]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2 = round(r2,3)
    if r2 <0 :
        r2 = 0
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    results = get_results(result_output)
    results.loc[len(results)] = [skip,len(bands),case_name,algorithm,train_size,r2,bands_str]
    results.to_csv(result_output, index=False)


if __name__ == "__main__":
    algorithm = ["lr","svr","rf","mlp"]
    skip = [1,10,50,90,130,170,210,250,290,330,370,410,450,490,530]
    train_size = [0.01,0.11,0.21,0.31,0.41,0.51,0.61,0.71,0.81,0.91]
    for a in algorithm:
        for s in skip:
            for t in train_size:
                run_case(a,t,s)

    bands = [0,45,334,430,657,1367,1976,2876,3013,3091,3397,3614,3821,3887,4199]
    for a in algorithm:
        for t in train_size:
            run_case(a,t, case_name=f"SPA_{t}_{a}",bands=bands)