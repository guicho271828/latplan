#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

def analyse_linear(data,excluded=[]):
    cols = data.columns.difference([ "ratio", ]+ excluded)

    x = data[cols]
    # x = (x-x.mean())/x.std()
    x = (x-x.min())/(x.max()-x.min())

    y = data["ratio"]

    model = LinearRegression()
    model.fit(x, y)
    coef = pd.DataFrame(model.coef_.reshape(1,-1), columns=cols).transpose()
    R2 = model.score(x, y)
    return coef, R2


def analyse(data,modelclass=Lasso,regularizations=[0.1,0.01,0.001],excluded=[]):
    cols = data.columns.difference([ "ratio", "elbo", ]+ excluded)

    x = data[cols]

    log_x = np.log(data[cols])
    log_x.columns = [ "log_"+col for col in log_x.columns]

    x = pd.merge(x, log_x, left_index=True, right_index=True)
    
    # x = (x-x.mean())/x.std()
    x = (x-x.min())/(x.max()-x.min())


    y = data["ratio"]


    for alpha in regularizations:
        model = modelclass(alpha)
        model.fit(x, y)
        coef = pd.DataFrame(model.coef_.reshape(1,-1), columns=x.columns).transpose()
        R2 = model.score(x, y)
        print(model)
        print(R2)
        print(coef)
    return



# analyse(pd.read_csv("metrics-ama4-kltune.csv"))


