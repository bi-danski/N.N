import pandas as pd
import numpy as np


filepath = "CX1_CX8.xlsx"
# filepath = "data/ChinkuData.xlsx"

def getXLSX():
    df = pd.concat(pd.read_excel(filepath, sheet_name=None), ignore_index=True)
    unnamed = []
    xfilter = []
    yfilter = []

    for col in df.columns:
        if type(col) == str:
            unnamed.append(col)
        else:
            continue

    df.drop(columns=unnamed, axis=1, inplace=True)
    df.dropna(inplace=True)
    # df.reset_index(drop=True)

    for index, col in enumerate(df.columns):
        if index > int(df.shape[1] * (70 / 100)):
            yfilter.append(col)
        else:
            xfilter.append(col)

    X = df[xfilter]
    y = df[yfilter]

    X.to_csv("features.csv")
    y.to_csv("target.csv")

    return X, y

X, y = getXLSX()
