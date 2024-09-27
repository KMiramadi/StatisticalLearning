import pandas as pd
import numpy as np
import sklearn

data=pd.read_csv("project_train.csv")
features=data.drop("Label", axis=1)
targets=data["Label"]
