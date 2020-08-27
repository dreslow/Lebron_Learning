import pandas as pd
import numpy as np
import scipy.optimize as opt

Data = np.array(pd.read_csv(r'LebronCLEstatsTest1.csv'))
X = Data[:, 0:16]
y = np.floor(Data[:, 17]/10).reshape((-1, 1)).astype(int)