import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt

file = "./lut/KrisLUT0_OUT1.csv"
csv_lut = np.genfromtxt(file, delimiter=",")
lut_df = pd.read_csv(file)


x = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]
x = lut_df.Voltage.values
y = [
    1,
    2,
    4,
    7,
    6,
    3,
    2,
    3,
    4,
    5,
    9,
    12,
    11,
    9,
    7,
    3,
    1,
    2,
    3,
    5,
    8,
    12,
    14,
    16,
    12,
    8,
    3,
]
y = lut_df[["1452.52"]].values.flatten()
plt.plot(x, y)  # plot the original dataset
polynomial = P.fit(x, y, 14)  # 14 is the degree of the polynomial
fx, fy = polynomial.linspace()
plt.plot(fx, fy)  # plot the calculated polynomial
plt.show()
