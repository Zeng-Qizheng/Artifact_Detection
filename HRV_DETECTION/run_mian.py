# coding=utf-8

"""
@date:         2020-09-13
@author:       jingxian
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Preprocessing import *

if __name__ == "__main__" :
    BCG   = pd.read_csv("./data/286_BCG1.txt", header=None).to_numpy().reshape(-1)
    ECG   = pd.read_csv("./data/286_ECG2.txt", header=None).to_numpy().reshape(-1)
    label = pd.read_csv("./label/22.txt", delimiter='\t', header=None).to_numpy()[:, 0].astype(int)
    print(label)
    # Rpeak = fin_turn(ECG, label)
    plt.figure()
    plt.plot(ECG)
    #plt.plot(Rpeak, ECG[Rpeak], 'r.')
    plt.plot(label, ECG[label], 'r.')
    # plt.plot(BCG)
    plt.show()