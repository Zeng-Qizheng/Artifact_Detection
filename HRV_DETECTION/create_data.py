# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import scipy
import random
from sklearn import preprocessing

# ECG_dir = "./data/%s/ECG.txt"
# BCG_dir = "./data/%s/orgData.txt"
# Rpeak_dir = "./data/%s/location_R.txt"
# Jpeak_dir = "./data/%s/location_J.txt"

BCG_dir = "./zhongkexinzhi/%s/心晓脉搏信号.txt"
ECG_dir = "./zhongkexinzhi/%s/心电信号.txt"
Rpeak_dir = "./zhongkexinzhi/%s/Rpeak.txt"
Jpeak_dir = "./zhongkexinzhi/%s/Jpeak.txt"

def cut_peak():
    for i in range(1, 11):
        ECG = pd.read_csv( ECG_dir % (str( i )), header = None ).to_numpy().reshape(-1)
        BCG = pd.read_csv( BCG_dir % (str( i )), header = None ).to_numpy().reshape( -1 )
        Rpeak = pd.read_csv( Rpeak_dir % (str( i )) ).to_numpy().reshape( -1 )
        Jpeak = pd.read_csv( Jpeak_dir % (str( i )) ).to_numpy().reshape( -1 )
        Rpeak = Rpeak[ np.where(Rpeak<len(ECG)) ]
        Jpeak = Jpeak[ np.where(Jpeak<len(BCG)) ]
        pd.DataFrame( Rpeak ).to_csv( Rpeak_dir % (str( i )), index = False, header = False )
        pd.DataFrame( Jpeak ).to_csv( Jpeak_dir % (str( i )), index = False, header = False )

index_dict = {
    100: [0,220000],
    101: [150000,580000],
    102: [10000,420000],
    104: [0,600000],
    105: [10000,420000],
    106: [0,250000],
    107: [0,270000],
    108: [140000,560000],
    109: [150000,600000],
    110: [0,300000],
    112: [60000,6100000],
}

def create_data():
    num = 105
    for i in range(2, 9):
        # if i not in index_dict.keys():continue
        print("**********",str(i),"***********")
        # ------------------------------ read data -----------------------------------
        ECG = pd.read_csv( ECG_dir % (str( i )), header = None ).to_numpy().reshape(-1)
        BCG = pd.read_csv( BCG_dir % (str( i )), header = None ).to_numpy().reshape( -1 )
        Rpeak = pd.read_csv( Rpeak_dir % (str( i )) ).to_numpy().reshape( -1 )
        Jpeak = pd.read_csv( Jpeak_dir % (str( i )) ).to_numpy().reshape( -1 )

        # ----------------------------------------------------------------------------
        ECG_label = np.zeros(ECG.shape[0])
        BCG_label = np.zeros(BCG.shape[0])
        ECG_label[Rpeak] = 1
        BCG_label[Jpeak] = 1

        # ------------------------------ cut into 30s --------------------------------
        ECG = ECG[:len(ECG)//30000*30000].reshape(-1, 30000)
        BCG = BCG[:len(BCG)//30000*30000].reshape(-1, 30000)
        ECG_label = ECG_label[:len(ECG_label)//30000*30000].reshape(-1, 30000)
        BCG_label = BCG_label[:len(BCG_label)//30000*30000].reshape(-1, 30000)
        print( ECG.shape[0], BCG.shape[0] ,ECG_label.shape[0], BCG_label.shape[0])
        choose_index = random.sample( range(min(len(BCG),len(ECG))), 5 )
        for index in choose_index:
            bcg = BCG[index]
            ecg = ECG[index]
            j_peak = BCG_label[index]
            r_peak = ECG_label[index]
            data = {
                "BCG": bcg, "ECG": ecg, "Jpeak":j_peak, "Rpeak":r_peak
            }
            pd.DataFrame(data).to_csv("./dataset/%s.csv"%(str(num)), index=False)
            num += 1


create_data()

# df = pd.read_csv("./dataset/18.csv")
# df['BCG'] = preprocessing.scale( df['BCG'] )
#
# df.plot()
# plt.show()



