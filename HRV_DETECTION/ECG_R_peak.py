# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecgdetectors import Detectors

BCG_dir = "./zhongkexinzhi/%s/心晓脉搏信号.txt"
ECG_dir = "./zhongkexinzhi/%s/心电信号.txt"
Rpeak_dir = "./zhongkexinzhi/%s/Rpeak.txt"
Jpeak_dir = "./zhongkexinzhi/%s/Jpeak.txt"

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

def fineTun(data, peaks, th=200):
    return_peak = []
    for peak in peaks :
        if peak > len(data):continue
        min_win, max_win = max(0, int(peak-th)), min(len(data), int(peak+th))
        return_peak.append( np.argmax(data[min_win: max_win]) + min_win )
    return return_peak



for i in range(2, 9):
    print("**"*10,i,"**"*10)
    # if i not in index_dict.keys(): continue
    BCG = pd.read_csv(BCG_dir%( str(i) ), delimiter = '\t', header = None).to_numpy().reshape(-1)
    ECG = pd.read_csv(ECG_dir%( str(i) ), delimiter = '\t', header = None).to_numpy().reshape(-1)
    Rpeak = pd.read_csv(Rpeak_dir%( str(i) ), delimiter = '\t', header = None).to_numpy().reshape(-1)
    #Jpeak = pd.read_csv( Jpeak_dir % (str( i )), delimiter = '\t', header = None ).to_numpy().reshape( -1 )
    # print(Rpeak)
    Jpeak = fineTun(BCG, Rpeak, th = 70)
    print(Jpeak)
    pd.DataFrame( Jpeak ).to_csv( Jpeak_dir % (str( i )), index = False, header = False )
    # --------------------------------截取--------------------------------------------
    # start, end = index_dict[i]
    # pd.DataFrame( BCG[start:end] ).to_csv( BCG_dir % (str( i )), index = False, header = False )
    # pd.DataFrame( ECG[start:end] ).to_csv( ECG_dir % (str( i )), index = False, header = False )
    # -------------------------------R 峰检测和保存---------------------------------------
    # detector = Detectors(1000)
    # r_peak = np.array( detector.pan_tompkins_detector(ECG) )
    # r_peak = fineTun(ECG, r_peak)
    # pd.DataFrame(r_peak).to_csv(Rpeak_dir%( str(i) ), index = False, header = False)

    # ---------------------------------绘图---------------------------------------------------
    #　plt.figure()
    #　plt.plot(BCG)
    #　plt.plot(ECG-300)
    #　plt.plot(Jpeak, BCG[Jpeak], 'r.')
    #　plt.plot( Rpeak, ECG[Rpeak]-300, 'b.' )
    #　plt.show()