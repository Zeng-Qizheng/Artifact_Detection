# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ecgdetectors import Detectors

import os
import shutil
import sys
import logging
import glob
from sklearn import preprocessing
from sampen import sampen2

from Preprocessing import BCG_Operation
from TemplateMatch import TemplateMarching,Cor,Dist
from TradictionalMethod import get_beat, get_beat2
from SQassess.SQIassessment import *
logging.basicConfig(level=logging.DEBUG,filename="new.log", filemode='a',format="%(message)s")

def test_templateMatching(data):
    preprocessing = BCG_Operation(sample_rate=1000)
    data = preprocessing.Butterworth(data, type="bandpass", low_cut=2, high_cut=8, order=2)
    tempplate = TemplateMarching(data, samplate_rate=1000)
    initPeak = tempplate.getInitPeak()
    model = tempplate.get_template(initPeak, num=2)
    envelop = tempplate.getEnvelop(data)
    beat = tempplate.beatDetection(Cor(data, model), envelop=envelop, maxs=1.4)
    beat = tempplate.refinement(data, beat)
    return beat

def test_templateMatching2(data):
    preprocessing = BCG_Operation(sample_rate=1000)
    data = preprocessing.Butterworth(data, type="bandpass", low_cut=2, high_cut=8, order=2)
    tempplate = TemplateMarching(data, samplate_rate=1000)
    initPeak = tempplate.getInitPeak()
    model = tempplate.get_template(initPeak, num=2)
    dist = Dist(data, model)
    dist = np.max(dist) - dist
    envelop = tempplate.getEnvelop(dist)
    beat = tempplate.beatDetection(dist, envelop=envelop, maxs=1.4)
    beat = tempplate.refinement(data, beat)
    return beat



data_dir = "./all_data/"
data_list = [os.path.join(data_dir, str(x)+".csv") for x in range(1, 2436)]




if __name__ == '__main__':
    for i in range(len(data_list)):
        print("Cur. :", i)
        logging.info("********"+str(i)+"*********")
        data = pd.read_csv(data_list[i])
        BCG = data["BCG"].to_numpy().reshape(-1)
        ECG = data["ECG"].to_numpy().reshape(-1)
        Jpeak = data["Jpeak"].to_numpy().reshape(-1)
        Rpeak = data["Rpeak"].to_numpy().reshape(-1)

        true_beat = np.argwhere(Jpeak == 1).reshape(-1)
        if len(true_beat) == 0:
            logging.info("meanHR:nan")
        else:
            logging.info("meanHR:" + str(np.mean(np.diff(true_beat))))

        beat_tempplate = test_templateMatching(BCG)
        logging.info("beat_tempplate:" + " ".join([str(i) for i in beat_tempplate]))

        beat_tempplate2 = test_templateMatching2(BCG)
        logging.info("beat_tempplate2:" + " ".join([str(i) for i in beat_tempplate2]))

        beat_heartvalue = get_beat(BCG, sample_rate=1000)
        logging.info("beat_heartvalue:" + " ".join([str(i) for i in beat_heartvalue]))

        beat_envelop = get_beat2(BCG)
        logging.info("beat_envelop:" + " ".join([str(i) for i in beat_envelop]))

        beat_tempplate_reversed = test_templateMatching(BCG[::-1])
        beat_tempplate_reversed = len(BCG) - np.array(beat_tempplate_reversed)
        logging.info("beat_tempplate_reversed:" + " ".join([str(i) for i in beat_tempplate_reversed]))

        beat_tempplate2_reversed = test_templateMatching2(BCG[::-1])
        beat_tempplate2_reversed = len(BCG) - beat_tempplate2_reversed
        logging.info("beat_tempplate2_reversed:" + " ".join([str(i) for i in beat_tempplate2_reversed]))

        preprocessing = BCG_Operation(sample_rate=1000)
        BCG = preprocessing.Butterworth(BCG, type="bandpass", low_cut=2, high_cut=10, order=2)

        format_bSQI = [
            (beat_tempplate, beat_tempplate2, "bSQI_temp1_2"),
            (beat_tempplate, beat_heartvalue, "bSQI_temp1_HV"),
            (beat_tempplate, beat_envelop, "bSQI_temp1_enve"),
            (beat_tempplate, beat_tempplate_reversed, "bSQI_temp1_1rev"),
            (beat_tempplate2, beat_tempplate2_reversed, "bSQI_temp2_2rev")
        ]
        for beat1, beat2, type_name in format_bSQI:
            temp = bSQI(beat1, beat2)
            logging.info(type_name + ":" + str(temp))

        format_tSQI = [
            (beat_tempplate, "tSQI_temp1"),
            (beat_heartvalue, "tSQI_HV"),
            (beat_envelop, "tSQI_enve"),
        ]
        for beat1, type_name in format_tSQI:
            temp = tSQI(BCG, beat1)
            logging.info(type_name + ":" + str(temp))

        format_iSQI = [
            (beat_tempplate, "iSQI_temp1"),
            (beat_envelop, "iSQI_enve")
        ]
        for beat1, type_name in format_iSQI:
            temp = iSQI(beat1)
            logging.info(type_name + ":" + str(temp))

        sSQI, kSQI = skSQI(BCG)
        logging.info("sSQI: " + str(sSQI))
        logging.info("kSQI: " + str(kSQI))

        snr = SNR2(BCG, beat_tempplate)
        logging.info("SNR: " + str(snr))




# detectors = Detectors(1000)
# for i in range(6, 22) :
#     BCG = pd.read_csv("./dataset/" + str(i) + "/BCG.txt",header=None).to_numpy().reshape(-1)
#     ECG = pd.read_csv("./dataset/" + str(i) + "/ECG.txt",header=None).to_numpy().reshape(-1)
#     R_peak = detectors.pan_tompkins_detector(ECG)
#     R_peak_new = trimming(ECG, R_peak, scope=200)
#     pd.DataFrame(R_peak_new).to_csv("./dataset/" + str(i) + "/location_R.txt", header=False, index=False)
#     print(i, BCG.shape[0], ECG.shape[0])
#     plt.figure()
#     #plt.plot(BCG-1000)
#     plt.plot(ECG)
#     plt.plot(R_peak_new, ECG[R_peak_new], "r.")
#     plt.show()
#     last_point = int(input())
#     if last_point == 0 :
#         continue
#     else :
#         print("last_point:%d s, %d min"%(last_point//1000, last_point//1000//60))
#         a = input()
#         pd.DataFrame( ECG[:last_point] ).to_csv("./dataset/" + str(i) + "/ECG.txt",header=False,index=False)







