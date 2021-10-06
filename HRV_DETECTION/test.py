# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from Preprocessing import BCG_Operation
from TemplateMatch import TemplateMarching,Cor,Dist


data_dir = "./dataset/%s.csv"

def test_templateMatching(data):
    preprocessing = BCG_Operation(sample_rate=1000)
    data = preprocessing.Butterworth(data, type="bandpass", low_cut=2, high_cut=10, order=2)
    tempplate = TemplateMarching(data, samplate_rate=1000)
    initPeak = tempplate.getInitPeak()
    model = tempplate.get_template(initPeak, num=8)
    envelop = tempplate.getEnvelop(data)
    beat = tempplate.beatDetection(Cor(data, model), envelop=envelop, maxs=1.3)
    beat = tempplate.refinement(data, beat)
    return beat

def stastic(beat, true_beat):
    label     = [1 if np.min(abs(beat - b)) < 30 else 0 for b in true_beat]
    predict   = [1 if np.min(abs(true_beat - b)) < 30 else 0 for b in beat]
    print("Predict:%d/%d,    label:%d/%d"%(np.sum(predict), len(predict), np.sum(label), len(label)))

if __name__ == '__main__':
    for i in range(0, 140):
        print("***"*10,i,"***"*10)
        df = pd.read_csv( data_dir%( str(i) ))
        BCG = df['BCG'].to_numpy().reshape(-1)
        print(BCG)
        print(type(BCG))
        print(len(BCG))
        ECG = df['ECG'].to_numpy().reshape(-1)
        Rpeak = df['Rpeak'].to_numpy().reshape(-1)
        Jpeak = df['Jpeak'].to_numpy().reshape(-1)

        beat = test_templateMatching(BCG)   ##模板匹配检测出来的J峰

        true_beat = np.argwhere(Jpeak==1)   ##标签的J峰
        print(len(beat),len(true_beat))
        stastic(beat, true_beat)
        plt.figure()
        plt.plot(BCG)
        plt.plot(beat, BCG[beat], 'r.')
        plt.plot(true_beat, BCG[true_beat], 'bx')
        plt.show()

