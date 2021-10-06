# _*_ coding: utf-8 _*_

"""
@ date:             2020-09-20
@ author:           jingxian
@ illustration:     Beat detection by template matching
"""
import numpy as np
from scipy import fftpack
from Preprocessing.Preprocessing import BCG_Operation

def distEuclidean(veca,vecb):
    """
    计算欧几里得距离
    """
    return np.sqrt( np.sum( np.square(veca-vecb) ) )

def findpeak(data):
    """
    :param data:                  输入序列信号
    :return:                      返还峰峰值对应的坐标数组( 峰峰值相隔>1s )
    """
    #建立峰峰值数组和其对应的坐标
    maxi = np.zeros(len(data) - 2)
    maxi_index = []
    # 获取峰峰值点对应的x坐标
    for i in range(1, len(data) - 2):
        maxi[i] = np.where([(data[i] - data[i - 1] > 0) & (data[i] - data[i + 1] > 0)],data[i], np.nan)
        if np.isnan(maxi[i]):
            continue
        maxi_index.append(i)
    return np.array( maxi_index ).astype(int)

def Cor(data, model):
    return np.correlate(data, model, "same")

def Dist(data, model):
    modelLength = len(model)
    dit = []
    for i in range(len(data) - len(model)):
        # para = 2-ASD(xindata[win][j:j + len(ModelBCG[win])], ModelBCG[win])/SAD(xindata[win][j:j + len(ModelBCG[win])], ModelBCG[win])
        para = 1
        dit.append(distEuclidean(data[i:i + len(model)], model) * para)
    dit = np.insert(dit, 0, np.full(modelLength // 2), dit[0])
    dit = np.append(dit, np.full(modelLength // 2), dit[-1])
    return dit


class TemplateMarching():

    def __init__(self, data, samplate_rate = 1000):
        self.data = data
        self.samplate_rate = samplate_rate

    def getInitPeak(self, mins = 0.5, maxs = 1.5):
        if len(self.data) < self.samplate_rate*5 : raise ValueError("The Input data is too short.")
        length = len( self.data )
        win_min = 0
        win_max = self.samplate_rate*1
        index = np.array([])
        while (True):
            beat = int(np.argmax(self.data[win_min:win_max]) + win_min)
            index = np.append(index, beat)
            win_min = max(0, beat + int(self.samplate_rate*mins))
            win_max = min(length, beat + int(self.samplate_rate*maxs))
            if (win_max >= length):
                break
        return index

    def get_template(self, Jpeak, num, length = 0.7, ):
        """
        :param Jpeak:                   The predict J peak in data
        :param length:                  template length (s)
        :return:                        template
        """
        if len(Jpeak) <= len(self.data)//self.samplate_rate//3 : raise ValueError("The length of input Jpeal is too small.")
        BCGModel = []
        ModelLength = int(self.samplate_rate * length)
        for peak in Jpeak:
            if peak < ModelLength // 2 or (peak + ModelLength) > len(self.data):continue
            else:
                BCGModel.append(self.data[int(peak - (ModelLength // 2)):int(peak + (ModelLength // 2))])
        meanBCG = np.mean(BCGModel, axis=0)                                         # ----------------------对初始预判J峰的信号段相加平均

        BCG_dit = [distEuclidean(model, meanBCG) for model in BCGModel]             # ----------------------计算初始预判信号与平均信号的相似性
        indexmin = np.array([])                                                     # -----------------------选择与平均信号最相似的2个原始信号
        for n in range(num):
            if len(BCG_dit) > 1:
                indexmin = np.append(indexmin, np.argmin(BCG_dit))
                BCG_dit[int(np.argmin(BCG_dit))] = float("inf")
            else:
                pass
        indexmin = indexmin.astype(int)
        Model = np.zeros(ModelLength)
        for n in indexmin:
            Model += BCGModel[n]
        Model = Model / num
        return Model

    def beatDetection(self, data, envelop, maxs = 1.5, MeanRR=900, up=1.86,down=0.53):
        """
            前向检测,根据Style选择合适的心跳位置
            :param data:                   输入数据信息
            :param up:                     上限倍数 Default = 1.86
            :param down:                   下限倍数 Default = 0.53
            :param style:                  根据峰或谷
            :return:                       心跳位置
            """
        length = len(data)

        # 心跳位置的索引
        index = []
        # 设置初始窗口
        win_min = 0
        win_max = int(self.samplate_rate * maxs)
        while (True):
            peak = findpeak(data[win_min: win_max]) + win_min  #返回[win_min:win_max]中所有峰值的横坐标（有多个峰值）
            if len(peak) == 0:  break
            peakmax = np.argmax(data[peak])    #返回最大峰值的横坐标
            beat = peak[peakmax]     #返回最大峰值的纵坐标
            if len(index) == 0 :
                if (beat > MeanRR * 1.2) and len(peak) > 1:  # 间隔过大，查询包络是否有峰值
                    EnvelopPeak = findpeak(envelop[win_min:win_min + self.samplate_rate]) + win_min
                    if len(EnvelopPeak) == 0:
                        index.append(beat)
                    else:
                        peak = findpeak(data[win_min:win_min + self.samplate_rate]) + win_min
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        index.append(beat)
                else:
                    index.append(beat)
            else:
                dit = beat - index[-1]
                std = np.mean(np.append(np.diff(index), MeanRR)) if len(index) != 1 else MeanRR
                while ((dit > (std * 1.4)) or (dit < (std / 1.4))) and (len(peak) > 1):
                    if (dit > (std * 1.4)):
                        EnvelopPeak = findpeak(envelop[win_min:win_min + int(self.samplate_rate*0.5)]) + win_min
                        if len(EnvelopPeak) == 0:
                            peak = np.delete(peak, peakmax)
                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            if data[Senbeat] > data[beat] * 0.9:
                                beat = Senbeat
                            dit = beat - index[-1]
                        else:
                            peak = findpeak(data[win_min:win_min + int(self.samplate_rate*0.5)]) + win_min
                            peak = peak.astype(int)
                            if (len(peak) == 0):
                                break
                            peakmax = np.argmax(data[peak])
                            Senbeat = peak[peakmax]
                            beat = Senbeat
                            dit = beat - index[-1]
                    elif dit < (std / 1.4):
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])
                        Senbeat = peak[peakmax]
                        if data[Senbeat] > data[beat] * 0.85:
                            beat = Senbeat
                        else:
                            break
                        dit = beat - index[-1]
                    elif dit > up * std or dit < down * std:
                        peak = np.delete(peak, peakmax)
                        peakmax = np.argmax(data[peak])
                        beat = peak[peakmax]
                        dit = beat - index[-1]
                    else:
                        break
                index.append(beat)
            win_min = int(max(0, index[-1] + self.samplate_rate*0.5))
            win_max = int(min(length, index[-1] + self.samplate_rate*maxs))
            if (win_min > length - 3):
                break
        return index

    def refinement(self, data, peak):
        if len(data) == 0 or len(peak) <=2 : return None
        firstPeak = peak[0]
        lastPeak = peak[-1]
        meanPeak = np.mean( data[peak[1:-1]] )
        if data[firstPeak] < meanPeak * 0.8 :
            peak = np.delete(peak, 0)
        if data[lastPeak] < meanPeak * 0.8 :
            peak = np.delete(peak, -1)
        return peak

    def getEnvelop(self, data):
        if len(data) == 0 : raise ValueError("the input data is empty.")
        hx = fftpack.hilbert(data)
        hy = np.sqrt(data ** 2 + hx ** 2)
        hy = BCG_Operation(sample_rate=self.samplate_rate).Butterworth(hy, type="lowpass", low_cut=1.5, order=4)
        return hy


