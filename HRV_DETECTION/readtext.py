import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "./dataset/%s.csv"

data = np.loadtxt('D:/FinishSample/282huangchengcheng/BCG.txt', delimiter= ',')
#输出结果是numpy中数组格式
print(data)

df = data[90000:120000]
savedata =pd.DataFrame(df)
savedata.rename(columns={'0':'BCG'},inplace=True)
print(savedata)
savedata.to_csv("./dataset/148.csv",index =0)

for i in range(140, 150):
    print("***" * 10, i, "***" * 10)
    df = pd.read_csv(data_dir % (str(i)))
    BCG = df['BCG'].to_numpy().reshape(-1)
    print(len(BCG))
    plt.figure()
    plt.plot(BCG)
    plt.show()









