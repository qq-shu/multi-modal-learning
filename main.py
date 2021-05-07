import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
from xenonpy.descriptor import Compositions
from PIL import Image
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import pickle

def read_data():
    raw = pd.read_csv('/home/lab106/zy/pgm_exp/NIMS_Fatigue.csv').astype('float64')
    # Fatigue Tensile Fracture Hardness
    # raw.drop(columns=['Fatigue', 'Tensile', 'Fracture'], inplace=True)
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    raw = raw.apply(max_min_scaler)
    return raw

def svrHelper(x_train, x_test, y_train, y_test):
    model = SVR(gamma='auto')
    model.fit(x_train, y_train)
    print('[SVR] t2 = {}'.format(model.score(x_test, y_test)))

def rfrHelper(x_train, x_test, y_train, y_test):
    reg1 = RandomForestRegressor(n_estimators=5)
    reg1.fit(x_train, y_train)
    print('[RFR] t2 = {}'.format(reg1.score(x_test, y_test)))

# 返回经过计算后的成分原子特征df
def compFeatureHelper():
    raw = pd.read_csv('/home/lab106/zy/pgm_exp/NIMS_Fatigue.csv').astype('float64')
    eleName = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Cu', 'Mo']
    comp = pd.DataFrame(raw, columns=eleName).values
    compObj = []
    for i in range(len(raw)):
        tmp = {}
        for j in range(len(eleName)):
            if comp[i][j] != 0:
                tmp[eleName[j]] = comp[i][j]
        compObj.append(tmp)
    
    cal = Compositions()
    compRes = cal.transform(compObj)
    print(compRes['sum:atomic_number'])
    # 406个特征
    features_list = compRes.columns.tolist()[94:]
    compFeatures = pd.DataFrame(compRes, columns=features_list)
    return compFeatures


# 将原始数据里的元素信息删掉，单独做归一化，与元素特征按行拼接在一起。
# 元素特征在前，原始数据在后
def generateTotalDF(compF):
    rawF = pd.read_csv('/home/lab106/zy/pgm_exp/NIMS_Fatigue.csv').astype('float64')
    eleName = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Cu', 'Mo']
    rawF.drop(columns=eleName, inplace=True)
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    rawF = rawF.apply(max_min_scaler)
    compF = compF.apply(max_min_scaler)
    result = pd.concat([compF, rawF], axis=1)

    result.fillna(0, inplace=True)
    return result

# 测试将原子特征和剩余原始特征拼接在一起
# Fatigue [SVR] t2 = 0.7821698852361179 [RFR] t2 = 0.911370625326628
# Tensile [SVR] t2 = 0.7859893593885802 [RFR] t2 = 0.9397066004423241
# Fracture [SVR] t2 = 0.8593007106390008 [RFR] t2 = 0.9206576905447157
# Hardness [SVR] t2 = 0.7839053506903382 [RFR] t2 = 0.9214985822556397
def exp1():
    compF = compFeatureHelper()
    raw = generateTotalDF(compF).values
    # print(total.columns.tolist())
    features = raw[:, :-4]
    target = raw[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    svrHelper(x_train, x_test, y_train, y_test)
    rfrHelper(x_train, x_test, y_train, y_test)

def compPic():
    comp = compFeatureHelper()
    # 归一化
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    comp = comp.apply(max_min_scaler)
    # 映射到255
    comp.fillna(0, inplace=True)
    # gray_scaler = lambda x : round(x * 255)
    # comp = comp.apply(gray_scaler)
    # comp = comp.astype('int')
    
    # 特征列名
    emptyList = []
    featureName = comp.columns.tolist()
    for name in featureName:
        tmp = comp[name].sum()
        if tmp == 0:
            emptyList.append(name)
    # 删除空列，但为了凑一个17*17的图片，在这里少 drop 2列
    comp.drop(columns=emptyList[:-2], inplace=True)
    compV = comp.astype('float32').values  # (360, 289)
    grayCompV = np.reshape(compV, (-1, 17, 17))
    for i in range(len(grayCompV[0])):
        for j in range(len(grayCompV[0][i])):
            grayCompV[0][i][j] = round(grayCompV[0][i][j], 2)
    # print(grayCompV[0])
    for i in range(len(grayCompV[0])):
        s = '['
        for j in range(len(grayCompV[0][i])):
            s += (',' + str(grayCompV[0][i][j]))
        print(s + ']')

    # # 存储图片
    # cv.imwrite('/home/lab106/zy/pgm_exp/outfile0.png', grayCompV[0])
    # cv.imwrite('/home/lab106/zy/pgm_exp/outfile1.png', grayCompV[1])

    return compV

def picTest():
    img = np.array([[255, 0], [0, 255]])
    cv.imwrite('/home/lab106/zy/pgm_exp/outfilet.jpg', img)
    # im = Image.fromarray(img, mode='L')
    # im.save('/home/lab106/zy/pgm_exp/outfilet.png')

class multimodal(nn.Module):
    def __init__(self, width):
        super(multimodal, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        # self.linear1 = nn.Linear(4, 1)

        self.linear_o1 = nn.Linear(11, 1)

    def forward(self, pics, other):
        # x: (1, 1, 17, 17)
        out = self.conv1.forward(pics)   # (1, 1, 13, 13)
        out = self.conv2.forward(out) # (1, 1, 10, 10)
        out = self.conv3.forward(out) # (1, 1,  8,  8)
        out = self.conv4.forward(out) # (1, 1,  6,  6)
        out = self.conv5.forward(out) # (1, 1,  4,  4)
        out = self.conv6.forward(out) # (1, 1,  2,  2)
        out = torch.flatten(out, start_dim=1)
        # out = nn.functional.relu(self.linear1(out))
        out = torch.cat([other, out], dim=1)
        out = nn.functional.relu(self.linear_o1(out))

        return out

def exp2():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    pics = compPic()  # (360, 289)
    raw = read_data().astype('float32')

    # 预测值
    targets = raw.values[:, -4]
    # 删除除开成分以外的其他数据，去掉目标值
    eleName = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Cu', 'Mo', 'Fatigue', 'Tensile', 'Fracture', 'Hardness']
    otherF = raw.drop(columns=eleName).values  # 7列
    combineData = np.concatenate((pics, otherF), axis=1)
    # with open('/home/lab106/zy/pgm_exp/combineData.bin', 'wb') as f:
    #     pickle.dump(combineData, f)

    x_train, x_test, y_train, y_test = train_test_split(combineData, targets, test_size=0.33, random_state=42)

    x_train_pics = torch.from_numpy(x_train[:, :-7].astype('float64')).reshape(-1, 1, 17, 17).to(device)
    x_test_pics = torch.from_numpy(x_test[:, :-7].astype('float64')).reshape(-1, 1, 17, 17).to(device)
    x_train_other = torch.from_numpy(x_train[:, -7:].astype('float64')).to(device)
    x_test_other = torch.from_numpy(x_test[:, -7:].astype('float64')).to(device)

    y_train = torch.from_numpy(y_train.astype('float64')).to(device)
    y_test = torch.from_numpy(y_test.astype('float64')).to(device)

    model = multimodal(width=17)
    model.to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 50000
    save_flag = False
    a = None
    b = None

    try:
        for epoch in range(epoch_num):
            def closure():
                optimizer.zero_grad()
                out = model(x_train_pics, x_train_other)
                loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
                loss.backward()
                return loss

            optimizer.step(closure)

            if epoch % 10 == 9:
                print('epoch : ', epoch)
                pred = model(x_test_pics, x_test_other)
                loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
                # with open('/home/lab106/zy/pgm_exp/newlogs/loss.txt', 'a+') as f:
                #     f.write('{}\n'.format(loss.data.item()))
                print('test loss:', loss.data.item())
                r2 = r2_score(torch.squeeze(y_test).cpu().detach().numpy(), torch.squeeze(pred).cpu().detach().numpy())
                a = torch.squeeze(y_test).cpu().detach().numpy()
                b = torch.squeeze(pred).cpu().detach().numpy()
                print('r2:', r2)
    except KeyboardInterrupt:
        torch.save(model, '/home/lab106/zy/pgm_exp/newlogs/fatigue.bin')
        with open('/home/lab106/zy/pgm_exp/newlogs/fatigue.txt', 'a+') as f:
            f.write('{}\n{}\n'.format(a, b))


if __name__ == '__main__':
    # compPic()
    # picTest()
    exp2()
    # compFeatureHelper()