import numpy as np
import pandas as pd
import datetime
import sys
import matplotlib.pyplot as plt
from keras.src.layers import Flatten, Lambda, Reshape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor


sys.path.append('../')

timebegin = datetime.datetime.now()
pathName = 'E:/study/量化交易算法'
fileName = 'E:/study/量化交易算法/StockData.xlsx'
sheetName = '399300'

stockName = '399300'  # 设置输入 sheet 名，沪深 300 指数

df0 = pd.read_excel(fileName, sheet_name=sheetName)


def get_ma(data, maRange):
    ndata = len(data)
    nma = len(maRange)
    ma = np.zeros((ndata, nma))
    for j in range(nma):
        for i in range(maRange[j] - 1, ndata):
            ma[i, j] = data[(i - maRange[j] + 1):(i + 1)].mean()
    return ma



if stockName == '399300':
    ibegin = 242
if stockName == 'sz50':
    ibegin = 243


DateBS = df0['Date'].apply(lambda x: datetime.datetime.strptime(x.strip(), '%Y/%m/%d'))


OpenP = df0['Open'].values
CloseP = df0['Close'].values
nrecords = df0.shape[0]
Volume = df0['Vol'].values


maRange = range(1, 241)
volRange = range(1, 31)
initflag = 0
fileName1 = pathName + stockName + 'MA.npy'
fileName3 = pathName + stockName + 'Vol.npy'
if initflag == 1:
    dayMA = get_ma(CloseP, maRange)
    np.save(fileName1, dayMA)
    dayVOL = get_ma(Volume, maRange)
    np.save(fileName3, dayVOL)
else:
    dayMA = np.load(fileName1)
    dayVOL = np.load(fileName3)


nDays = 5
nPDays = 1

df1 = df0.iloc[:, 1:5]
df2 = pd.DataFrame(dayMA[:, [4, 9, 19]])
df3 = pd.DataFrame(dayVOL[:, [0]])
df = pd.concat([df1, df2, df3], axis=1)
priceName = ['Open', 'High', 'Low', 'Close', 'MA1', 'MA2', 'MA3']
volName = ['Vol']
df.columns = priceName + volName
df1 = df
dfNCol = df1.shape[1]
yC0 = np.array(df0.iloc[:, 4])
yC1 = np.array(df0.iloc[:, 4].shift(-nPDays))
dataY0 = pd.DataFrame({'yC0': yC0, 'yC1': yC1})

sc = MinMaxScaler()
scY = MinMaxScaler()

if stockName == '399300':
    ignoredays = 242
if stockName == 'sz50':
    ignoredays = 243

monthstep = 1
date1 = DateBS[ignoredays]
date2 = DateBS[nrecords - 1]
allsteps = round(np.ceil((date2.year - date1.year) * 12 + date2.month - date1.month) + 1) / monthstep
allsteps = int(allsteps)
m_dtmonth = np.zeros((allsteps, 3), dtype=np.int64)
m_dtmonth[0, 0] = date1.month
m_dtmonth[0, 1] = ignoredays
imonth = 0
m1 = date1.month
for i in range(1, nrecords):
    date2 = DateBS[i]
    m2 = date2.month + (date2.year - date1.year) * 12
    if (m2 - m1 < monthstep):
        m_dtmonth[imonth, 2] = i
    else:
        imonth = imonth + 1
        m_dtmonth[imonth, 0] = date2.month
        m_dtmonth[imonth, 1] = i
        m_dtmonth[imonth, 2] = i
        date1 = date2
        m1 = date1.month


if stockName == '399300':
    sampmonths = 8 * 12 // monthstep
    monthbegin = 8 * 12 // monthstep
if stockName == 'sz50':
    sampmonths = 9 * 12 // monthstep
    monthbegin = 9 * 12 // monthstep
imonth_par = allsteps - monthbegin

yP = np.zeros((nrecords, 1))
result_err1 = np.zeros((imonth_par, 4))
result_err2 = np.zeros((1, 2))


def data_sc(sc, dftr, dfte):
    dataset1 = dftr.values
    dataset2 = dataset1.reshape(-1, 1)
    dataset3 = sc.fit_transform(dataset2).reshape(dataset1.shape)
    df = pd.DataFrame(dataset3)
    df.columns = dftr.columns
    # 测试数据归一化
    dataset1 = dfte.values
    dataset2 = dataset1.reshape(-1, 1)
    dataset3 = sc.transform(dataset2).reshape(dataset1.shape)
    dft = pd.DataFrame(dataset3)
    dft.columns = dftr.columns
    return df, dft


def dataX_pre(df1, timeStep):
    nRecords = df1.shape[0]
    d1v = df1.values
    result = []
    for i in range(timeStep - 1):
        result.append(d1v[i:(i + timeStep)])
    for i in range(timeStep - 1, nRecords):
        result.append(d1v[(i - timeStep + 1):(i + 1)])
    x_train = np.array(result)
    return x_train


def prepareData(yC1, df1, df1t, nDays):
    df2, df2t = data_sc(sc, df1[priceName], df1t[priceName])
    df3, df3t = data_sc(sc, df1[volName], df1t[volName])
    df = pd.concat([df2, df3], axis=1)
    dft = pd.concat([df2t, df3t], axis=1)
    dataX = dataX_pre(df, nDays)[(nDays - 1):, :]
    dataXt = dataX_pre(dft, nDays)[(nDays - 1):, :]
    yC1s = scY.fit_transform(yC1.reshape(-1, 1))[:, 0]
    # 根据输入数据的长度和时间步长计算 yC1s 的期望长度
    expected_length = len(yC1) - nDays + 1
    if len(yC1s) < expected_length:
        padding_length = expected_length - len(yC1s)
        yC1s = np.pad(yC1s, (0, padding_length), mode='constant')
    elif len(yC1s) > expected_length:
        yC1s = yC1s[:expected_length]
    min_length = min(dataX.shape[0], len(yC1s))
    dataX = dataX[:min_length]
    yC1s = yC1s[:min_length]
    print(f"DataX shape: {dataX.shape}, yC1s shape: {yC1s.shape}")
    return dataX, dataXt, yC1s


def modelPredict(datX, y0, model):
    yp = model.predict(datX, verbose=0)
    yp = scY.inverse_transform(yp)[:, 0]
    return yp


def errorCalu(y0, y1):
    # 检查 y0 和 y1 的形状是否相同
    if y0.shape!= y1.shape:
        min_length = min(len(y0), len(y1))
        y0 = y0[:min_length]
        y1 = y1[:min_length]
    # 改动：在形状不匹配时，截取为相同长度
    x0 = np.array(y0)
    x1 = np.array(y1)
    error = (x1 - x0) / x0
    errorAbs = abs(error) * 100
    errAbsMean = errorAbs.mean()
    return errAbsMean


from tensorflow.keras.layers import Attention, Input, Concatenate
from tensorflow.keras.models import Model


x_train, _, y_train = prepareData(yC1, df1, df1, nDays)
y_train = y_train.reshape(-1, 1)


# 定义输入形状，这里假设 x_train 已经被正确定义
input_shape = (nDays, x_train.shape[2])


def create_attention_model(units=128, activation='tanh', dense_units=32, dense_activation='tanh'):
    # Define input layer
    inputs = Input(shape=input_shape)
    lstm_output = LSTM(units=units, activation=activation)(inputs)
    attention = Attention()([lstm_output, lstm_output])
    concatenated = Concatenate()([lstm_output, attention])
    flattened = Flatten()(concatenated)
    dense = Dense(dense_units, activation=dense_activation)(flattened)
    output = Dense(1, activation='linear')(dense)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


# Create a KerasRegressor wrapper for the model
model = KerasRegressor(model=create_attention_model, units=64, epochs=60, batch_size=32, verbose=0, activation='tanh', dense_activation='tanh', dense_units=16)


# 现在可以在 GridSearchCV 中搜索 activation 参数
param_grid = {
    'units': [64, 128, 256],
    'activation': ['tanh', 'relu'],
    'dense_units': [16, 32, 64],
    'dense_activation': ['tanh', 'relu']
}


grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')


# 运行网格搜索
grid_result = grid.fit(x_train, y_train)


print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)


modelflag = 1


for imonth in range(imonth_par):
    ibegin = m_dtmonth[monthbegin - sampmonths + imonth, 1]
    iend = m_dtmonth[monthbegin - 1 + imonth, 2] + 1
    ibegint = m_dtmonth[monthbegin + imonth, 1]
    iendt = m_dtmonth[monthbegin + imonth, 2] + 1
    yC1 = np.array(dataY0['yC1'][(ibegin - nPDays):(iend - nPDays)])
    dfxtr = df1.iloc[(ibegin - nPDays - nDays + 1):(iend - nPDays), :]
    dfxte = df1.iloc[(ibegint - nPDays - nDays + 1):(iendt - nPDays), :]
    xTrain0, xTest0, yC1s = prepareData(yC1, dfxtr, dfxte, nDays)
    yTrain0 = dataY0.iloc[(ibegin - nPDays):(iend - nPDays), :]
    index_length = len(yTrain0.index)
    if len(yC1s) < index_length:
        padding_length = index_length - len(yC1s)
        yC1s = np.pad(yC1s, (0, padding_length), mode='constant')
    elif len(yC1s) > index_length:
        yC1s = yC1s[:index_length]
    ytemp = pd.DataFrame(yC1s, index=yTrain0.index, columns=['yC1s'])
    yTrain0 = np.array(pd.concat([yTrain0, ytemp], axis=1))
    xTrain = xTrain0
    yTrain = yTrain0[:, 2]


    # Define and compile the new Attention model
    input_shape = (nDays, xTrain.shape[2])
    model = create_attention_model(units=128, activation='relu', dense_units=64, dense_activation='relu')


    filename = f'models/{nDays}_{nPDays}_{sampmonths}_{imonth}.h5'


    if modelflag > 1:
        model.load_weights(filename)


    from tensorflow.keras.callbacks import EarlyStopping


    if modelflag < 3:
        checkpoint = ModelCheckpoint(filepath=filename, save_weights_only=True, monitor='val_loss', mode='min',
                                 save_best_only=True, verbose=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # 新增早停回调
        history = model.fit(xTrain, yTrain, batch_size=32, epochs=60, validation_split=0.1,
                          callbacks=[checkpoint, early_stopping],
                          verbose=0)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


    yFita = modelPredict(xTrain0, yTrain0[:, 0], model)
    yTraina = yTrain0[:, 1]
    err1a = errorCalu(yTraina, yFita)


    if imonth == 0:
        yP[(ibegin - nPDays):(iend - nPDays), 0] = yFita


    yTest0 = dataY0.iloc[(ibegint - nPDays):(iendt - nPDays), :]
    yTesta = np.array(yTest0.yC1)
    yPrea = modelPredict(xTest0, np.array(yTest0.yC0), model)
    err2a = errorCalu(yTesta, yPrea)
    yP[(ibegint - nPDays):(iendt - nPDays), 0] = yPrea
    result_err1[imonth, 0:6] = [sampmonths, imonth, err1a, err2a]


    print(f'{DateBS[ibegint].strftime("XY-%m")} e1a={err1a:.4f} e2a={err2a:.4f}')
    from sklearn.metrics import mean_absolute_error, mean_squared_error


    print(mean_absolute_error(yPrea, yTesta))
    print(mean_squared_error(yPrea, yTesta))


ibegin = m_dtmonth[monthbegin, 1]
iend = m_dtmonth[allsteps - 1, 2] + 1
yTest0 = dataY0.iloc[(ibegin - nPDays):(iend - nPDays), :]
yTesta = np.array(yTest0.yC1)
yPrea = yP[(ibegin - nPDays):(iend - nPDays), 0]
err2a = errorCalu(yTesta, yPrea)
result_err2[0, 0:2] = [sampmonths, err2a]
print(f'samp={sampmonths} 总误差a={err2a:.4f}')


# Save results to files
df5 = df0.iloc[:, 0:5]
df5 = pd.concat([df5, dataY0[['yC1']]], axis=1)
df6 = pd.DataFrame(yP)
df6 = pd.concat([df5, df6], axis=1)
fileName = stockName + 'xP.xlsx'
sheet_name = stockName + 'xP'
df6.to_excel(fileName, sheet_name=sheet_name, index=False)


# Save summary results
listNum = {0}
listSave = [['样本内数据步长数', '第imonth窗口', '拟合误差a', '预测误差']]
listSave.extend(result_err1.tolist())
listNum.add(len(listSave))
listSave.append(['样本内数据月份数', '总预测误差a'])
listSave.extend(result_err2.tolist())
df_record = pd.DataFrame(listSave[1:len(listSave)])
df_record.to_excel('1-14att-lstm1好模型自动调参后.xlsx', sheet_name='dow0_b', index=False)


# Print total elapsed time
print(datetime.datetime.now() - timebegin)