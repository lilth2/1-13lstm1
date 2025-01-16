import numpy as np
import pandas as pd
import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Attention, Flatten, Dropout, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV


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
    # 调整滑动窗口的生成逻辑，添加边界检查
    for i in range(max(0, nRecords - timeStep)):
        result.append(d1v[i:(i + timeStep)])
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

def modelPredict(dataX, y0, model):
    y_pred= model.predict(dataX).reshape(-1, 1)
    y_pred = scY.inverse_transform(y_pred)[:, 0]
    return y_pred


def errorCalu(y0, y1):
    x0 = np.array(y0)
    x1 = np.array(y1)
    if np.any(x0 == 0):
        # 将 x0 中为 0 的元素替换为极小值，避免除以零
        x0[x0 == 0] = 1e-10
    error = (x1 - x0) / x0
    errorAbs = abs(error) * 100
    errAbsMean = errorAbs.mean()
    return errAbsMean


def build_attention_lstm_model(input_shape, lstm_units1=64, lstm_units2=32, dropout_rate1=0.4, dropout_rate2=0.4, dense_units1=32, dense_units2=16):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(lstm_units1, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(inputs)
    lstm_out = LSTM(lstm_units2, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(lstm_out)
    attention_out = Attention()([lstm_out, lstm_out])
    flatten_out = Flatten()(attention_out)
    dense1 = Dense(dense_units1, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flatten_out)
    dropout1 = Dropout(dropout_rate1)(dense1)
    dense2 = Dense(dense_units2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dropout1)
    dropout2 = Dropout(dropout_rate2)(dense2)
    outputs = Dense(1, activation='linear')(dropout2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(0.001), loss='mean_squared_error')
    return model


def create_model(lstm_units1=64, lstm_units2=32, dropout_rate1=0.4, dropout_rate2=0.4, dense_units1=32, dense_units2=16):
    return build_attention_lstm_model((nDays, df1.shape[1]), lstm_units1=lstm_units1, lstm_units2=lstm_units2, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, dense_units1=dense_units1, dense_units2=dense_units2)


# 定义参数搜索空间
param_grid = {
    'lstm_units1': [32, 64, 128],
    'lstm_units2': [32, 64],
    'dropout_rate1': [0.2, 0.4, 0.6],
    'dropout_rate2': [0.2, 0.4, 0.6],
    'dense_units1': [16, 32],
    'dense_units2': [8, 16]
}


# 使用 KerasRegressor 包装 create_model 函数，并使用 GridSearchCV 进行参数搜索
model = KerasRegressor(model=create_model, lstm_units1=64, lstm_units2=32, dropout_rate1=0.4, dropout_rate2=0.4, dense_units1=32, dense_units2=16, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)


# 准备用于参数搜索的数据，假设使用全部数据进行搜索
x_train, _, y_train = prepareData(yC1, df1, df1, nDays)
y_train = y_train.reshape(-1, 1)


# 进行参数搜索
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
    # 检查 yC1s 和 yTrain0 的形状

    # 创建 DataFrame
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
    filename = 'models/' + \
               str(nDays) + '_' + \
               str(nPDays) + '_' + \
               str(sampmonths) + '_' + \
               str(imonth) + '.h5'
    model = build_attention_lstm_model((nDays, xTrain.shape[2]))
    if modelflag > 1:
        model.load_weights(filename)

    if modelflag < 3:
        checkpoint = ModelCheckpoint(filepath=filename, save_weights_only=True,
                                     monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(xTrain, yTrain, batch_size=32, epochs=200,
                            validation_split=0.1, callbacks=[checkpoint, early_stopping], verbose=0)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    yFita = modelPredict(xTrain0, yTrain0[:, 0], model)
    yTraina = yTrain0[:, 1]
    if yTraina.shape!= yFita.shape:
        print("Warning: Shapes of yTraina and yFita do not match. Attempting to fix...")
        # 找到两个数组中的较小长度
        min_length = min(yTraina.shape[0], yFita.shape[0])
        # 截取两个数组以匹配较小长度
        yTraina = yTraina[:min_length]
        yFita = yFita[:min_length]

    # 现在可以安全地调用 errorCalu 函数
    err1a = errorCalu(yTraina, yFita)

    if imonth == 0:
        yP[(ibegin - nPDays):(iend - nPDays), 0] = yFita

    yTest0 = dataY0.iloc[(ibegint - nPDays):(iendt - nPDays), :]
    yTesta = np.array(yTest0.yC1)
    yPrea = modelPredict(xTest0, np.array(yTest0.yC0), model)
    if yTesta.shape!= yPrea.shape:
        print("Warning: Shapes of yTesta and yPrea do not match. Attempting to fix...")
        # 找到两个数组中的较小长度
        min_length = min(yTesta.shape[0], yPrea.shape[0])
        # 截取两个数组以匹配较小长度
        yTesta = yTesta[:min_length]
        yPrea = yPrea[:min_length]
    err2a = errorCalu(yTesta, yPrea)
    yP[(ibegint - nPDays):(iendt - nPDays), 0] = yPrea
    result_err1[imonth, 0:4] = [sampmonths, imonth, err1a, err2a]
    print('%s e1a=%6.4f e2a=%6.4f' % (DateBS[ibegint].strftime('XY-%m'), err1a, err2a))


ibegin = m_dtmonth[monthbegin, 1]
iend = m_dtmonth[allsteps - 1, 2] + 1
yTest0 = dataY0.iloc[(ibegin - nPDays):(iend - nPDays), :]
yTesta = np.array(yTest0.yC1)
yPrea = yP[(ibegin - nPDays):(iend - nPDays), 0]
err2a = errorCalu(yTesta, yPrea)
result_err2[0, 0:2] = [sampmonths, err2a]
print('samp=%d 总误差a=%6.4f' % (sampmonths, err2a))

df5 = df0.iloc[:, 0:5]
df5 = pd.concat([df5, dataY0[['yC1']]], axis=1)
df6 = pd.DataFrame(yP)
fileName = stockName + 'xP.xlsx'
df6 = pd.concat([df5, df6], axis=1)
fileName = stockName + 'xP.xlsx'
sheet_name = stockName + 'xP'
df6.to_excel(fileName, sheet_name=sheet_name, index=False)

listNum = {0}
listSave = [['样本内数据步长数', '第imonth窗口', '拟合误差a', '预测误差']]
listSave.extend(result_err1.tolist())
listNum.add(len(listSave))

listSave.append(['样本内数据月份数', '总预测误差a'])
listSave.extend(result_err2.tolist())
df_record = pd.DataFrame(listSave[1:len(listSave)])
df_record.to_excel('新的test.xlsx', sheet_name='dow0_b', index=False)

print(datetime.datetime.now() - timebegin)