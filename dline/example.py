import pandas as pd
import torch
from dline import *
from data import *
from app import *
from stand import *
def loadData(path):
    return np.loadtxt(path, delimiter=",", skiprows=1,dtype=np.float64)
if __name__=="__main__":
    oriPath="BTC-USDT.txt"
    data=namedHistoryCandleSticks(pd.read_csv(oriPath))
    data=getEffectiveHistoryCandleSticks(data)
    data.to_csv('BTC_USDT.csv',header=True)
    windowSize=30
    rollingZScoreStand(data,windowSize,'open')
    rollingZScoreStand(data,windowSize,'high')
    rollingZScoreStand(data,windowSize,'low')
    rollingZScoreStand(data,windowSize,'close')
    CSNStand(data,Type.MONTH,'date')
    CSNStand(data,Type.DAY,'date')
    LOGZSCOREStand(data,windowSize,'volCcyQuote')
    data=data[:][windowSize:]
    rawData=data.copy()
    print(data)
    datanp=data[['openScaled','highScaled','lowScaled','closeScaled','dateMonthSin','dateMonthCos','dateDaySin','dateDayCos','volCcyQuoteLogScaled']].to_numpy()



    #
    # convertData(oriPath,dstPath)
    # data=loadData(dstPath)
    # # 设定窗口参数
    SEQ_LEN = 30  # 用过去 30 天的数据
    PRED_LEN = 5  # 预测未来 5 天
    trainData=datanp[:len(datanp)-SEQ_LEN-PRED_LEN]
    testStartTrimmedIndex=len(datanp)-SEQ_LEN-PRED_LEN
    rawData=rawData[:][:len(rawData)-5]
    testData=datanp[testStartTrimmedIndex:testStartTrimmedIndex+SEQ_LEN]
    trueData=datanp[testStartTrimmedIndex+SEQ_LEN:testStartTrimmedIndex+SEQ_LEN+SEQ_LEN]
    train_and_save(data=trainData,save_path="./model.pt",seq_len=SEQ_LEN,pred_len=PRED_LEN,epochs=50)
    model=load_model("./model.pt")
    testData=testData[0:SEQ_LEN]
    res=predict(testData,model)
    print("实际标准化空间数据:",trueData)
    print("标准化空间预测结果:", res)
    startIndex=len(rawData)-4
    priceKeys={"open":0,"high":1,"low":2,"close":3}
    restored=restorePredictions(res, rawData, windowSize, startIndex,priceKeys=priceKeys)
    print("还原后的 open:", restored["open"])
    print("还原后的 high:", restored["high"])
    print("还原后的 low:", restored["low"])
    print("还原后的 close:", restored["close"])

    restored=restorePredictions(trueData, rawData, windowSize, startIndex,priceKeys=priceKeys)
    print("真实还原后的 open:", restored["open"])
    print("真实还原后的 high:", restored["high"])
    print("真实还原后的 low:", restored["low"])
    print("真实还原后的 close:", restored["close"])






    # train_dataset = StockDataset(trainData, SEQ_LEN, PRED_LEN)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # # 1. 实例化模型、损失函数和优化器
    # model = DLinearForStock(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_features=num_features)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # # 2. 训练循环
    # epochs = 10
    # model.train()
    #
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     for batch_x, batch_y in train_loader:
    #         # batch_x: [32, 30, 5], batch_y: [32, 5, 5]
    #         optimizer.zero_grad()
    #
    #         # 预测
    #         outputs = model(batch_x)
    #
    #         # 计算损失
    #         loss = criterion(outputs, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += loss.item()
    #
    #     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    #
    # # 3. 模拟单次预测验证
    # model.eval()
    # with torch.no_grad():
    #     # 模拟最近30天的股票真实数据
    #     recent_30_days = torch.tensor(testData[:SEQ_LEN], dtype=torch.float32).unsqueeze(
    #         0)  # 增加 batch 维度 -> [1, 30, 5]
    #
    #     # 预测未来5天
    #     future_5_days_pred = model(recent_30_days)
    #
    #     print("\n--- 预测完成 ---")
    #     print("输入最近1天的价格(最后一行):", recent_30_days[0, -1, :].numpy())
    #     print("预测未来5天的价格:\n", future_5_days_pred[0].numpy())
    #     print("实际未来5天的价格:\n", testData[SEQ_LEN:SEQ_LEN+5])

