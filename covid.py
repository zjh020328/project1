import time
import torch
import numpy as np
import csv
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest
from torch import optim
from torch.utils.data import DataLoader, Dataset

def get_feature_importance(feature_data, label_data, k =4, column = None):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    这个函数的目的是， 找到所有的特征种， 比较有用的k个特征， 并打印这些列的名字。
    """
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    feature_data = np.array(feature_data, dtype=np.float64)
    # label_data = np.array(label_data, dtype=np.float64)
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]        #[::-1]表示反转一个列表或者矩阵。
    # argsort这个函数， 可以矩阵排序后的下标。 比如 indices[0]表示的是，scores中最小值的下标。

    if column:                            # 如果需要打印选中的列
        k_best_features = [column[i+1] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。

class CovidDataset(Dataset):
    # 将数据从文件中读入，并处理数据格式
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:  # 以只读的方式打开文件
            ori_data = list(csv.reader(f))     # 读入文件 可迭代对象list化
            csv_data = np.array(ori_data)[1:, 1:].astype(float)    #处理源数据，去掉第一行和第一列并将矩阵中的数据转为float

        #训练集 只取一部分数据 验证集 取另一部分 全部取是测试集
        if mode == "train":
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]   #取下标不是5的倍数的
            data = torch.tensor(csv_data[indices, :-1])                 #取(indices, 不要最后一列）的数据X
            self.y = torch.tensor(csv_data[indices, -1])                #取X对应的Y
        elif mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]   # 取下标是5的倍数的，5个取四个
            data = torch.tensor(csv_data[indices, :-1])                 # 取(indices, 不要最后一列）的数据X
            self.y = torch.tensor(csv_data[indices, -1])
        else:
            indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices, :])

        self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)# 归一化
        self.mode = mode

    # 返回某一行
    def __getitem__(self, item):
        if self.mode != "test":
            return self.data[item].float(), self.y[item].float() # 返回一行
        else :
            return self.data[item].float()

    # 返回共多少行
    def __len__(self):
        return len(self.data)

class MyModel(nn.Module):
    # inDim为输入的维度
    def __init__(self, inDim):
        super(MyModel, self).__init__() # 继承父类的init
        # 定义一个全连接模型
        self.fc1 = nn.Linear(inDim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    # 取一批次的数据前向过程 x.shape = (16, 93) -> (16, 64) -> (16, 1) (squeeze)-> (16)
    def forward(self, x):
        # print("输入到 fc1 之前 x 的形状:", x.shape)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze(1)

        return x

def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)

    plt_train_loss = [] # 用于记录每一轮的loss
    plt_val_loss = []   # 同上

    min_val_loss = 999999999999 # 记录最小的loss

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time() #记录开始时间

        model.train()   #模型为训练模型
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)#0.取出实际数据X, Y
            pred_y = model(x)# 1.forward得到预测值
            train_batch_loss = loss(pred_y, target, model)# 2.计算该批次的loss
            train_batch_loss.backward()# 3.计算参数的梯度
            optimizer.step() # 4.更新模型
            optimizer.zero_grad() # 5.清除梯度
            train_loss += train_batch_loss.cpu().item() # 6.将一个批次loss从gpu中取出，并转为python内置数据类型加到总loss上
        plt_train_loss.append(train_loss / train_loader.__len__()) # 7.将这一轮的评价loss记录下来


        model.eval()    #模型为验证模型
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)  # 0.取出实际数据X, Y
                pred_y = model(x)  # 1.forward得到预测值
                val_batch_loss = loss(pred_y, target, model)  # 2.计算该批次的loss
                val_loss += val_batch_loss.cpu().item()  # 3.将一个批次loss从gpu中取出，并转为python内置数据类型加到总loss上
            plt_val_loss.append(val_loss / val_loader.__len__())  # 7.将这一轮的评价loss记录下来
            if val_loss < min_val_loss:
                torch.save(model, save_path)
                min_val_loss = val_loss

        print("[%03d/%03d] %2.2f sec(s) Trainloss: %.6f | Valloss: %.6f"% \
              (epoch, epochs, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1]))

    # plt.plot(plt_train_loss)
    # plt.plot(plt_val_loss)
    # plt.title("loss")
    # plt.legend(["train", "val"])
    # plt.show()
    epoch_range = np.arange(1, epochs + 1)  # 从 1 开始，更符合通常的 epoch 计数习惯
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("The comparison of loss between the training set and the validation set")
    plt.legend(["train", "val"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 设置 x 轴刻度为整数
    plt.xticks(epoch_range)
    plt.show()

def evaluate(save_path, test_loader,device,rel_path ):   #得出测试结果文件
    model = torch.load(save_path).to(device)
    rel = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())
    print(rel)
    with open(rel_path, "w", newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id", "tested_positive"])
        for i, value in enumerate(rel):
            csvWriter.writerow([str(i), str(value)])
    print("文件已经保存到" + rel_path)



train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(train_file, "val")
test_dataset = CovidDataset(test_file, "test")

#   按批次取数据 DataLoader是一个加载器，每次取一个批次的数据
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)# 测试集不能打乱取 因为要与答案验证

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "lr": 0.001,
    "epochs":20,
    "momentum":0.8,
    "save_path":"model_save/best_model.pth",
    "rel_path":"pred.csv"
}

def mseLoss_with_reg(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    ''' Calculate loss '''
    regularization_loss = 0                    # 正则项
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)                  # 计算所有参数平方
    return loss(pred, target) + 0.00075 * regularization_loss             # 返回损失。

# 训练过程
# 初始化
model = MyModel(inDim=93).to(device) #模型
loss = mseLoss_with_reg # loss函数
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])# 随机梯度下降优化模型

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])

evaluate(config["save_path"], test_loader, device, config["rel_path"])