## 理论分析

### 流程分析

一个最简单的完整的神经网络项目，他的训练流程一般需要这几步步。

1. **处理数据。**把文件数据处理成可以用pytorch处理的数据。
2. **定义模型。**将自己的模型架构搭建好。如有多少输入，几个隐藏层，几个输出。
3. **初始化模型、超参。**定义好了就要初始化，并且初始化超参的值。
4. **在训练集上进行训练。**有了模型就可以开始根据训练集的数据进行训练了。通过训练调参，最终得到一个较好的模型。
5. **在验证集上验证。**训练得到的模型可能只是在训练集上效果好，所以需要在验证集上进一步验证。找到最好的模型，并将其保存。
6. **在测试集上测试最终结果。**用保存的模型在测试集上测试。

基本流程就是这样。下面这个新冠感染人数预测的项目，也是这个流程。



## 代码

### 项目背景

美国，有40个州， 统计了连续三天的新冠阳性人数，和每天的一些社会特征，比如带口罩情况， 居家办公情况等等。现在把第三天的数据遮住，让我们用前两天的数据以及第三天的特征，来预测第三天的阳性人数。



### 处理数据

首先需要处理输入的文件数据（excel表格），我们需要将文件数据读入到我们的编程环境中。用python中的panda或者csv包实现。这里采用csv实现。

首先要继承Dataset类，得到一个`CovidDataset`类，在这个类中重写`__init__`、`__getitem__`和`__len__`函数。函数的功能和代码如下。

```python
class CovidDataset(Dataset):
    # 将数据从文件中读入，并处理数据格式
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:  # 以只读的方式打开文件
            ori_data = list(csv.reader(f))     # 读入文件 可迭代对象list化
            csv_data = np.array(ori_data)[1:, 1:].astype(float)    #处理源数据，去掉第一行和第一列并将矩阵中的数据转为float

          
        if mode == "train": #训练集 只取一部分数据
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]   #取下标不是5的倍数的
            data = torch.tensor(csv_data[indices, :-1])                 #取(indices, 不要最后一列）的数据X
            self.y = torch.tensor(csv_data[indices, -1])                #取X对应的Y
        elif mode == "val": #验证集 取另一部分
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]   # 取下标是5的倍数的，5个取四个
            data = torch.tensor(csv_data[indices, :-1])                 # 取(indices, 不要最后一列）的数据X
            self.y = torch.tensor(csv_data[indices, -1])
        else: #全部取是测试集
            indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices, :]) 	# 测试集需要全部列，因为测试集没有y

        self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)# 归一化
        self.mode = mode

    # 返回某一行
    def __getitem__(self, item):
        if self.mode != "test":
            return self.data[item].float(), self.y[item].float() # 返回一行
        else :
            return self.data[item].float()

    # 返回数据共多少行
    def __len__(self):
        return len(self.data)
```

训练集和验证集取的数据是一个数据文件。然后训练集取其中的4/5，验证集是另一个数据文件。

接下来对象实例化的过程

```python
train_file = "covid.train.csv" # 训练集和验证集的数据
test_file = "covid.test.csv" # 测试集的数据

train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(train_file, "val")
test_dataset = CovidDataset(test_file, "test")

#   按批次取数据 DataLoader是一个加载器，每次取一个批次的数据
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)# 测试集不能打乱取 因为要与答案验证
```

使用`DataLoader`加载数据，实现每一次调用`mode_loader`取一个批次的数据。需要特别注意的是测试集的每一个是去一个数据进行测试，并且测试集的数据是不能打乱的，因为要和官方给的答案进行验证，保证下标相同。



### 定义模型

完成输入数据的加载之后，就可定义模型了，如下。

```python
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
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        #消除维度为1的维度
        if len(x.size()) > 1:
            return x.squeeze(1)

        return x
```

消除维度为1是为了张量与取出数据y的张量的维度保持一致。



### 初始化模型、超参

```python
# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "lr": 0.001,
    "epochs":20,
    "momentum":0.9,
    "save_path":"model_save/best_model.pth"
}

# 初始化
model = MyModel(inDim=93).to(device) #模型
loss = nn.MSELoss() # 使用的loss函数
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])# 随机梯度下降优化器优化模型
```

lr：学习率

epochs：训练轮次

momentum：动量

save_path：最好模型的保存路径



### 训练和验证

```python
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
            train_batch_loss = loss(pred_y, target)# 2.计算该批次的loss
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
                val_batch_loss = loss(pred_y, target)  # 2.计算该批次的loss
                val_loss += val_batch_loss.cpu().item()  # 3.将一个批次loss从gpu中取出，并转为python内置数据类型加到总loss上
            plt_val_loss.append(val_loss / val_loader.__len__())  # 7.将这一轮的评价loss记录下来
            if val_loss < min_val_loss:
                torch.save(model, save_path)
                min_val_loss = val_loss

        print("[%03d/%03d] %2.2f sec(s) Trainloss: %.6f |Valloss: %.6f"% \
              (epoch, epochs, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1]))

    epoch_range = np.arange(1, epochs + 1, 2)  # 从 1 开始，更符合通常的 epoch 计数习惯
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("The comparison of loss between the training set and the validation set")
    plt.legend(["train", "val"])
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    # 设置 x 轴刻度为整数
    plt.xticks(epoch_range)
    plt.show()

```

在训练集和验证集上得出的结果。

![image-20250106145513082](picture/image-20250106145513082.png)

### 测试

```python
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
```

在测试集上跑出结果，并写入到文件中。这样就完成了整一个项目。
