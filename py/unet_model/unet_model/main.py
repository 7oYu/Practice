# 这是一个示例 Python 脚本。
from time import sleep

import torch.optim
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import VocDataSet
from torch.utils.data import DataLoader
from torch import nn
from model import UNet
from PIL import Image


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Count:
    cnt = 0


def train():
    writer = SummaryWriter()
    dataset = VocDataSet()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
    model = UNet(3, 256)
    loss_fn = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(8):
        for batch, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.long().to(device)
            optimizer.zero_grad()
            pred = model(x)
            print('pred shape')
            print(pred.shape)
            # 打印目标值 y 的统计信息
            print(f"y min: {y.min().item()}, y max: {y.max().item()}")
            # 打印预测值 pred 的统计信息
            print(f"pred min: {pred.min().item()}, pred max: {pred.max().item()}")
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            print(f"loss: {loss:>7f}")
            writer.add_scalar('train loss', loss, global_step=Count.cnt)
            Count.cnt += 1
    torch.save(model, 'MyUNet.pt')


def load():
    model = torch.load('./MyUNet.pt')
    dataset = VocDataSet()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in data_loader:
        x = x.to(device)
        y = y.long().to(device)
        pred = model(x)
        print(pred.shape)
        pred = torch.squeeze(pred, dim=0)
        print(pred.shape)
        to_img = torchvision.transforms.ToPILImage()
        for i in range(pred.shape[0]):
            pred_img = to_img(pred[i])
            pred_img.show()
            sleep(4)
        break


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # train()
    load()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
