import argparse
import copy
import os
import lib.alphabets as alphabets
import torch
import random
import numpy as np
from lib import dataset, convert
from torchvision import transforms
from torch.utils.data import DataLoader
from net.CRNN_Net import CRNN
import torch.optim as optim
from tqdm import tqdm
import time

# train_data_path = r"E:\Desktop\dataset\cblprd_train_lmdb"
train_data_path = r"C:\Users\gg\cblprd_train_lmdb"
val_data_path = r"E:\Desktop\dataset\cblprd_val_lmdb"
weight = r"E:\experiment\python\plate_id\crnn_ctc\runs\train\best_weights.pth"


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=train_data_path)
    parser.add_argument('--val_data', type=str, default=val_data_path)
    parser.add_argument('--image_size', type=tuple, default=(32, 100))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--alphabet', type=str, default=alphabets.alphabets)
    parser.add_argument('--project', default='./runs/train/')
    parser.add_argument('--random_seed', type=int, default=111)
    parser.add_argument('--using_cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default="RMSprop")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--load_weight', type=bool, default=True)
    parser.add_argument('--weight_path', type=str, default=weight)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_and_eval(model, epochs, loss_func, optimizer, train_loader, val_loader):
    # 初始化参数
    lst = []
    t_start = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        logs = "-" * 100 + "\n"
        # train
        model.train()
        running_loss, running_acc = 0.0, 0.0
        train_bar = tqdm(train_loader)
        train_bar.desc = f"第{epoch + 1}次训练，Processing"
        for inputs, labels in train_bar:
            logs += "*" * 50 + "\n"
            inputs = inputs.to(device)  # 数据放到device中
            optimizer.zero_grad()
            outputs = model(inputs)
            # 计算这个batch的损失值
            # print(type(labels))
            text, text_length = converter.ocr_encode(text=labels)
            # print(text)
            # print(text_length)
            text, text_length = text.to(device), text_length.to(device)  # 数据放到device中
            outputs_length = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), device=device,
                                        dtype=torch.long)

            loss = loss_func(outputs, text, outputs_length, text_length)
            running_loss += loss.item() * outputs.size(1)
            # 计算这个batch正确识别的字符数
            preds_size = torch.IntTensor([outputs.size(0)] * outputs.size(1))
            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            # print(preds.view(64, 22))
            # print(preds_size)
            sim_preds = converter.ocr_decode(preds.data, preds_size.data)
            # print(len(sim_preds), sim_preds)
            counter, lst = 0, []
            for i in labels:
                lst.append(i.decode("utf-8", "strict"))
            # print(lst)
            for pred, target in zip(sim_preds, lst):
                # print(pred, target)
                if pred == target:
                    counter += 1
            logs += f"target:{lst}\n"
            logs += f"pred:{sim_preds}\n"
            logs += "*" * 50 + "\n"
            running_acc = counter+running_acc
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

        # 计算本epoch的损失值和正确率
        train_loss = running_loss / len(train_dataset)
        train_acc = running_acc / len(train_dataset)
        train_state = f"第{epoch + 1}次训练，train_loss:{train_loss:.6f}, train_acc:{train_acc:.6f}\n"
        logs += train_state

        # eval
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        with torch.no_grad():
            eval_bar = tqdm(val_loader)
            eval_bar.desc = f"第{epoch + 1}次评估，Processing"
            for inputs, labels in eval_bar:
                inputs = inputs.to(device)
                outputs = model(inputs).to(device)
                # 计算这个batch的损失值
                text, text_length = converter.ocr_encode(text=labels)
                text, text_length = text.to(device), text_length.to(device)  # 数据放到device中
                outputs_length = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), device=device,
                                            dtype=torch.long)

                loss = loss_func(outputs, text, outputs_length, text_length)
                running_loss += loss.item() * outputs.size(1)
                # 计算这个batch正确识别的字符数
                preds_size = torch.IntTensor([outputs.size(0)] * outputs.size(1))
                _, preds = outputs.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.ocr_decode(preds.data, preds_size.data)
                counter, lst = 0, []
                for i in labels:
                    lst.append(i.decode("utf-8", "strict"))
                for pred, target in zip(sim_preds, lst):
                    if pred == target:
                        counter += 1
                running_acc = counter+running_acc

        # 计算本epoch的损失值和正确率
        val_loss = running_loss / len(val_dataset)
        val_acc = running_acc / len(val_dataset)
        val_state = f"第{epoch + 1}次评估，val_loss:{val_loss:.6f}, val_acc:{val_acc:.6f}\n"
        logs += val_state
        logs += "-" * 100 + "\n"
        print(logs)
        lst.append(logs)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    t_end = time.time()
    total_time = t_end - t_start
    result = f"{epochs}次训练与评估共计用时{total_time // 60:.0f}m{total_time % 60:.0f}s\n最高正确率是{best_acc:.6f}"
    print(result)
    lst.append(result)

    # 加载最佳的模型权重
    model.load_state_dict(best_weights)
    return model, lst


if __name__ == "__main__":
    opt = parse_opt()
    print(type(opt.train_data))
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)
    print(opt.epochs)
    if torch.cuda.is_available() and opt.using_cuda:
        device = torch.device('cuda:0')
        # torch.backends.cudnn.deterministic = True
        print("使用单个gpu进行训练")
    else:
        device = torch.device('cpu')
        print("使用cpu进行训练")
    # 随机种子
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # 数据增强
    transformer = {"train": transforms.Compose([transforms.Resize((32, 100)),
                                                transforms.ToTensor(),
                                                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                                                transforms.ColorJitter(brightness=0.5,),
                                                transforms.Normalize(0.418, 0.141)]),
                   "val": transforms.Compose([transforms.Resize((32, 100)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(0.418, 0.141)])}
    # 制作数据集
    train_dataset = dataset.LmdbDataset(root=opt.train_data, transform=transformer["train"])
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              pin_memory=opt.pin_memory)
    val_dataset = dataset.LmdbDataset(root=opt.val_data, transform=transformer["val"])
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                            pin_memory=opt.pin_memory)
    n_class = len(opt.alphabet) + 1
    print(f"字母表的长度是{n_class}")  # 包含blank
    # 字符与标签转换器
    converter = convert.StrLabelConverter(opt.alphabet)
    # crnn网络
    net = CRNN(n_class)
    net.apply(weights_init)
    net = net.to(device)
    if opt.load_weight:
        net.load_state_dict(torch.load(opt.weight_path, map_location=device))
    print(net)
    # ctc损失函数
    ctc = torch.nn.CTCLoss()
    # 优化器
    if opt.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr)
    elif opt.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=opt.lr)
    elif opt.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    # 训练和评估
    best_model, log = train_and_eval(net, opt.epochs, ctc, optimizer, train_loader, val_loader)
    best_weights = best_model.state_dict()
    torch.save(best_weights, opt.project + "best_weights.pth")
    with open(opt.project + "logs.txt", "w") as f:
        for i in log:
            f.write(i)