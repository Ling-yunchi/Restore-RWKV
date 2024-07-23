import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.HYPSO1 import HYPSO1_Dataset
from data.MedicalDataUniform import DataSampler
from data.common import dataIO
from model.Restore_RWKV import Restore_RWKV
from tools import mkdir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

io = dataIO()


def save_model(net_model, save_dir, optimizer=None, ex=""):
    save_path = os.path.join(save_dir, "Model")
    mkdir(save_path)
    G_save_path = os.path.join(save_path, 'Generator{}.pth'.format(ex))
    torch.save(net_model.cpu().state_dict(), G_save_path)
    net_model.cuda()

    if optimizer is not None:
        opt_G_save_path = os.path.join(save_path, 'Optimizer_G{}.pth'.format(ex))
        torch.save(optimizer.state_dict(), opt_G_save_path)


'''
Testing Code
'''

total_epoch = int(3e5)
val_interval = int(20)

lr = 2e-4

batch_size = 1
eps = 1e-8

loss_min = 1e8

data_root = "./dataset/1-DATA WITH GROUND-TRUTH LABELS"
save_dir = "experiment/Restore_RWKV"

model_path = None
optimizer_path = None

img_size = (256, 256)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(img_size),
    transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in [0, 90, 180, 270]]),
])

train_dataset = HYPSO1_Dataset(data_root, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_sampler = DataSampler(train_loader)

valid_dataset = HYPSO1_Dataset(data_root, train=False, transform=transforms.Resize(img_size))
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

net = Restore_RWKV(inp_channels=120, out_channels=3, add_raw=False)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps)
lr_scheduler = CosineAnnealingLR(optimizer, total_epoch, eta_min=1.0e-6)
criterion = nn.CrossEntropyLoss().cuda()

if model_path is not None:
    net.load_state_dict(torch.load(model_path))
if optimizer_path is not None:
    optimizer.load_state_dict(torch.load(optimizer_path))

running_loss = {
    "train_loss": [],
    "train_accuracy": []
}
eval_loss = {
    "val_loss": [],
    "val_accuracy": []
}

print("################ Train ################")
val_loss = "inf"
val_accuracy = "inf"

pbar = tqdm(total=int(total_epoch))
for epoch in list(range(1, int(total_epoch) + 1)):

    train_data, train_label = next(iter(train_sampler))
    train_data, train_label_f = train_data.type(torch.FloatTensor).cuda(), train_label.type(torch.FloatTensor).cuda()

    net.train()
    optimizer.zero_grad()
    train_result = net(train_data)
    train_loss = criterion(train_result, train_label_f)
    train_loss.backward()
    optimizer.step()

    train_result_classes = torch.argmax(train_result, dim=1)
    train_label_classes = torch.argmax(train_label.cuda(), dim=1)
    train_correct = (train_result_classes == train_label_classes).sum().item()
    train_total = torch.numel(train_result_classes)
    train_accuracy = train_correct / train_total

    running_loss["train_loss"].append(train_loss.item())
    running_loss["train_accuracy"].append(train_accuracy)

    torch.cuda.empty_cache()
    lr_scheduler.step()

    if epoch % val_interval == 0:
        val_loss = 0
        val_accuracy = 0
        net.eval()
        for i, (test_data, test_label) in enumerate(tqdm(valid_loader)):
            test_data, test_label_f = test_data.type(torch.FloatTensor).cuda(), test_label.type(
                torch.FloatTensor).cuda()

            with torch.no_grad():
                test_result = net(test_data)

                test_loss = criterion(test_result, test_label_f)
                val_loss += test_loss.item()

                test_result_classes = torch.argmax(test_result, dim=1)
                test_label_classes = torch.argmax(test_label.cuda(), dim=1)

                test_correct = (test_result_classes == test_label_classes).sum().item()
                test_total = torch.numel(test_result_classes)

                val_accuracy = test_correct / test_total

            torch.cuda.empty_cache()

        val_loss /= len(valid_loader)
        val_accuracy /= len(valid_loader)

        eval_loss["val_loss"].append(val_loss)
        eval_loss["val_accuracy"].append(val_accuracy)

        save_model(net_model=net, save_dir=save_dir, optimizer=None, ex="_iteration_{}".format(epoch))
        if val_loss <= loss_min:
            loss_min = val_loss
            io.save("Best Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, val_loss, val_accuracy),
                    os.path.join(save_dir, "best.txt"))
            save_model(net_model=net, save_dir=save_dir, optimizer=optimizer, ex="_best")

        io.save(running_loss, os.path.join(save_dir, "running_loss.bin"))
        io.save(eval_loss, os.path.join(save_dir, "eval_loss.bin"))

    pbar.set_description("train_loss:{:6}, train_acc: {:6}, val_loss:{:6}, val_acc:{:6}"
                         .format(train_loss.item(), train_accuracy, val_loss, val_accuracy))
    pbar.update()

pbar.close()
