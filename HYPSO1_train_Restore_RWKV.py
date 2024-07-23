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
val_interval = int(1e3)
save_interval = int(1e2)

lr = 2e-4

batch_size = 1
eps = 1e-8

loss_min = 0

data_root = "./dataset/1-DATA WITH GROUND-TRUTH LABELS"
save_dir = "experiment/Restore_RWKV"

img_size = (256, 256)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.Resize(img_size),
])

train_dataset = HYPSO1_Dataset(data_root, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

valid_dataset = HYPSO1_Dataset(data_root, train=False, transform=transforms.Resize(img_size))
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

modality_list = ["MRI"]

net = Restore_RWKV(inp_channels=120, out_channels=3, add_raw=False)
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps)
lr_scheduler = CosineAnnealingLR(optimizer, total_epoch, eta_min=1.0e-6)
criterion = nn.CrossEntropyLoss().cuda()

running_loss = []
eval_loss = {
    "val_loss": [],
    "val_accuracy": []
}

print("################ Train ################")
pbar = tqdm(total=int(total_epoch))
for epoch in list(range(1, int(total_epoch) + 1)):

    l_G = []
    train_data, train_label = next(DataSampler(train_loader))

    train_data, train_label = train_data.type(torch.FloatTensor).cuda(), train_label.type(torch.FloatTensor).cuda()

    net.train()
    optimizer.zero_grad()

    train_result = net(train_data)
    train_loss = criterion(train_result, train_label)

    train_loss.backward()
    optimizer.step()

    l_G.append(train_loss.item())
    torch.cuda.empty_cache()
    lr_scheduler.step()

    if epoch % save_interval == 0:
        save_model(net_model=net, save_dir=save_dir, optimizer=None, ex="_iteration_{}".format(epoch))

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

                result_classes = torch.argmax(test_result, dim=1)
                label_classes = torch.argmax(test_label, dim=1).cuda()

                correct = (result_classes == label_classes).sum().item()
                total = torch.numel(result_classes)

                val_accuracy = correct / total

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
        io.save(eval_loss, os.path.join(save_dir, "evaluationLoss.bin"))

    pbar.set_description("loss_G:{:6}, val_loss:{:6}, val_accuracy:{:6}"
                         .format(train_loss.item(), eval_loss["val_loss"][-1] if eval_loss else 0,
                                 eval_loss["val_accuracy"][-1] if eval_loss else 0))
    pbar.update()

pbar.close()
