import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from loaddata import kittidata

import numpy as np
import time
import os

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

n_class = 12

batch_size = 4
epochs = 500
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5
configs = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

train_rgb_path = './kitti_semseg_unizg/train/rgb'
train_depth_path = './kitti_semseg_unizg/train/depth'
train_label_path = './kitti_semseg_unizg/train/labels'

test_rgb_path = './kitti_semseg_unizg/test/rgb'
test_depth_path = './kitti_semseg_unizg/test/depth'
test_label_path = './kitti_semseg_unizg/test/labels'

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = kittidata(train_rgb_path, train_label_path, 0.8, transform=tf)


train_loader = DataLoader(dataset.setphase('train'), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset.setphase('val'), batch_size=1, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vgg_model = VGGNet(requires_grad=True, model='vgg11', remove_fc=True).to(device)
fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class).to(device)

print('data loading finished')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            ti = time.time()
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, time {} sec, loss: {}".format(epoch, iter, time.time()-ti, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    tol_time = 0
    with torch.no_grad():
        for iter, (inputs, labels) in enumerate(val_loader):
            ti = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            output = fcn_model(inputs)
            output = output.data.cpu().numpy()

            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            writer.add_image('result %d'%(iter), inputs)

            target = labels.cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

            ti = time.time() - ti
            tol_time += ti
            writer.add_scalar('time', ti, iter)
            print('Iteration {} takes {:.2f} sec'.format(iter, ti))
    print('total valuation time is {:.2f}'.format(tol_time))
    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
            # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    train()