import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from loaddata import kittidata, kittidata_split

import numpy as np
import time
import os

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

n_class = 12

batch_size = 8
multi_thread_loader = 0
epochs = 500
lr = 1e-3
lr_pretrain = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 10
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

kittiset = kittidata_split(train_rgb_path, train_label_path, 0.7)

train_data, train_label = kittiset.getdata('train')
val_data, val_label = kittiset.getdata('val')

train_set = kittidata(train_data, train_label, shrink_rate=0.6, flip_rate=0.5)
val_set = kittidata(val_data, val_label, shrink_rate=1, flip_rate=0)  # keep data unchanged
print('{} for training, {} for validation'.format(len(train_data), len(val_data)))

# set the validation batch_size equal to the device amount to fully utilize the (multi)GPU
device_amount = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
print('there are {} devices'.format(device_amount))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=multi_thread_loader)
val_loader = DataLoader(val_set, batch_size=device_amount, shuffle=False, num_workers=multi_thread_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg_model = VGGNet(requires_grad=True, model='vgg11', remove_fc=True).to(device)
fcn_model = nn.DataParallel(FCN8s(pretrained_net=vgg_model, n_class=n_class)).to(device)

print('data loading finished')

criterion = nn.NLLLoss()

params = list()
for name, param in fcn_model.named_parameters():
    if 'pretrained_net' in name:  # use small learning rate for
        params += [{'params': param, 'lr': lr_pretrain}]
    else:
        params += [{'params': param, 'lr': lr}]

optimizer = optim.Adam(params, weight_decay=w_decay)
optimizer = nn.DataParallel(optimizer)
scheduler = lr_scheduler.StepLR(optimizer.module, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

total_train_set = len(train_data)
total_iter = len(train_loader)
def train():
    ti = time.time()
    for epoch in range(epochs):
        scheduler.step()
        epoch_loss = 0
        ts = time.time()
        for iter, (inputs, _, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = fcn_model(inputs)
            loss = criterion(F.log_softmax(outputs, dim=1), labels).to(device)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.module.step()

            writer.add_scalar('single_iter_loss', loss.item(), epoch * total_iter + iter)
            if iter % 10 == 0:
                print("epoch{}, iter{}, time elapsed {:.1f} sec".format(epoch, iter, time.time()-ti))
        mel = epoch_loss / total_train_set
        writer.add_scalar('mean_epoch_loss', mel, epoch)
        print("Epoch {} takes {} with mean loss {:.5f}".format(epoch, time.time() - ts, mel))
        torch.save(fcn_model, model_path + '_epoch{}'.format(epoch))  # store the model each epoch

        val(epoch)


def val(epoch):
    print('validation starts')
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    tol_time = 0
    with torch.no_grad():
        for iter, (inputs, labels, num_labels) in enumerate(val_loader):
            ti = time.time()
            inputs = inputs.to(device)
            output = fcn_model(inputs)
            output = output.data.cpu().numpy()

            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            if iter == 0:
                writer.add_image('result of {}'.format(iter), image_grid(inputs, pred, num_labels), epoch)

            target = num_labels.numpy()
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

            ti = time.time() - ti
            tol_time += ti
            break
    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    writer.add_scalar('pixel_acc', pixel_accs, epoch)
    writer.add_scalar('meanIoU', np.nanmean(ious), epoch)

    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    print('total validation time is {:.2f}'.format(tol_time))


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


# align images (batch, orginial, pred, ground_truth)
def image_grid(image, pred, label):
    l = list()
    for i in range(image.shape[0]):
        l.extend([train_set.denormalize(image[i]), train_set.visualize(pred[i]), train_set.visualize(label[i])])
    return utils.make_grid(l, nrow=3)

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    #train()