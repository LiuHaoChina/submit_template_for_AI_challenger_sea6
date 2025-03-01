import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import csv
import os
import time
import perlin_noise

topk = 10
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

parser = argparse.ArgumentParser('Running script', add_help=False)
parser.add_argument('--input_dir', default='../input_dir', type=str)
parser.add_argument('--output_dir', default='../output_dir', type=str)

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, meta_file, transform=None):

        self.data_dir = data_dir
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []

        with open(os.path.join(data_dir, meta_file)) as f:
            for line in f.readlines()[1:]:
                img_path, label = line.strip().split(',')
                self._indices.append((os.path.join(self.data_dir, 'images', img_path), label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        img_name = img_path.split('/')[-1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_name


def tensor2img(input_tensor, save_dir, save_name):

    if input_tensor.is_cuda == True:
        input_tensor = input_tensor.cpu()

    input_tensor = input_tensor.permute(0, 2, 3, 1).data.numpy()
    for i in range(input_tensor.shape[0]):
        Image.fromarray((input_tensor[i] * 255).astype(np.uint8)).save('{}/{}'.format(save_dir, save_name[i]))
        print('{} saved in {}.'.format(save_name[i], save_dir))



# 对图像进行分类
def get_classify_result(imgs, models):
    result_labels = []
    # 查询模型们，得到攻击后的结果
    for model in models:
        result_model = model(imgs)
        _, result_model_label = torch.topk(result_model, topk)
        result_labels.append(result_model_label)
    return result_labels


# 调用generate进行攻击
class PGD(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def generate(self, x, **params):
        self.parse_params(**params)
        adv_x = self.attack(x, self.y)
        return adv_x

    def parse_params(self, eps=0.5, iter_eps=0.01, nb_iter=40, clip_min=-2.12, clip_max=2.25, C=0.0,
                     y=None, ord=np.inf, rand_init=True, flag_target=False):
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.ord = ord
        self.rand_init = rand_init
        self.flag_target = flag_target
        self.C = C

    def sigle_step_attack(self, x, pertubation, labels):
        adv_x = x + pertubation
        # 获取x的梯度数据
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True
        loss_func = nn.CrossEntropyLoss()

        # 下面把每个模型的loss计算出来，相加
        preds_0 = self.models[0](adv_x)
        loss = loss_func(preds_0, labels)
        self.models[0].zero_grad()

        for j in range(0, len(self.models)-1):
            index = j + 1
            preds = self.models[index](adv_x)
            loss = loss + loss_func(preds, labels)
            self.models[index].zero_grad()
            preds = None

        loss.backward()
        grad = adv_x.grad.data

        pertubation = pertubation + self.iter_eps * np.sign(grad.cpu()).to(device)
        adv_x = adv_x.detach() + pertubation
        x = x.detach()
        pertubation = torch.clamp(adv_x, self.clip_min, self.clip_max) - x
        pertubation = clip_pertubation(pertubation, self.ord, self.eps)

        return pertubation

    def attack(self, x, labels):

        x_tmp = torch.clone(x)
        # 噪声初始化
        noise = perlin_noise.create_perlin_noise(px=500)
        noise = torch.tensor(noise[0])
        noise = noise.to(device)
        noise = noise.permute(2, 0, 1)
        pertubation = torch.unsqueeze(noise, 0)

        # 需要其 正确tag 跌出所有模型的 前10 便停止递进 or 循环超过100次
        for i in range(self.nb_iter):
            pertubation = self.sigle_step_attack(x_tmp, pertubation=pertubation, labels=labels)
            x_test = x + pertubation
            # 查看图像tag是否已经跌出前10，全都跌出前10便认为攻击成功
            suc = True
            res = get_classify_result(x_test, models)
            for result in res:
                if labels[0] in result:
                    suc = False
                    break
            if suc:
                break
        adv_x = x + pertubation
        adv_x = adv_x.detach()

        adv_x = adv_x.clamp(self.clip_min, self.clip_max)

        return adv_x


def clip_pertubation(eta, norm, eps):
    """
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=device), eps / norm
        )
        eta *= factor
    return eta


mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)


# 对图像进行反归一化
def un_normalized_img_and_save(img, path):
    # img = img[0, :, :, :]
    img = img * std + mean
    img = torch.unsqueeze(img, 0)
    torchvision.utils.save_image(img, path+'.png')
    os.rename(path+'.png', path)


# 获取模型们
def get_models():
    model_1 = torchvision.models.densenet161(pretrained=True).to(device)
    # model_2 = torchvision.models.resnet152(pretrained=True).to(device)

    model_3 = torchvision.models.resnet50()
    model_3 = torch.nn.DataParallel(model_3)
    check_point_3 = torch.load('code/imagenet_model_weights_4px.pth')
    model_3.load_state_dict(check_point_3['state_dict'])
    model_3.to(device)

    model_4 = torchvision.models.resnet50()
    model_4 = torch.nn.DataParallel(model_4)
    check_point_4 = torch.load('code/model_best.pth.tar')
    model_4.load_state_dict(check_point_4['state_dict'])
    model_4.to(device)

    # model_5 = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet85', pretrained=True).to(device)
    # model_6 = torchvision.models.googlenet(pretrained=True).to(device)
    #
    # model_7 = torchvision.models.wide_resnet50_2(pretrained=True).to(device)
    model_8 = torchvision.models.inception_v3(pretrained=True).to(device)
    # model_9 = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True).to(device)
    # model_10 = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl', pretrained=True).to(device)
    # model_11 = torchvision.models.resnext101_32x8d(pretrained=True).to(device)
    model_12 = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True).to(device)

    model_1.eval()
    # model_2.eval()
    model_3.eval()
    model_4.eval()
    # model_5.eval()
    # model_6.eval()
    #
    # model_7.eval()
    model_8.eval()
    # model_9.eval()
    # model_10.eval()
    # model_11.eval()
    model_12.eval()

    target_models = [
                    model_1,
                    # model_2,
                    model_3,
                    model_4,
        #              model_5,
        #              model_6,
        #              model_7,
                     model_8,
        #              model_9,
        #              model_10,
        #              model_11,
                     model_12,
                     ]
    return target_models


# 保存图像
def save_imgs(img_names, result_imgs, labels, models, dir_path, count):
    # 查询模型们，得到攻击后的结果
    result_labels = get_classify_result(result_imgs, models)

    for j in range(0, len(img_names)):
        for index in range(0, len(models)):
            # 检查对抗性
            if labels[j] in result_labels[index][j]:
                attack_suc = False
                count = count + 1
                print(f'img={img_names[j]} 攻击失败！图片标签={labels[j]} model_{index + 1} 识别标签={result_labels[index][j]} 当前总计失败次数={count}次')
                break
        un_normalized_img_and_save(result_imgs[j], os.path.join(dir_path, img_names[j]))

    return count


if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 此种归一化，最大值约为2.2489，最小值约为-2.1179
    ])
    # 加载模型们
    models = get_models()
    pgd = PGD(models)

    dataset = ImageNetDataset(data_dir=args.input_dir, meta_file='dev.csv', transform=data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    count = 0
    for i, (ori_imgs, labels, img_names) in enumerate(data_loader):
        loop_start = time.time()
        # 加载至加速计算设备
        ori_imgs = ori_imgs.to(device)
        labels = labels.to(device)
        # 使用PGD生成对抗图像
        adv_imgs = pgd.generate(x=ori_imgs, y=labels)
        # 保存对抗图像
        count = save_imgs(img_names, adv_imgs, labels, models, args.output_dir, count)
        print(f'img_names={img_names} 攻击完成，耗时={time.time() - loop_start} 共计失败{count}次\n')

    print(f'攻击完成，耗时={time.time() - start} 攻击失败{count}次\n')

    
