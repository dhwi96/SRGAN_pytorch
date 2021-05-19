#!/usr/bin/env python
import argparse
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim, MS_SSIM
from torch.autograd import Variable
#from IQA_pytorch import SSIM, utils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import configure, log_value
from models import Generator, Discriminator, FeatureExtractor
from utils import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=15, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

def get_psnr(img1, img2, min_value=0, max_value=255):
    if type(img1) == torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = max_value - min_value

    return 10 * math.log10((PIXEL_MAX ** 2) / mse)

"""
def get_ssim(img1, img2,  min_value=0, max_value=255):

    res_img1 = tf.image.convert_image_dtype(img1, tf.float32)
    res_img2 = tf.image.convert_image_dtype(img2, tf.float32)
    #res_img2 = cv2.resize(np.float32(img2), (256,256))
    
    ssim = tf.image.ssim(res_img1, res_img2, max_val=255, filter_size=256, filter_sigma=1.5, k1=0.01, k2=0.03)
    #result_ssim = ssim.numpy()

    return ssim
"""

try:
    os.makedirs(opt.out)
except OSError:
    pass

results = {'psnr': [],'ssim': []}

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])])

if opt.dataset == 'folder':
    # folder dataset
    dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transform)
elif opt.dataset == 'cifar100':
    dataset = datasets.CIFAR100(root=opt.dataroot, train=True, download=True, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

generator = Generator(16, opt.upSampling)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))
#print (generator)

discriminator = Discriminator()
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
#print (discriminator)

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=False))
print (feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

configure('logs/' + opt.dataset + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR), flush_secs=5)
visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# Pre-train generator using raw MSE loss
print ('Generator pre-training')
for epoch in range(1):
    mean_generator_content_loss = 0.0

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real, _ = data

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            low_res[j] = scale(high_res_real[j])
            high_res_real[j] = normalize(high_res_real[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))

        ######### Train generator #########
        generator.zero_grad()

        generator_content_loss = content_criterion(high_res_fake, high_res_real)
        mean_generator_content_loss += generator_content_loss.data

        generator_content_loss.backward()
        optim_generator.step()

        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, i, len(dataloader), generator_content_loss.data))
        #plt.savefig('test.png')
        #plt.show()
        visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

    sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, 2, i, len(dataloader), mean_generator_content_loss/len(dataloader)))
    log_value('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)
plt.savefig('test.png')

# Do checkpointing
torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

# SRGAN training
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR*0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1)

print ('SRGAN training')
for epoch in range(opt.nEpochs):

    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):

        # Generate data
        high_res_real, _ = data

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            low_res[j] = scale(high_res_real[j])
            high_res_real[j] = normalize(high_res_real[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)
        
        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data
        
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.data

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data
        
        generator_total_loss.backward()
        optim_generator.step()   
        
        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' 
        % (epoch, opt.nEpochs, i, len(dataloader),
        discriminator_loss.data, 
        generator_content_loss.data, 
        generator_adversarial_loss.data,
        generator_total_loss.data))
        # visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
    mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

    log_value('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
    log_value('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
    log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
    log_value('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
    torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)

high_res_real = Variable(torch.rand(1,1,256,256))
high_res_fake = Variable(torch.rand(1,1,256,256))
#plt.imshow(high_res_real)
#plt.imshow(high_res_fake)
#plt.show()

if torch.cuda.is_available():
    high_res_real = high_res_real.cuda()
    high_res_fake = high_res_fake.cuda()
    #plt.imshow(high_res_real)
    #plt.imshow(high_res_fake)
    #plt.show()

#ssim_model = SSIM(channels=3)
#score = ssim_model(high_res_fake, high_res_real, as_loss=False)

result_psnr = get_psnr(high_res_real, high_res_fake, 0, 255)
result_ssim = ms_ssim(high_res_fake, high_res_real, win_size=11, data_range=255, size_average=False)

#print(type(result_ssim))
print("SSIM : " + str(result_ssim))
#print ('Accuracy SSIM : %.4f' % score.item())
print('PSNR : %.4f' % result_psnr)
#print(tf.constant(result_ssim.cpu()))


plt.plot([0], [10], [20])
plt.ylabel('Accuracy')
plt.xlabel('Size')
plt.show()

os.exit()