import argparse
import time
import os

from os.path import basename, normpath

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from models import Generator

def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location=torch.device('cpu'))

#_load_from_bytes(b)	
parser = argparse.ArgumentParser(description='SR single image')
parser.add_argument('--lr', type=str, help='test image path')
parser.add_argument('--m', default=r'C:\Users\KIM\Desktop\making\checkpoints\generator_final.pth', type=str, help='model')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()

lr = opt.lr
pth = opt.m
with torch.no_grad():
	sr_path = 'generated/'
	if not os.path.exists(sr_path):
		os.makedirs(sr_path)
		
	model = Generator(16,opt.upSampling)

	#if torch.cuda.is_available():
	#	model.cuda()
	model.load_state_dict(torch.load(r'C:\Users\KIM\Desktop\making\checkpoints\generator_final.pth', map_location=torch.device('cpu')))
	image = Image.open(lr)
	image = Variable(ToTensor()(image)).unsqueeze(0)
	
	#if torch.cuda.is_available():
	#	image = image.cuda()

	out = model(image)
	out_img = ToPILImage()(out[0].data.cpu())
	#out_img.save(sr_path + basename(normpath(lr)))