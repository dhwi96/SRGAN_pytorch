import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from models import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='.\checkpoints\generator_final.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(16,opt.upscale_factor).eval()
#if opt.cuda:
#    model.cuda()
#    model.load_state_dict(torch.load(MODEL_NAME))
#    image = image.cuda()
#else:
model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))
#image = image

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

start = time.time()
out = model(image)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(opt.upscale_factor) + '_' + IMAGE_NAME)