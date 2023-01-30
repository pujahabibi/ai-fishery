from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
#from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import torchvision.transforms as transforms

import albumentations as A

from PIL import Image
import  numpy as np

import cv2

# cv2.setNumThreads(0)
#
# cv2.ocl.setUseOpenCL(False)

class AAugmentation(object):

    def __init__(self, augmentation):

        self.augmentation = augmentation

    def __call__(self, img):

        #print(img)

        img = np.array(img)

        for a in self.augmentation:

            img = a(image=img)['image']
        #print(type(img))
        return img#Image.fromarray((img*255).astype(np.unit8))


    def __repr__(self):

        return self.__class__.__name__ + '(augmentation={0})'.format(self.augmentation)

class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)

        self.size = size

        self.interpolation = interpolation

    def __call__(self, img):

        """

        Args:

            img (PIL Image): Image to be scaled.

​

        Returns:

            PIL Image: Rescaled image.

        """

        return resize(img, self.size, self.interpolation)

    def __repr__(self):

        interpolate_str = _pil_interpolation_to_str[self.interpolation]

        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def resize(img, size, interpolation=Image.BILINEAR):


    if isinstance(size, int):

        w, h = img.size


        # if (w == size) and (h == size):

        #     return img

        # elif w > h:

        #     ow = size

        #     oh = int(size * h / w)

        #     if oh < 10:

        #         oh = 10

        #     return img.resize((ow, oh), interpolation)

        # elif h > w:

        oh = size

        ow = int(size * w / h)

        if ow < 10:

            ow = 10

        return img.resize((ow, oh), interpolation)

        # else:

        #     return img.resize((size, size), interpolation)

    else:

        return img.resize(size[::-1], interpolation)

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print(model)
  #print(summary(model.cuda(), (3, 512, 512)))
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  augmentationTransforms = A.Compose([

    #AAugmentation([A.ChannelShuffle(p=1.)]),

    #AAugmentation([A.InvertImg(p=1.)]),

    #AAugmentation([A.Blur(p=1.)]),

    AAugmentation([A.RandomBrightnessContrast(p=1.)]),

    #AAugmentation([A.MedianBlur(blur_limit=3)]),
    AAugmentation([A.RGBShift()]),
    AAugmentation([A.RandomGamma(p=0.5)]),


    AAugmentation([A.CLAHE(p=1.)]),

    AAugmentation([A.HueSaturationValue(p=1.)]),

    # AAugmentation([A.ElasticTransform(sigma=1, alpha_affine=2, border_mode=0, p=1.)]),

    #AAugmentation([A.CoarseDropout(max_holes=50, max_height=2, max_width=2, p=1.)]),

    transforms.RandomGrayscale(p=1.0),

    # transforms.RandomRotation(degrees=15),

  ])

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=0,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train', transform=augmentationTransforms),
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
