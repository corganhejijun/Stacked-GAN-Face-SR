# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
from PIL import Image

dataset_dir = 'datasets'
dataset = 'data_crop_512_jpg'
train_dir = 'train'
test_dir = 'val_test'
val_dir = 'val'
resize_to = 256
scale = 4

if not os.path.exists(os.path.join(dataset_dir, train_dir)):
    os.mkdir(os.path.join(dataset_dir, train_dir))
if not os.path.exists(os.path.join(dataset_dir, test_dir)):
    os.mkdir(os.path.join(dataset_dir, test_dir))
if not os.path.exists(os.path.join(dataset_dir, val_dir)):
    os.mkdir(os.path.join(dataset_dir, val_dir))

fileList = os.listdir(os.path.join(dataset_dir, dataset))
for index, file in enumerate(fileList):
    imgPath = os.path.join(dataset_dir, dataset, file)
    if os.path.isdir(imgPath):
        continue
    print("procesing " + file + " " + str(index+1) + '/' + str(len(fileList)))
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img = misc.imresize(img, (resize_to, resize_to), interp='bilinear')
    size = int(img.shape[0] / scale)
    resizeImg = misc.imresize(img, (size, size), interp='bilinear')
    finalImg = misc.imresize(resizeImg, (img.shape[0], img.shape[0]), interp='bilinear')
    combineImg = Image.new('RGB', (img.shape[0]*2, img.shape[0]))
    combineImg.paste(Image.fromarray(finalImg), (0,0))
    combineImg.paste(Image.fromarray(img), (img.shape[0]+1,0))
    savePath = ""
    # sample ratio of train:test:val is 6:2:2
    if index % 10 < 6:
        savePath = os.path.join(dataset_dir, train_dir, file)
    elif index % 10 < 8:
        savePath = os.path.join(dataset_dir, test_dir, file)
    else:
        savePath = os.path.join(dataset_dir, val_dir, file)
    misc.imsave(savePath, combineImg)
