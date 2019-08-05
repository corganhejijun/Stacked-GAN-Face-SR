import os
import cv2
from scipy import misc

data_path = os.path.join('datasets', 'yourData')
both_path = os.path.join(data_path, 'val_test')
test_path = os.path.join(data_path, 'test')
gt_path = os.path.join(data_path, 'gt')

if not os.path.isdir(test_path):
    os.mkdir(test_path)
if not os.path.isdir(gt_path):
    os.mkdir(gt_path)

totalLength = str(len(os.listdir(both_path)))
for index, file in enumerate(os.listdir(both_path)):
    print("processing " + file + " " + str(index+1) + " of total " + totalLength)
    imgFile = cv2.cvtColor(cv2.imread(os.path.join(both_path, file)), cv2.COLOR_BGR2RGB)
    imgWidth = int(imgFile.shape[1] / 2)
    imgGan = imgFile[:, :imgWidth, :]
    imgGt = imgFile[:, imgWidth:, :]
    misc.imsave(os.path.join(test_path, file[:-4] + ".png"), imgGan)
    misc.imsave(os.path.join(gt_path, file[:-4] + ".png"), imgGt)