# -----------------------------------------------------------------------
# gen_training_data.py:
# Training File Generator for prepared image files
#
# Creation Date   : 04/Aug./2017
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the GPL v2.0 License.
# 
# -----------------------------------------------------------------------

from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob
import cv2
#import cPickle as pickle # python 2.7
import _pickle as pickle # python 3.5
import matplotlib.pyplot as plt
import argparse
import random
from scipy import ndimage
import sys

parser = argparse.ArgumentParser(description='training dataset generator')
parser.add_argument('--pathfile', '-p', type=str, default='./imglist.txt',
                        help='Image File List (test file)')
parser.add_argument('--dataset', '-d', type=str, default='./hoge',
                        help='Pickle object for dataset output file name')
parser.add_argument('--size', '-s', type=int, default=32,
                        help='dataset size (default 32x32)')

# options for argumentation
parser.add_argument('--rotate', '-r', type=int, default=1,
                        help='Rotate')
parser.add_argument('--flip', '-f', type=str, default='no',
                        help='Flip')
parser.add_argument('--crop', '-c', type=int, default=1,
                        help='Crop')
parser.add_argument('--keepaspect', '-k', type=str, default='no',
                        help='Keep aspect ratio (default no)')

args = parser.parse_args()

dataset_fname = args.dataset + '_dataset.pkl'
label_fname = args.dataset + '_label.pkl'
tag_fname = args.dataset + '_tag.txt'


print("[INFO] IMAGE PATH FILE %s" % args.pathfile)
print("[INFO] DATASET FILE %s" % dataset_fname)
print("[INFO] LABEL FILE %s" % label_fname)
print("[INFO] TAG FILE %s" % tag_fname)

print("[INFO] DATASET SIZE %dx%d" % (int(args.size),int(args.size)))
print("[INFO] ROTATION %s" % args.rotate)
print("[INFO] FLIPPING %s" % args.flip)
print("[INFO] CROPPING %s" % args.crop)
print("[INFO] KEEP ASPECT RATIO %s" % args.keepaspect)

with open(args.pathfile, mode='r') as f:
	lines2 = f.readlines()
 
pathsAndLabels = []
label_idx = 0
tags = []
for line in lines2:
	words = line.split()
	tags.append(words[1])
	choped_line = words[0].rstrip('\n\r') + '/'
	pathsAndLabels.append(np.asarray([choped_line, label_idx]))
	print("[INFO] %s* are assigned to %d" % (choped_line, label_idx))
	label_idx = label_idx + 1

# fileout tags
f = open(tag_fname, 'w')
for x in tags:
	f.write(str(x) + "\n")
f.close()

# set data size
width = args.size
height = args.size

# get image path
allData = []
for pathAndLabel in pathsAndLabels:
	path = pathAndLabel[0]
	label = pathAndLabel[1]
	imagelist = glob.glob(path + "*")
	for imgName in imagelist:
		allData.append([imgName, label])

allData = np.random.permutation(allData)

# set augmentation options
n_crop = args.crop
n_rotate = args.rotate

if args.flip == 'yes' or args.rotate > 1:
	n_flip = 2
else:
	n_flip = 1

# register all images, and normalization if needs,,,
imageData = np.zeros((len(allData)*n_crop*n_rotate*n_flip,3,width,height))
labelData = np.zeros(len(allData)*n_crop*n_rotate*n_flip)

idx = 0
for pathAndLabel in allData:
	sys.stderr.write('\r\033[K' + "CONVERTING IMAGE %d/%d" % (idx,len(allData)*n_crop*n_rotate*n_flip))
	sys.stderr.flush()

	org_img = cv2.imread(pathAndLabel[0])

	if org_img is None:
		print("ERROR %s CANNOT BE OPENED" % pathAndLabel[0])
		exit()

	for i in range(n_crop):
		for k in range(n_flip):
			for j in range(n_rotate):
				# padding empy pixels to keep aspect ratio
				if args.keepaspect == 'yes':

					h, w = org_img.shape[:2]

					if h > w:
						dst_img = np.zeros((h,h,3)).astype(np.uint8) #* 128
						d = int((h-w)/2)
						dst_img[0:h,d:d+w] = org_img[:,:]
					else:
						dst_img = np.zeros((w,w,3)).astype(np.uint8) #* 128
						d = int((w-h)/2)
						dst_img[d:d+h,0:w] = org_img[:,:]

					org_img = dst_img

				# cropping
				if i > 0:
					h, w = org_img.shape[:2]

					if args.keepaspect == 'no':
						h4 = h / 4
						w4 = w / 4
						left = random.randint(0,w4)
						right = random.randint(w-w4,w)
						top = random.randint(0,h4)
						bottom = random.randint(h - h4,h)

						img = org_img[top:bottom,left:right] # y:y+h,x:x+h
					else:
						rows,cols = org_img.shape[:2]

						# resize with cropping
						dd = random.randint(0,rows/8)
						org_img = org_img[dd:rows-dd,dd:cols-dd]
						rows = rows - dd
						cols = cols - dd

						# sliding
						h4 = rows / 4
						w4 = cols / 4
						dw = random.randint(w4*(-1),w4)
						dh = random.randint(h4*(-1),h4)
						M = np.float32([[1,0,dw],[0,1,dh]])
						img = cv2.warpAffine(org_img,M,(cols,rows))

				else:
					img = org_img


				#flipping (if rotate, then flipping is also applied)
				if k == 0:
					pass
				else:
					img = cv2.flip(img, 1)

				# rotation
				img = ndimage.rotate( img, 2 * j, reshape=False)

				# Resize
				img = cv2.resize(img,(width,height))

				# Transpose for Chainer dataset
				reshaped = img.transpose(2, 0, 1) # (Y,X,BGR) -> (BGR,Y,X)

				# store temporary memory
				imageData[idx] = reshaped #bench
				labelData[idx] = np.int32(pathAndLabel[1])

				idx = idx + 1

imageData = imageData.astype(np.uint8)

# generate pickle file
threshold = np.int32(len(imageData)/10*9)

image = {}
label = {}
image['train'] = imageData[0:threshold]
image['test'] = imageData[threshold:]
label['train'] = labelData[0:threshold]
label['test'] = labelData[threshold:]

print("[INFO] SAVE %s as an image dataset" % dataset_fname)
with open(dataset_fname, mode='wb') as f:
	pickle.dump(image, f)

print("[INFO] SAVE %s as a label dataset" % label_fname)
with open(label_fname, mode='wb') as f:
	pickle.dump(label, f)

# -----------------------------------------------------------------------
# END OF PROGRAM
# -----------------------------------------------------------------------
