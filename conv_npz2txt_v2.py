# -----------------------------------------------------------------------
# conv_npz2txt_v2.py:
# Convert to a binarized weight and an integer bias
#
# Creation Date   : 04/Aug./2017
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the GPL v2.0 License.
# 
# -----------------------------------------------------------------------

import pickle
from chainer import serializers
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='Weight converter')
parser.add_argument('--config_path', '-c', type=str, default='./hoge',
                        help='Configuration pickle file path')
args = parser.parse_args()

# load configuration from guiness GUI
config_file = args.config_path + "/config.pickle"
with open(config_file, mode='rb') as f:
	config = pickle.load(f)

initial_options = config['initial_options']
n_in_fmaps = config['n_in_fmaps']
n_ou_fmaps = config['n_ou_fmaps']
infmap_siz = config['infmap_siz']
ksiz = config['ksiz']
imgsiz = config['imgsiz']
max_dense_siz = config['max_dense_siz']
out_dense_siz = config['out_dense_siz']
bias_siz = config['bias_siz']
weight_siz = config['weight_siz']
max_bconv_width = config['max_bconv_width']
num_layer = config['num_layer']

model_file = args.config_path + "/temp.model"
dat = np.load(model_file) 

# convert .model to weights
dense_idx = 0
conv_idx = 0
bn_idx = 0

for layer in range(num_layer):
	# weights for convolutional layer
	if initial_options[layer] == 0 or initial_options[layer] == 1:
		key = 'conv%d/W' % conv_idx
		print("converting %s" % key)

		bincoef = np.where(dat[key]>=0,1,0).astype(dat[key].dtype, copy=False)

		bincoef2 = bincoef.reshape(-1,)

		# Text File Out
		fname = args.config_path + '/sdsoc/to_sd_card/conv%dW.txt' % conv_idx

		print(' Fileout (.txt) -> %s' % fname)
		np.savetxt(fname, bincoef2,fmt="%.0f",delimiter=",")

		# Header file out
		fname = args.config_path + '/HLS/conv%dW.csv' % conv_idx
		np.savetxt(fname, bincoef2[None,:],delimiter=",",fmt="%.0f")

		f = open(fname)
		line = f.read()
		f.close()

		header = 'ap_uint<1> t_bin_conv%dW[%d]={' % (conv_idx,len(bincoef2)) + line + '};' 

		fname = args.config_path + '/HLS/t_bin_conv%dW.h' % conv_idx
		print(' Fileout (HLS) -> %s' % fname)
		f = open(fname, 'w')
		f.write(header)
		f.close()

		# Update Index
		conv_idx += 1


	# weights for FC layer
	if initial_options[layer] == 4:
		key = 'fc%d/W' % dense_idx
		print("converting %s" % key)
		bincoef = np.where(dat[key]>=0,1,0).astype(dat[key].dtype, copy=False)

		bincoef2 = bincoef.reshape(-1,)

		#File out Textfile for SDSoC
		fname = args.config_path + '/sdsoc/to_sd_card/fc%dW.txt' % dense_idx

		print(' Fileout -> %s' % fname)
		np.savetxt(fname, bincoef2,fmt="%.0f",delimiter=",")

		# Fileout headerfile for HLS
		fname = args.config_path + '/HLS/fc%dW.csv' % dense_idx
		np.savetxt(fname, bincoef2[None,:],delimiter=",",fmt="%.0f")

		f = open(fname)
		line = f.read()
		f.close()

		header = 'ap_uint<1> t_bin_fc%dW[%d]={' % (dense_idx,len(bincoef2)) + line + '};' 

		fname = args.config_path + '/HLS/t_bin_fc%dW.h' % dense_idx
		print(' Fileout (HLS) -> %s' % fname)
		f = open(fname, 'w')
		f.write(header)
		f.close()

		# Update Index
		dense_idx += 1

	# bias
	if initial_options[layer] == 0 or initial_options[layer] == 1 or initial_options[layer] == 4:
		key = 'b%d' % bn_idx
		print("converting %s" % key)
		var = dat[key+'/avg_var']
		beta = dat[key+'/beta']
		gamma = dat[key+'/gamma']
		mean = dat[key+'/avg_mean']
		bn_val = np.floor((np.sqrt(var) * beta) / gamma - mean)

		txt_val = ''
		head_val = ''
		for ofeat in range(int(n_ou_fmaps[layer])):
			txt_val += "%d\n" % int(round(bn_val[ofeat],0))
			if ofeat != 0:
				head_val += ','
			head_val += "%d" % int(round(bn_val[ofeat],0))

		# Fileout Textfile for SDSoC
		fname = args.config_path + '/sdsoc/to_sd_card/b%d_BNFb.txt' % bn_idx

		print(' Fileout -> %s' % fname)
		with open(fname,'w') as f:
			f.write(txt_val)

		# Fileout headerfile for HLS
		fname = args.config_path + '/HLS/b%d_BNFb.h' % bn_idx

		if bn_idx == 0:
			header = 'ap_int<20> b%d_BNFb[%d] ={' % (bn_idx,int(n_ou_fmaps[layer])) + head_val + '};' 
		else:
			header = 'ap_int<16> b%d_BNFb[%d] ={' % (bn_idx,int(n_ou_fmaps[layer])) + head_val + '};' 

		print(' Fileout -> %s' % fname)
		with open(fname,'w') as f:
			f.write(header)

		# Update Index
		bn_idx += 1

# -----------------------------------------------------------------------
# END OF PROGRAM
# -----------------------------------------------------------------------
