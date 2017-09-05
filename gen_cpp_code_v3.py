# -----------------------------------------------------------------------
# gen_cpp_code_v3.py
# C++ code generator for a high-level synthesis toward an FPGA realization
#
# Creation Date   : 04/Aug./2017
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the GPL v2.0 License.
#
# -----------------------------------------------------------------------

#!/usr/bin/python
# coding: UTF-8

import argparse
import re
import pickle

parser = argparse.ArgumentParser(description='C++ code generator')
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
numimg = int(n_in_fmaps[0])

#(SET_WEIGHT_MEM)
set_weight_mem = ''
set_bias_mem = ''
bconv_reg_pragma = ''
bconv_reg_select = ''
bconv_weight_select = ''
bconv_bias_select = ''

conv_idx = 0
bn_idx = 0
dense_idx = 0
offset_weight = 0
offset_bias = 0

#(DEF_CNN_LAYER)
from collections import Counter
def_cnn_layer = ''

bn_idx = 0
dense_idx = 0
counter = Counter(initial_options)
for layer_type, cnt in counter.items():
	if layer_type == 0 and cnt > 0:
		for i in range(len(initial_options)):
			if initial_options[i] == 0:
				def_cnn_layer += '            case %d:\n' % i
		def_cnn_layer += '            int_conv2d_layer<bit_64, bit_%d, 64, %d, %d, %d>\n            ( in_img, fb_tmp, conv0W, b0_BNFb);\n            break;\n' % (max_bconv_width,max_bconv_width,int(infmap_siz[0]),int(infmap_siz[0]))

	elif layer_type == 1 and cnt > 0:
		for i in range(len(initial_options)):
			if initial_options[i] == 1:
				def_cnn_layer += '            case %d:\n' % i
		def_cnn_layer += '            bin_conv2d_pipeline(fb_tmp,bin_layer_idx,fsize[layer],n_in[layer],n_out[layer]);\n            bin_layer_idx++;\n            break;\n'

	elif layer_type == 2 and cnt > 0:
		for i in range(len(initial_options)):
			if initial_options[i] == 2:
				def_cnn_layer += '            case %d:\n' % i
		def_cnn_layer += '            max_pooling_layer<bit_%d, %d, %d>(fb_tmp);\n            break;\n' % (max_bconv_width,int(imgsiz),int(infmap_siz[i]))

	elif layer_type == 3 and cnt > 0:
		for i in range(len(initial_options)):
			if initial_options[i] == 3:
				def_cnn_layer += '            case %d:\n' % i
				def_cnn_layer += '            {\n'
				def_cnn_layer += '                ap_int<%d>mask = 0x1;\n' % int(n_in_fmaps[i])
				def_cnn_layer += '                for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
				def_cnn_layer += '                	ap_int<11> tmp = 0;\n'
				def_cnn_layer += '                	for( y = 0; y < %d; y++){\n' % int(infmap_siz[i])
				def_cnn_layer += '                		for( x = 0; x < %d; x++){\n' % int(infmap_siz[i])
				def_cnn_layer += '                			if( (fb_tmp[y][x] & mask) != 0)\n'
				def_cnn_layer += '                				tmp++;\n'
				def_cnn_layer += '                		}\n'
				def_cnn_layer += '                	}\n'
				def_cnn_layer += '                	if( tmp >= %d*%d/2)\n' % (int(infmap_siz[i]),int(infmap_siz[i]))
				def_cnn_layer += '                		fc_tmp[of] = 1;\n'
				def_cnn_layer += '                	else\n'
				def_cnn_layer += '                		fc_tmp[of] = 0;\n'
				def_cnn_layer += '                	mask = mask << 1;\n'
				def_cnn_layer += '                }\n                }\n            break;\n'
	
	elif layer_type == 4 and cnt > 0:
		for i in range(len(initial_options)):
			if initial_options[i] == 4:
				def_cnn_layer += '            case %d:\n' % i
				def_cnn_layer += '            fc_layer< %d, %d>( fc_tmp, fc%dW, b%d_BNFb, fc_result);\n            break;\n' % (int(n_ou_fmaps[i]),int(n_in_fmaps[i]),dense_idx,bn_idx)
				bn_idx += 1
				dense_idx += 1
			elif initial_options[i] == 0 or initial_options[i] == 1:
				bn_idx += 1

def_cnn_layer += '            default: break;\n'

#(DEF_CNN_PARAMETER)
def_cnn_parameter = '    int fsize[%d] = {' % (len(initial_options))
for i in range(len(initial_options)):
	if i != 0:
		def_cnn_parameter += ','
	def_cnn_parameter += '%3d' % int(infmap_siz[i])
def_cnn_parameter += '};\n'
def_cnn_parameter += '    int n_in[%d]  = {' % (len(initial_options))
for i in range(len(initial_options)):
	if i != 0:
		def_cnn_parameter += ','
	def_cnn_parameter += '%3d' % int(n_in_fmaps[i])
def_cnn_parameter += '};\n'
def_cnn_parameter += '    int n_out[%d] = {' % (len(initial_options))
for i in range(len(initial_options)):
	if i != 0:
		def_cnn_parameter += ','
	def_cnn_parameter += '%3d' % int(n_ou_fmaps[i])
def_cnn_parameter += '};\n'

#(BCONV_REG_SELECT)
#(BCONV_WEIGHT_SELECT)
#(BCONV_BIAS_SELECT)
conv_idx = 0
for i in range(len(initial_options)):
	if initial_options[i] == 0:
		conv_idx += 1
	if initial_options[i] == 1:
		bconv_reg_select += '        case  %d: shift_reg1[ 2 * (%d+2) + 3 - 1] = din; break;\n' % (conv_idx,int(infmap_siz[i]))

		bconv_weight_select += '                        case %d:\n' % conv_idx
		bconv_weight_select += '                            bx = shift_reg1[ky * (%d+2) + kx];\n' % int(infmap_siz[i])
		bconv_weight_select += '                            bw = (ap_uint<%d>)conv%dW[ofeat][ky*3+kx];\n' % (max_bconv_width,conv_idx)
		bconv_weight_select += '                            mask = ~(~allzero << %d);\n' % int(n_in_fmaps[i])
		bconv_weight_select += '                        break;\n'

		bconv_bias_select += '            	case %d:  bias = b%d_BNFb[ofeat]; break;\n' % (conv_idx,conv_idx)

		conv_idx += 1
bconv_reg_select += '        default: break;\n'
bconv_weight_select += '                        default: break;\n'
bconv_bias_select += '            	default: break;\n'

#(BCONV_REG_PRAGMA)
conv_idx = 0
for i in range(len(initial_options)):
	if initial_options[i] == 0:
		conv_idx += 1
	if initial_options[i] == 1:
		bconv_reg_pragma += '    #pragma HLS ARRAY_PARTITION variable=conv%dW cyclic factor=9 dim=2\n' % conv_idx
		conv_idx += 1

conv_idx = 0
bn_idx = 0
dense_idx = 0

#(READ_WEIGHT_MEM)


#(READ_BIAS_MEM)
read_bias_mem = ''
read_weight_mem = ''

def_weight_mem = ''
def_bias_mem = ''

for i in range(len(initial_options)):
	if initial_options[i] == 0 or initial_options[i] == 1:
		set_weight_mem += '    printf("load conv%dW\\n");\n' % conv_idx
		set_weight_mem += '    offset = %d;\n' % offset_weight
		set_weight_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		set_weight_mem += '        for( y = 0; y < 3; y++){\n'
		set_weight_mem += '            for( x = 0; x < 3; x++){\n'
		set_weight_mem += '                ap_uint<%d>tmp = 0x1;\n' % int(n_in_fmaps[i])
		set_weight_mem += '                for( inf = 0; inf < %d; inf++){\n' % int(n_in_fmaps[i])
		set_weight_mem += '                     if( t_bin_convW[of*%d*3*3+inf*3*3+y*3+x+offset] == 1){\n' % int(n_in_fmaps[i])
		set_weight_mem += '                         conv%dW[of][y*3+x] |= tmp;\n' % conv_idx
		set_weight_mem += '                     }\n'
		set_weight_mem += '                tmp = tmp << 1;\n'
		set_weight_mem += '                }\n'
		set_weight_mem += '            }\n'
		set_weight_mem += '        }\n'
		set_weight_mem += '    }\n'

		set_bias_mem += '    printf("load b%d_BNFb\\n");\n' % bn_idx
		set_bias_mem += '    offset = %d;\n' % offset_bias
		set_bias_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		set_bias_mem += '        b%d_BNFb[of] = t_BNFb[of+offset];\n' % bn_idx
		set_bias_mem += '    }\n'

		read_weight_mem += '    printf("conv%dW.txt\\n");\n' % conv_idx
		read_weight_mem += '    if( (fp = fopen("conv%dW.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\\n");\n' % conv_idx
		read_weight_mem += '    offset = %d;\n' % offset_weight
		read_weight_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		read_weight_mem += '        for( inf = 0; inf < %d; inf++){\n' % int(n_in_fmaps[i])
		read_weight_mem += '            for( y = 0; y < 3; y++){\n'
		read_weight_mem += '                for( x = 0; x < 3; x++){\n'
		read_weight_mem += '                    if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\\n"); sscanf( line, "%d", &d_value);\n'
		read_weight_mem += '                    t_bin_convW[of*%d*3*3+inf*3*3+y*3+x+offset] = d_value;\n' % int(n_in_fmaps[i])
		read_weight_mem += '                }\n'
		read_weight_mem += '            }\n'
		read_weight_mem += '        }\n'
		read_weight_mem += '    }\n'
		read_weight_mem += '    fclose(fp);\n'

		read_bias_mem += '    printf("b%d_BNFb.txt\\n");\n' % bn_idx
		read_bias_mem += '    if( (fp = fopen("b%d_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\\n");\n' % bn_idx
		read_bias_mem += '    offset = %d;\n' % offset_bias
		read_bias_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		read_bias_mem += '        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\\n");\n'
		read_bias_mem += '        sscanf( line, "%d", &d_value);\n'
		read_bias_mem += '        t_BNFb[of+offset] = d_value;\n'
		read_bias_mem += '    }\n'
		read_bias_mem += '    fclose(fp);\n'

		def_weight_mem += 'ap_int<%d>  conv%dW[%d][3*3];\n' % (int(n_in_fmaps[i]),conv_idx,int(n_ou_fmaps[i]))
		if initial_options[i] == 0:
			def_bias_mem += 'ap_int<20> b%d_BNFb[%d];\n' % (bn_idx,int(n_ou_fmaps[i]))
		else:
			def_bias_mem += 'ap_int<16> b%d_BNFb[%d];\n' % (bn_idx,int(n_ou_fmaps[i]))

		conv_idx += 1
		bn_idx += 1
		offset_weight += (int(n_in_fmaps[i]) * int(n_ou_fmaps[i]) * 3 * 3)
		offset_bias += int(n_ou_fmaps[i])
	elif initial_options[i] == 4:
		set_weight_mem += '    printf("load fc%dW\\n");\n' % dense_idx
		set_weight_mem += '    offset = %d;\n' % offset_weight
		set_weight_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		set_weight_mem += '        for( inf = 0; inf < %d; inf++){\n' % int(n_in_fmaps[i])
		set_weight_mem += '            fc%dW[of][inf] = (ap_int<1>)t_bin_convW[of*%d+inf+offset];\n' % (dense_idx,int(n_in_fmaps[i]))
		set_weight_mem += '        }\n'
		set_weight_mem += '    }\n'

		set_bias_mem += '    printf("load b%d_BNFb\\n");\n' % bn_idx
		set_bias_mem += '    offset = %d;\n' % offset_bias
		set_bias_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		set_bias_mem += '        b%d_BNFb[of] = t_BNFb[of+offset];\n' % bn_idx
		set_bias_mem += '    }\n'

		read_weight_mem += '    printf("fc%dW.txt\\n");\n' % dense_idx
		read_weight_mem += '    if( (fp = fopen("fc%dW.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\\n");\n' % dense_idx
		read_weight_mem += '    offset = %d;\n' % offset_weight
		read_weight_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		read_weight_mem += '        for( inf = 0; inf < %d; inf++){\n' % int(n_in_fmaps[i])
		read_weight_mem += '            if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\\n"); sscanf( line, "%d", &d_value);\n'
		read_weight_mem += '            t_bin_convW[of*%d+inf+offset] = d_value;\n' % int(n_in_fmaps[i])
		read_weight_mem += '        }\n'
		read_weight_mem += '    }\n'
		read_weight_mem += '    fclose(fp);\n'

		read_bias_mem += '    printf("b%d_BNFb.txt\\n");\n' % bn_idx
		read_bias_mem += '    if( (fp = fopen("b%d_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\\n");\n' % bn_idx
		read_bias_mem += '    offset = %d;\n' % offset_bias
		read_bias_mem += '    for( of = 0; of < %d; of++){\n' % int(n_ou_fmaps[i])
		read_bias_mem += '        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\\n");\n'
		read_bias_mem += '        sscanf( line, "%d", &d_value);\n'
		read_bias_mem += '        t_BNFb[of+offset] = d_value;\n'
		read_bias_mem += '    }\n'
		read_bias_mem += '    fclose(fp);\n'

		def_weight_mem += 'ap_int<1>  fc%dW[%d][%d];\n' % (dense_idx,int(n_ou_fmaps[i]),int(n_in_fmaps[i]))
		def_bias_mem += 'ap_int<16> b%d_BNFb[%d];\n' % (bn_idx,int(n_ou_fmaps[i]))


		dense_idx += 1
		bn_idx += 1
		offset_weight += (int(n_in_fmaps[i]) * int(n_ou_fmaps[i]))
		offset_bias += int(n_ou_fmaps[i])

# Check # of f.maps
bin_xor_mac = 'bxor = (ap_uint<(MAX_BCONV_WIDTH)>)(bx ^ bw);'
for i in range(len(initial_options) - 2):
	if int(n_in_fmaps[i+1]) != int(n_in_fmaps[i+2]):
		bin_xor_mac = 'bxor = (ap_uint<(MAX_BCONV_WIDTH)>)(bx ^ bw) & mask;'

# generate C++ code for a binarized CNN ------------------------------------
f = open('template_cpp_r7_bcnn.cpp')
lines2 = f.readlines()
f.close()

cpp_file = ''

for line in lines2:
    converted = line.replace("(BIAS_SIZ)",str(bias_siz))
    converted = converted.replace("(BIN_XOR_MAC)",bin_xor_mac)
    converted = converted.replace("(KSIZ)",str(ksiz))
    converted = converted.replace("(MAX_DENSE_SIZ)",str(max_dense_siz))
    converted = converted.replace("(OUT_DENSE_SIZ)",str(out_dense_siz))
    converted = converted.replace("(WEIGHT_SIZ)",str(weight_siz))
    converted = converted.replace("(MAX_BCONV_WIDTH)",str(max_bconv_width))
    converted = converted.replace("(NUM_LAYER)",str(num_layer))
    converted = converted.replace("(IMGSIZ)",str(imgsiz))
    converted = converted.replace("(NUMIMG)",str(numimg))

    converted = converted.replace("(BCONV_REG_PRAGMA)",bconv_reg_pragma)
    converted = converted.replace("(BCONV_REG_SELECT)",bconv_reg_select)
    converted = converted.replace("(BCONV_BIAS_SELECT)",bconv_bias_select)
    converted = converted.replace("(BCONV_WEIGHT_SELECT)",bconv_weight_select)
    converted = converted.replace("(DEF_CNN_PARAMETER)",def_cnn_parameter)
    converted = converted.replace("(DEF_CNN_LAYER)",def_cnn_layer)
    converted = converted.replace("(DEF_BIAS_MEM)",def_bias_mem)
    converted = converted.replace("(DEF_WEIGHT_MEM)",def_weight_mem)
    converted = converted.replace("(SET_BIAS_MEM)",set_bias_mem)
    converted = converted.replace("(SET_WEIGHT_MEM)",set_weight_mem)
    converted = converted.replace("(READ_BIAS_MEM)",read_bias_mem)
    converted = converted.replace("(READ_WEIGHT_MEM)",read_weight_mem)

    cpp_file += converted
    
cnn_file = args.config_path + "/sdsoc/cnn.cpp"
with open(cnn_file,'w') as f:
	f.write(cpp_file)

# generate C++ main code ---------------------------------------------------
f = open('template_cpp_r7_main.cpp')
lines2 = f.readlines()
f.close()

cpp_file = ''

for line in lines2:
    converted = line.replace("(BIAS_SIZ)",str(bias_siz))
    converted = converted.replace("(BIN_XOR_MAC)",bin_xor_mac)
    converted = converted.replace("(KSIZ)",str(ksiz))
    converted = converted.replace("(MAX_DENSE_SIZ)",str(max_dense_siz))
    converted = converted.replace("(OUT_DENSE_SIZ)",str(out_dense_siz))
    converted = converted.replace("(WEIGHT_SIZ)",str(weight_siz))
    converted = converted.replace("(MAX_BCONV_WIDTH)",str(max_bconv_width))
    converted = converted.replace("(NUM_LAYER)",str(num_layer))
    converted = converted.replace("(IMGSIZ)",str(imgsiz))
    converted = converted.replace("(NUMIMG)",str(numimg))

    converted = converted.replace("(BCONV_REG_PRAGMA)",bconv_reg_pragma)
    converted = converted.replace("(BCONV_REG_SELECT)",bconv_reg_select)
    converted = converted.replace("(BCONV_BIAS_SELECT)",bconv_bias_select)
    converted = converted.replace("(BCONV_WEIGHT_SELECT)",bconv_weight_select)
    converted = converted.replace("(DEF_CNN_PARAMETER)",def_cnn_parameter)
    converted = converted.replace("(DEF_CNN_LAYER)",def_cnn_layer)
    converted = converted.replace("(DEF_BIAS_MEM)",def_bias_mem)
    converted = converted.replace("(DEF_WEIGHT_MEM)",def_weight_mem)
    converted = converted.replace("(SET_BIAS_MEM)",set_bias_mem)
    converted = converted.replace("(SET_WEIGHT_MEM)",set_weight_mem)
    converted = converted.replace("(READ_BIAS_MEM)",read_bias_mem)
    converted = converted.replace("(READ_WEIGHT_MEM)",read_weight_mem)

    cpp_file += converted
    
cnn_file = args.config_path + "/sdsoc/main.cpp"
with open(cnn_file,'w') as f:
	f.write(cpp_file)

# generate C++ main code including a socket communication via an Ethernet ------------
f = open('template_cpp_r7_socket_main.cpp')
lines2 = f.readlines()
f.close()

cpp_file = ''

for line in lines2:
    converted = line.replace("(BIAS_SIZ)",str(bias_siz))
    converted = converted.replace("(BIN_XOR_MAC)",bin_xor_mac)
    converted = converted.replace("(KSIZ)",str(ksiz))
    converted = converted.replace("(MAX_DENSE_SIZ)",str(max_dense_siz))
    converted = converted.replace("(OUT_DENSE_SIZ)",str(out_dense_siz))
    converted = converted.replace("(WEIGHT_SIZ)",str(weight_siz))
    converted = converted.replace("(MAX_BCONV_WIDTH)",str(max_bconv_width))
    converted = converted.replace("(NUM_LAYER)",str(num_layer))
    converted = converted.replace("(IMGSIZ)",str(imgsiz))
    converted = converted.replace("(NUMIMG)",str(numimg))

    converted = converted.replace("(BCONV_REG_PRAGMA)",bconv_reg_pragma)
    converted = converted.replace("(BCONV_REG_SELECT)",bconv_reg_select)
    converted = converted.replace("(BCONV_BIAS_SELECT)",bconv_bias_select)
    converted = converted.replace("(BCONV_WEIGHT_SELECT)",bconv_weight_select)
    converted = converted.replace("(DEF_CNN_PARAMETER)",def_cnn_parameter)
    converted = converted.replace("(DEF_CNN_LAYER)",def_cnn_layer)
    converted = converted.replace("(DEF_BIAS_MEM)",def_bias_mem)
    converted = converted.replace("(DEF_WEIGHT_MEM)",def_weight_mem)
    converted = converted.replace("(SET_BIAS_MEM)",set_bias_mem)
    converted = converted.replace("(SET_WEIGHT_MEM)",set_weight_mem)
    converted = converted.replace("(READ_BIAS_MEM)",read_bias_mem)
    converted = converted.replace("(READ_WEIGHT_MEM)",read_weight_mem)

    cpp_file += converted

cnn_file = args.config_path + "/sdsoc/socket_main.cpp"
with open(cnn_file,'w') as f:
	f.write(cpp_file)

###########################################################################################
# END OF PROGRAM
###########################################################################################
