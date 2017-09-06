# GUINNESS: A GUI based binarized Neural NEtwork SyntheSizer toward an FPGA

This GUI based framework includes both a training on a GPU, and a bitstream generation for an FPGA using the Xilinx Inc. SDSoC. This tool uses the Chainer deep learning framework to train a binarized CNN. Also, it uses optimization techniques for an FPGA implementation. Details are shown in following papers:

[Nakahara IPDPSW2017] H. Yonekawa and H. Nakahara, "On-Chip Memory Based Binarized Convolutional Deep Neural Network Applying Batch Normalization Free Technique on an FPGA," IPDPS Workshops, 2017, pp. 98-105.  
[Nakahara FPL2017] H. Nakahara et al., "A Fully Connected Layer Elimination for a Binarized Convolutional Neural Network on an FPGA", FPL, 2017, (to appear).

### 1. Requirements:

Ubuntu 14.04 or 16.04  
Python 2.7.6+  
CUDA 8.0 (+GPU), not neessary to install a cuDNN library  
Chainer 1.23.0 or 1.24.0  

SDSoC 2016.4 (or 2017.1)  
FPGA board: Xilinx ZC702, ZCU102, Digilent Zedboard, Zybo  
(In the near future, I will support the PYNQ board)  

PyQt4, matplotlib, python-opencv2, numpy, scipy,   

### 2. Setup Libraries

 Install the following python libraries:

 Chainer 

 sudo pip install chainer==1.24.0
 
 PyQt4 (not PyQt5!)

 sudo apt-get install python-qt4 pyqt4-dev-tools

### 3. Run GUINNESS

 $ python guinness.py

### 4. Tutorial

 Read a following document.

 The GUINNESS introduction and BCNN implementation on an FPGA  
 guinness_tutorial1.pdf (located on the same folder)
 or download from <https://www.dropbox.com/s/fyskw81ua1mqtze/guinness_tutorial1.pdf?dl=0>

### 5. On-going works
 I'm developing extend versions of the binarized CNN applications.
 
 High performance image recognition (fully pipelined version)  

 Object detector on a low-cost FPGA  

### 6. Acknowledgements
 This work is based on following projects:

 Chainer binarized neural network by Daisuke Okanohara  
 https://github.com/hillbig/binary_net

 Various CNN models including Deep Residual Networks (ResNet)   
  for CIFAR10 with Chainer by mitmul  
 https://github.com/mitmul/chainer-cifar10

 This research is supported in part by the Grants in Aid for Scientistic Research of JSPS,  
and an Accelerated Innovation Research Initiative Turning Top Science and Ideas into High-Impact  
Values program(ACCEL) of JST. Also, thanks to the Xilinx University Program (XUP), Intel University Program,
 and the NVidia Corp.'s support.
