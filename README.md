# GUINNESS: A GUI based binarized Neural NEtwork SyntheSizer toward an FPGA (Trial version)

This GUI based framework includes both a training on a GPU, and a bitstream generation for an FPGA using the Xilinx Inc. SDSoC. This tool uses the Chainer deep learning framework to train a binarized CNN. Also, it uses optimization techniques for an FPGA implementation. Details are shown in following papers:

[Nakahara IPDPSW2017] H. Yonekawa and H. Nakahara, "On-Chip Memory Based Binarized Convolutional Deep Neural Network Applying Batch Normalization Free Technique on an FPGA," IPDPS Workshops, 2017, pp. 98-105.  

[Nakahara FPL2017] H. Nakahara et al., "A Fully Connected Layer Elimination for a Binarized Convolutional Neural Network on an FPGA", FPL, 2017, pp. 1-4.

[Nakahara FPL2017 Demo] H. Nakahara et al., "A demonstration of the GUINNESS: A GUI based neural NEtwork SyntheSizer for an FPGA", FPL, 2017, page 1.

### 1. Requirements:

Ubuntu 16.04 LTS (14.04 LTS is also supported)  

Python 3.5.1
(Note that, my recommendation is to install by Anaconda 4.1.0 (64bit)+Pyenv,
 for Japanese Only, I prepared the Python 3.5 by following http://blog.algolab.jp/post/2016/08/21/pyenv-anaconda-ubuntu/)

CUDA 8.0 (+GPU)
(Note that, CUDA 9.0 is also supported)

Chainer 1.24.0 + CuPy 2.0

Xilinx Inc. SDSoC 2017.2 (2016.4 is also supported for the low-end FPGAs only)

FPGA board: Xilinx ZC702, ZC706, ZCU102, Digilent Zedboard, Zybo  
(Soon, I will support Intel's FPGAs!, and the PYNQ board)  

PyQt4, matplotlib, OpenCV3, numpy, scipy,
(Above libraries are installed by the Anaconda, however, you must individually install the OpenCV by "conda install -y -c menpo opencv3")

### 2. Setup Libraries

 Install the following python libraries:

 Chainer 

 sudo pip install chainer==1.24.0
 
 PyQt4 (not PyQt5!), it is already installed by the Anaconda

 sudo apt-get install python-qt4 pyqt4-dev-tools

 OpenCV3
 
 conda install -y -c menpo opencv3

### 3. Run GUINNESS

 $ python guinness.py

### 4. Tutorial

 Read a following document (25/Oct./2017 Updated!!)

 1 The GUINNESS introduction and BCNN implementation on an FPGA  
 guinness_tutorial1_v2.pdf <https://www.dropbox.com/s/oe6gptgyi4y92el/guinness_tutorial1_v2.pdf?dl=0>

 2 The GUINNESS for the Intel FPGAs (Soon, will be uploaded)
 
 3 Pedestrian detection (Under preparing)

 4 Make a custom IP core for your own FPGA board (Under preparing) 

### 5. On-going works
 This is a just trial version. I have already developed the extend version including following ones.
 
 Supporing the Intel's FPGA (DE5-net, DE10-nano, and DE5a-net boards with the Intel SDK for OpenCL)
 
 High performance image recognition (fully pipelined and SIMD CNNs)  
 
 Object detector on a low-cost FPGA (e.g., pedestrian detection)

 If you are interesting the extended one, please, contact me.

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
