# -----------------------------------------------------------------------
# guinness.py
# A GUI based Neural NEtwork SyntheSizer for an FPGA deep learning
#
# Creation Date   : 04/Aug./2017
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the GPL v2.0 License.
# 
# Acknowledgements:
# This source code is based on following projects:
#
# Chainer binarized neural network by Daisuke Okanohara
# https://github.com/hillbig/binary_net
# Various CNN models including Deep Residual Networks (ResNet) 
#  for CIFAR10 with Chainer by mitmul
# https://github.com/mitmul/chainer-cifar10
# -----------------------------------------------------------------------

import sys,random,time,os
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from subprocess import check_call
import pickle
import subprocess
#import seaborn as sns # this is optional...
import shutil

#global variables
n_dim = 3 # the number of dimensions for the first layer (BGR format)
img_siz = 32 # default input image size
n_class = 10 # default the number of classes to be inferenced
is_load_pretrain = 0

class Layout(QtGui.QWidget):
    def __init__(self):
        super(Layout,self).__init__()

        global is_load_pretrain
        global n_dim # BGR color image
        global img_siz # 32x32 image
        global n_class # #classes
        
        is_load_pretrain = 0
        n_dim = 3
        img_siz = 32
        n_class = 10

        self.setMyself()
        self.set_project_name()
        self.show()

    def setMyself(self):
        self.setGeometry(50,50,1100,600)
        self.setWindowTitle("GUINNESS: A GUI based Neural NEtwork SyntheSizer")

    def set_project_name(self):
        ##################################################################
        # Left Column
        ##################################################################
        vbox_left_column = QtGui.QVBoxLayout()
#        vbox_left_column.setGeometry(QtCore.QRect(0,0,800,24))

        # project setup --------------------------------------------------
        project_setup_box = QtGui.QGroupBox("1. Project Setup")
        project = QtGui.QLabel('Project Name')
        self.projectEdit = QtGui.QLineEdit()
        self.projectEdit.setText('Project1')

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(project)
        hbox.addWidget(self.projectEdit)

        vbox_proj = QtGui.QVBoxLayout()
        vbox_proj.addLayout(hbox)

        ProjSaveButton = QtGui.QPushButton("SAVE")
        self.connect(ProjSaveButton,QtCore.SIGNAL('clicked()'),self.SaveProj)
        ProjLoadButton = QtGui.QPushButton("LOAD")
        self.connect(ProjLoadButton,QtCore.SIGNAL('clicked()'),self.LoadProj)
        hbox_proj = QtGui.QHBoxLayout()
        hbox_proj.addWidget(ProjSaveButton)
        hbox_proj.addWidget(ProjLoadButton)
        vbox_proj.addLayout(hbox_proj)

        project_setup_box.setLayout(vbox_proj)
        vbox_left_column.addWidget(project_setup_box)

        # cnn setup table ------------------------------------------------
        cnn_setup_box = QtGui.QGroupBox("2. CNN Specificaion")

        vbox_cnn = QtGui.QVBoxLayout()

        cnntype = QtGui.QLabel('Type')
        self.combo1 = QtGui.QComboBox()
        self.combo1.addItem("LeNet5")
        self.combo1.addItem("TinyCNN")
        self.combo1.addItem("VGG9ave")
        self.combo1.addItem("VGG11ave")
        self.combo1.addItem("VGG16ave")
        self.combo1.addItem("VGG19ave")
        LoadButton = QtGui.QPushButton("LOAD CONFIG")
        self.connect(LoadButton,QtCore.SIGNAL('clicked()'),self.LoadConfig)
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(cnntype)
        hbox2.addWidget(self.combo1)
        hbox2.addWidget(LoadButton)

        vbox_cnn.addLayout(hbox2)

        self.table = QtGui.QTableWidget()
        self.table.setColumnCount(5)

        labels = ["Type","In #Fmaps","Out #Fmaps","In Fsiz","Train?"]
        self.table.setHorizontalHeaderLabels(labels);
        self.table.setColumnWidth(0, 90);
        self.table.setColumnWidth(1, 80);
        self.table.setColumnWidth(2, 80);
        self.table.setColumnWidth(3, 50);
        self.table.setColumnWidth(4, 50);

        self.LoadConfig()

        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.contextMenu_)

        vbox_cnn.addWidget(self.table)
        cnn_setup_box.setLayout(vbox_cnn)
        vbox_left_column.addWidget(cnn_setup_box)

        ##################################################################
        # Right Column
        ##################################################################
        vbox_right_column = QtGui.QVBoxLayout()

        training_setup_box = QtGui.QGroupBox("3. Training")
        vbox_training = QtGui.QVBoxLayout()
        # parameters for traning -----------------------------------------
        # training data
        tdlabel = QtGui.QLabel('Training Data')
        ld_button = QtGui.QPushButton("Load")
        ld_button.clicked.connect(self.open_FileDialog)
        self.td_label = QtGui.QLineEdit("image.pkl")

        hbox_td = QtGui.QHBoxLayout()
        hbox_td.addWidget(tdlabel) 
        hbox_td.addWidget(ld_button) 
        hbox_td.addWidget(self.td_label)
        vbox_training.addLayout(hbox_td)

        # training label
        tllabel = QtGui.QLabel('Training Label')
        ll_button = QtGui.QPushButton("Load")
        ll_button.clicked.connect(self.open_FileDialog_tl)
        self.tl_label = QtGui.QLineEdit("label.pkl") 

        hbox_tl = QtGui.QHBoxLayout()
        hbox_tl.addWidget(tllabel) 
        hbox_tl.addWidget(ll_button) 
        hbox_tl.addWidget(self.tl_label)
        vbox_training.addLayout(hbox_tl)

        # # of training
        n_trains = QtGui.QLabel('Number of traning')
        self.n_trains_Edit = QtGui.QLineEdit()
        self.n_trains_Edit.setText("10")

        hbox_ntrain = QtGui.QHBoxLayout()
        hbox_ntrain.addWidget(n_trains)
        hbox_ntrain.addWidget(self.n_trains_Edit)

        vbox_training.addLayout(hbox_ntrain)

        # optimizer
        hbox3 = QtGui.QHBoxLayout()
        cnntype = QtGui.QLabel('Optimizer')        
        self.b11=QtGui.QRadioButton("SGD")
        self.b11.setChecked(True)
        self.b12=QtGui.QRadioButton("Adam")
        bg1=QtGui.QButtonGroup()
        bg1.addButton(self.b11)
        bg1.addButton(self.b12)
        hbox3.addWidget(cnntype)
        hbox3.addWidget(self.b11)
        hbox3.addWidget(self.b12)

        vbox_training.addLayout(hbox3)

        # Use GPU?
        self.cb = QtGui.QCheckBox('Use GPU')
        self.cb.setChecked(True)
        vbox_training.addWidget(self.cb)

        # message
        train_process = QtGui.QLabel('Training Process View')
        vbox_training.addWidget(train_process)

        # matplotlib
        self.canvas = Canvas()

        self.canvas.refresh(int(self.n_trains_Edit.text()))

        vbox_training.addWidget(self.canvas)
        
        # training button
        hbox_control = QtGui.QHBoxLayout()
        self.bstart=QtGui.QPushButton("Start Training")
        bg1.addButton(self.bstart)
        self.bstart.clicked.connect(self.start_training)
        bstop=QtGui.QPushButton("Stop Training")
        bstop.setVisible(False)
        bg1.addButton(bstop)
        hbox_control.addWidget(self.bstart)
        hbox_control.addWidget(bstop)

        vbox_training.addLayout(hbox_control)
        training_setup_box.setLayout(vbox_training)
        vbox_right_column.addWidget(training_setup_box)

        # FPGA implementation ------------------------------------------------
        # Select fpga board
        fpga_setup_box = QtGui.QGroupBox("4. C/C++ Code Generation for FPGA Implementation")
        vbox_fpga = QtGui.QVBoxLayout()

        fpgaboard = QtGui.QLabel('Target FPGA Board')
        self.combo2 = QtGui.QComboBox()
        self.combo2.addItem("zed")
        self.combo2.addItem("zybo")
        self.combo2.addItem("vc702")
        self.combo2.addItem("zcu102")
        hbox3 = QtGui.QHBoxLayout()
        hbox3.addWidget(fpgaboard)
        hbox3.addWidget(self.combo2)

        vbox_fpga.addLayout(hbox3)

#        # Setup Clock Frequency
#        clkfreq = QtGui.QLabel('Clock Frequency (MHz)')
#        combo3 = QtGui.QComboBox()
#        combo3.addItem("100.0")
#        combo3.addItem("147.6")
#        combo3.addItem("150.0")
#        combo3.addItem("200.0")
#        hbox4 = QtGui.QHBoxLayout()
#        hbox4.addWidget(clkfreq)
#        hbox4.addWidget(combo3)
#
#        vbox_fpga.addLayout(hbox4)

        # Run Bitstream Generation
#        bstart_bitgen=QtGui.QPushButton("Generate Bitstream")
        bstart_bitgen=QtGui.QPushButton("Generate C/C++ Code")
        bg1.addButton(bstart_bitgen)
        bstart_bitgen.clicked.connect(self.start_bitgen)

        vbox_fpga.addWidget(bstart_bitgen)

        fpga_setup_box.setLayout(vbox_fpga)
        vbox_right_column.addWidget(fpga_setup_box)

        # -------------------------------------------------------
        # overall layout
        # -------------------------------------------------------
        hbox_global = QtGui.QHBoxLayout()
        hbox_global.addLayout(vbox_left_column)
        hbox_global.addLayout(vbox_right_column)

        self.setLayout(hbox_global)

    # -----------------------------------------------------------
    # Context Menu for the CNN configuration table
    # -----------------------------------------------------------
    def contextMenu_(self, event):
        menu = QtGui.QMenu()
        addAction = menu.addAction('Add layer',)
        delAction = menu.addAction('Delete layer',)

        action = menu.exec_(QtGui.QCursor.pos())

        initial_options = []
        n_in_fmaps = []
        n_ou_fmaps = []
        infmap_siz = []

        for i in range(self.table.rowCount()):
            itm1 = self.table.cellWidget(i,0)
            itm2 = self.table.item(i,1)
            itm3 = self.table.item(i,2) 
            itm4 = self.table.item(i,3) 
            val1 = itm1.currentIndex()
            val2 = str(itm2.text())
            val3 = str(itm3.text())
            val4 = str(itm4.text())

            initial_options.append(val1)
            n_in_fmaps.append(val2)
            n_ou_fmaps.append(val3)
            infmap_siz.append(val4)

        if action == addAction:
            initial_options.insert(self.table.currentRow(),1)
            n_in_fmaps.insert(self.table.currentRow(),'0')
            n_ou_fmaps.insert(self.table.currentRow(),'0')
            infmap_siz.insert(self.table.currentRow(),'0')

        elif action == delAction:
            initial_options.pop(self.table.currentRow())
            n_in_fmaps.pop(self.table.currentRow())
            n_ou_fmaps.pop(self.table.currentRow())
            infmap_siz.pop(self.table.currentRow())

        self.table.setRowCount(len(initial_options))
        for index in range(len(initial_options)):
            combo = QtGui.QComboBox()
            for t in self.combo_box_options:
                combo.addItem(t)
            combo.setCurrentIndex(initial_options[index])
            self.table.setCellWidget(index,0,combo)
            item1 = QtGui.QTableWidgetItem(n_in_fmaps[index])
            self.table.setItem(index,1,item1)
            item2 = QtGui.QTableWidgetItem(n_ou_fmaps[index])
            self.table.setItem(index,2,item2)
            item3 = QtGui.QTableWidgetItem(infmap_siz[index])
            self.table.setItem(index,3,item3)

            item4 = QtGui.QCheckBox('')
            item4.setChecked(True) # isChecked() == True?False?
            self.table.setCellWidget(index,4,item4)

    # -----------------------------------------------------------------------
    # Performe Training
    #  First, generate customized net.py
    #   then, call external trainer.py
    #  During training, the GUI plots traning process
    # -----------------------------------------------------------------------
    def start_training(self):
        # remove temporary logfile, if new traning start
        global is_load_pretrain
        if is_load_pretrain == 0 and os.path.exists("./temp_log.csv") == True:
            print("CLEARN UP LOGFILE")
#            os.remove("temp_log.csv")

        # generate CNN python code (this version only supports chainer 1.21-24.0)
        print("[INFO] GENERATE PYTHON CODE FOR CNN")
        f = open('header.txt')
        pcode = f.read()
        pcode += '\n'
        f.close()

        conv_idx = 0
        bn_idx = 0
        dense_idx = 0
        for i in range(self.table.rowCount()):
            itm1 = self.table.cellWidget(i,0)
            itm2 = self.table.item(i,1)
            itm3 = self.table.item(i,2) 
            itm4 = self.table.item(i,3) 
            val1 = int(itm2.text())
            val2 = int(itm3.text())
            val3 = int(itm4.text())
            
            if itm1.currentText() == 'Conv(Int)':
                pcode += '            conv%d=IC.Convolution2D(%d,%d,3, stride=1, pad=1, nobias=True),\n' % (conv_idx,val1,val2)
                pcode += '            b%d=L.BatchNormalization(%d)' % (bn_idx,val2)
                conv_idx += 1
                bn_idx += 1
            elif itm1.currentText() == 'Conv(Bin)':
                pcode += '            conv%d=BC.Convolution2D(%d,%d,3, stride=1, pad=1, nobias=True),\n' % (conv_idx,val1,val2)
                pcode += '            b%d=L.BatchNormalization(%d)' % (bn_idx,val2)
                conv_idx += 1
                bn_idx += 1
            elif itm1.currentText() == 'Max Pool':
                pass
            elif itm1.currentText() == 'Ave Pool':
                pass
            else: # Dense
                pcode += '            fc%d=BL.BinaryLinear(%d,%d),\n' % (dense_idx,val1,val2)
                pcode += '            b%d=L.BatchNormalization(%d)' % (bn_idx,val2)
                dense_idx += 1
                bn_idx += 1
            
            if i == self.table.rowCount() - 1:
                pcode += '\n        )\n'
            else:
                if itm1.currentText() == 'Max Pool' or itm1.currentText() == 'Ave Pool':
                    pass
                else:
                    pcode += ',\n'


        pcode += '\n    def __call__(self, x, train):\n'
        conv_idx = 0
        bn_idx = 0
        dense_idx = 0
        for i in range(self.table.rowCount()):
            itm1 = self.table.cellWidget(i,0)
            itm2 = self.table.item(i,1)
            itm3 = self.table.item(i,2) 
            itm4 = self.table.item(i,3) 
            val1 = int(itm2.text())
            val2 = int(itm3.text())
            val3 = int(itm4.text())
            
            if itm1.currentText() == 'Conv(Int)':
                pcode += '        h = bst.bst(self.b%d(self.conv%d(x)))\n' % (bn_idx,conv_idx)
                bn_idx += 1
                conv_idx += 1
            elif itm1.currentText() == 'Conv(Bin)':
                pcode += '        h = bst.bst(self.b%d(self.conv%d(h)))\n' % (bn_idx,conv_idx)
                bn_idx += 1
                conv_idx += 1
            elif itm1.currentText() == 'Max Pool':
                pcode += '        h = F.max_pooling_2d(h, 2)\n'
            elif itm1.currentText() == 'Ave Pool':
                pcode += '        h = F.average_pooling_2d(h, %d)\n' % val3
            else: # Dense
                if i < self.table.rowCount() - 1:
                    if i == 0:
                        pcode += '        h = bst.bst(self.b%d(self.fc%d(x)))\n' % (bn_idx,dense_idx)
                    else:
                        pcode += '        h = bst.bst(self.b%d(self.fc%d(h)))\n' % (bn_idx,dense_idx)
                else:
                    pcode += '        h = self.b%d(self.fc%d(h))\n' % (bn_idx,dense_idx)
                bn_idx += 1
                dense_idx += 1

        pcode += '        return h'

        # code generation ----------------------------------------------------
        f = open('net2.py', 'w')
        f.write(pcode)
        f.close()

        # for test CNN by Python code (eval.py)
        net3_file = ''

        net3_file = pcode.replace("=L.","=LBN.")
        net3_file = net3_file.replace("./","../")

        # generate project directory if it not exist
        project_dir = "./" + self.projectEdit.text()
        if os.path.exists(project_dir) == False:
            os.mkdir(project_dir)

        # save Python simulation codes
        fname = "./" + self.projectEdit.text() + '/net3.py'
        print("[INFO] Python evaluation codes are seved to %s" % fname)
        with open(fname,'w') as f:
            f.write(net3_file)

        fname = "./" + self.projectEdit.text() + '/eval.py'
        print("[INFO] COPY evaluation code")
        shutil.copyfile('eval.py',fname)

        # setup training -----------------------------------------------------
        n_iter = int(self.n_trains_Edit.text())

        train_dataset = self.td_label.text()
        label_dataset = self.tl_label.text()
        if self.b11.isChecked() == True:
            optimizer_alg = "sgd"
        else:
            optimizer_alg = "adam"

        project_name = "temp"

        project_dir = "./" + self.projectEdit.text()
        if os.path.exists(project_dir) == False:
            os.mkdir(project_dir)

        # start training -----------------------------------------------------
        if self.cb.isChecked() == True:
            print("[INFO] START TRAINING: GPU MODE")
            gpu = "0"
        else:
            print("[INFO] START TRAINING: CPU MODE")
            gpu = "-1"

        if is_load_pretrain == 1:
            print("[INFO RESUME TRANINING]")
            resume = "yes"
            
            # copy pre-trained model,log files
            if os.path.isfile('./temp.model') == True:
                os.remove('./temp.model')
            model_file = "./" + self.projectEdit.text() + '/temp.model'
            if os.path.isfile(model_file) == True:
                print("[INFO] RESUME PRE-TRAINED MODEL FILE %s" % model_file)
                shutil.copyfile(model_file,'./temp.model')
            else:
                print("[ERROR] model file %s not found" % model_file)
                exit()

            if os.path.isfile('./temp_log.csv') == True:
                os.remove('./temp_log.csv')
            log_file = "./" + self.projectEdit.text() + '/temp_log.csv'
            if os.path.isfile(log_file) == True:
                print("[INFO] RESUME PRE-TRAINED LOG FILE %s" % log_file)
                shutil.copyfile(log_file,'./temp_log.csv')
            else:
                print("[ERROR] log file %s not found" % log_file)
                exit()
            
        else:
            resume = "no"

        # Peform training
        global n_dim
        global img_siz
        
        subprocess.Popen(["python","train.py","-g",gpu,"--iter",str(n_iter),"--dim",str(n_dim),"--siz",str(img_siz),"--dataset",train_dataset,"--label",label_dataset,"--optimizer",optimizer_alg,"--prefix",project_name,"--lr_decay_iter","100","--resume",resume]) # background job = python train.py &

        # set process file
        with open("train_status.txt","w") as f:
            f.write("run")

        # eliminate training start button
        self.bstart.setVisible(False)

        # Start training check process
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateCanvas)
        self.timer.start(1000)
    
    # -----------------------------------------------------------------------
    # Update Canvas for training process view
    # -----------------------------------------------------------------------
    def updateCanvas(self):
        global is_load_pretrain
        log_file = "temp_log.csv"

        if( os.path.exists(log_file) == True):
            check = 0
            n_lines_in_logfile = 0
            with open(log_file,'r') as f:
                n_lines_in_logfile = len(f.readlines())
                if n_lines_in_logfile > 2:
                    check = 1

            if check == 1:
                train_loss,train_acc,test_loss,test_acc = np.loadtxt(log_file, delimiter=',', skiprows=1,usecols=(1,2,5,6),unpack=True)
                self.canvas.push_data(train_acc,test_acc,train_loss,test_loss)
                self.canvas.refresh(n_lines_in_logfile - 1)

        with open("train_status.txt", "r") as f:
            status = f.read()

            if status != 'run':
                print("[INFO] FINISH TRAINING")
                project_path = "./" + self.projectEdit.text()
                subprocess.Popen(["cp","temp.model",project_path]) # background job = python train.py &
                subprocess.Popen(["cp","temp_log.csv",project_path]) # background job = python train.py &
                self.timer.stop()
                ret = QtGui.QMessageBox.information(None, "Training Status", "Training Finished")

                # set continue training mode
                self.bstart.setVisible(True)
                self.bstart.setText('Continue Training')
                is_load_pretrain = 1

    # -----------------------------------------------------------------------
    # Save CNN Configuration File
    # -----------------------------------------------------------------------
    def save_configfile(self):
        # generate configuration file
        print("------------- GENERATE CONFIGURATION FILE --------------")
        print("TARGET DEVICE: %s" % self.combo2.currentText())
        print("[INFO] Generate Configuration File")

        config = {}
        initial_options = []
        n_in_fmaps = []
        n_ou_fmaps = []
        infmap_siz = []
        max_dense_siz = 0
        max_bconv_width = 0
        bias_siz = 0
        weight_siz = 0

        global img_siz
        global n_class

        for i in range(self.table.rowCount()):
            itm1 = self.table.cellWidget(i,0)
            itm2 = self.table.item(i,1)
            itm3 = self.table.item(i,2) 
            itm4 = self.table.item(i,3) 
            val1 = str(itm2.text())
            val2 = str(itm3.text())
            val3 = str(itm4.text())

            if itm1.currentIndex() == 4:
                if max_dense_siz < int(val1):
                    max_dense_siz = int(val1)

            if itm1.currentIndex() == 0 or itm1.currentIndex() == 1 or itm1.currentIndex() == 4:
                bias_siz += int(val2)

            if itm1.currentIndex() == 1:
                if max_bconv_width < int(val2):
                    max_bconv_width = int(val2)

            if itm1.currentIndex() == 0 or itm1.currentIndex() == 1:
                weight_siz += (int(val1) * int(val2) * 3 * 3)

            if itm1.currentIndex() == 4:
                weight_siz += (int(val1) * int(val2))

            initial_options.append(itm1.currentIndex())
            n_in_fmaps.append(val1)
            n_ou_fmaps.append(val2)
            infmap_siz.append(val3)

        config['initial_options'] = initial_options
        config['n_in_fmaps'] = n_in_fmaps
        config['n_ou_fmaps'] = n_ou_fmaps
        config['infmap_siz'] = infmap_siz

        config['ksiz'] = 3
        config['imgsiz'] = infmap_siz[0]
        config['max_dense_siz'] = max_dense_siz
        config['out_dense_siz'] = n_ou_fmaps[len(initial_options) - 1]
        config['bias_siz'] = bias_siz
        config['weight_siz'] = weight_siz
        config['max_bconv_width'] = max_bconv_width
        config['num_layer'] = len(initial_options)

        config_file = "./" + self.projectEdit.text() + "/config.pickle"
        with open(config_file, mode='wb') as f:
            pickle.dump(config, f)
       
    # -----------------------------------------------------------------------
    # Generate Bitstream
    # -----------------------------------------------------------------------
    def start_bitgen(self):
        # generate configuration file
        print("------------- GENERATE CONFIGURATION FILE --------------")
        print("TARGET DEVICE: %s" % self.combo2.currentText())
        print("[INFO] Generate Configuration File")

        # save configuration file
        self.save_configfile()

        # generate SDSoC directory
        sdsoc_dir = "./" + self.projectEdit.text() + "/sdsoc"
        if os.path.exists(sdsoc_dir) == False:
            os.mkdir(sdsoc_dir)

        # Call C++ code generator for the SDSoC
        print("[INFO] GENERATE C++ CODE")
        config_path = "./" + self.projectEdit.text()

        subprocess.Popen(["python","gen_cpp_code_v3.py","--config_path",config_path]) # background job = python train.py &

        # generate makefile using template files
        print("[INFO] GENERATE Makefile for the SDSoC")
        f = open('template_Makefile')
        lines2 = f.readlines()
        f.close()

        makefile_txt = ''

        for line in lines2:
            tmp = line.replace("(CNN_C_SOURCE)","cnn.cpp")
            tmp = tmp.replace("(ELF_FILE_PATH)",self.projectEdit.text() + ".elf")
            tmp = tmp.replace("(TARGET_BOARD)",self.combo2.currentText())

            makefile_txt += tmp

        makefile_name = "./" + self.projectEdit.text() + "/sdsoc/Makefile"
        with open(makefile_name,'w') as f:
            f.write(makefile_txt)

        # generate sdsoc/sd_card directory
        print("[INFO] MAKE A DIRECTROY: ./%s/sdsoc/to_sd_card" % self.projectEdit.text())
        sd_card_dir = "./" + self.projectEdit.text() + "/sdsoc/to_sd_card"
        if os.path.exists(sd_card_dir) == False:
            os.mkdir(sd_card_dir)

        # generate HLS directory
        print("[INFO] MAKE A DIRECTROY: ./%s/HLS" % self.projectEdit.text())
        HLS_dir = "./" + self.projectEdit.text() + "/HLS"
        if os.path.exists(HLS_dir) == False:
            os.mkdir(HLS_dir)

        # convert trained *.model to weight text file
        print("[INFO] CONVERT TRAINED WEIGHTS INTO TEXT FILE")
        config_path = "./" + self.projectEdit.text()
        proc = subprocess.Popen(["python","conv_npz2txt_v2.py","--config_path",config_path]) # background job = python train.py &
        proc.wait()

        print(" ... [FINISH]")

        # copy benchmark file from trainer, if it exist
        print("[INFO] COPY BENCHMARK IMAGE FILE")
        image_file = "./test_img.txt"
        if os.path.isfile(image_file) == True:
            sd_card_dir = "./" + self.projectEdit.text() + "/sdsoc/to_sd_card"
            subprocess.Popen(["cp",image_file,sd_card_dir])
            print(" ... [FINISH]")
        else:
            print("FAILURE")

        # performe system generation, call SDSoC by make command
        # (subprocess!!!)
#        print("[INFO] GENERATE BITSTREAM, WAIT TENS MINUTES...")
        print("[INFO] SUCCESSFULLY C/C++ CODE GENERATION")
        print("[INFO] PLEASE, ``SAVE'' YOUR CURRENT DESIGN")

#        # show message
#        ret = QtGui.QMessageBox.information(None, "Bistream Generation Status", "C++ code generated")

    # -----------------------------------------------------------------------
    # FileOpen Dialog for Project Configuration
    # -----------------------------------------------------------------------
    # save configuration file
    def SaveProj(self):
        config = ''
        config += 'PROJECT_NAME: %s\n' % self.projectEdit.text()
        config += 'TRAINING_DATA: %s\n' % self.td_label.text()
        config += 'TRAINING_LABEL: %s\n' % self.tl_label.text()
        config += 'NUM_OF_EPOCS: %d\n' % int(self.n_trains_Edit.text())
        if self.b11.isChecked() == True:
            config += 'OPTIMIZER: SGD\n'
        else:
            config += 'OPTIMIZER: Adam\n'
        if self.cb.isChecked() == True:
            config += 'USE_GPU: YES\n'
        else:
            config += 'USE_GPU: NO\n'
        config += 'FPGA_BOARD: %s\n' % self.combo2.currentText()

        config_file = "./" + self.projectEdit.text() + "/" + self.projectEdit.text() + ".proj"
        config_dir = "./" + self.projectEdit.text()
        if os.path.exists(config_dir) == False:
            os.mkdir(config_dir)

        with open(config_file, mode='w') as f:
            f.write(config)

        self.save_configfile()

    # load project configuration file
    def LoadProj(self):
        global is_load_pretrain
        filename = QtGui.QFileDialog.getOpenFileName(self, 'File Open', './')

        with open(filename, mode='r') as f:
            lines2 = f.readlines()
        
            for line in lines2:
                key, val = line.split()
                
                if key == 'PROJECT_NAME:':
                    self.projectEdit.setText(val)
                elif key == 'TRAINING_DATA:':
                    self.td_label.setText(val)
                elif key == 'TRAINING_LABEL:':
                    self.tl_label.setText(val)
                elif key == 'NUM_OF_EPOCS:':
                    self.n_trains_Edit.setText(val)
                elif key == 'OPTIMIZER:':
                    if val == 'SGD':
                        self.b11.setChecked(True)
                        self.b12.setChecked(False)
                    else:
                        self.b11.setChecked(False)
                        self.b12.setChecked(True)
                elif key == 'USE_GPU:':
                    if val == 'YES':
                        self.cb.setChecked(True)
                    else:
                        self.cb.setChecked(False)
                elif key == 'FPGA_BOARD:':
                    if val == 'zed':
                        idx = 0
                    elif val == 'zybo':
                        idx = 1
                    elif val == 'vc702':
                        idx = 2
                    else: # zcu102
                        idx = 3
                    self.combo2.setCurrentIndex(idx)
                else:
                    pass        

        # Restore CNN Configuration Table
        config_file = "./" + self.projectEdit.text() + "/config.pickle"
        with open(config_file, mode='rb') as f:
            config = pickle.load(f)

        initial_options = config['initial_options']
        n_in_fmaps = config['n_in_fmaps']
        n_ou_fmaps = config['n_ou_fmaps']
        infmap_siz = config['infmap_siz']

        self.table.setRowCount(len(initial_options))
        for index in range(len(initial_options)):
            combo = QtGui.QComboBox()
            for t in self.combo_box_options:
                combo.addItem(t)
            combo.setCurrentIndex(initial_options[index])
            self.table.setCellWidget(index,0,combo)
            item1 = QtGui.QTableWidgetItem(n_in_fmaps[index])
            self.table.setItem(index,1,item1)
            item2 = QtGui.QTableWidgetItem(n_ou_fmaps[index])
            self.table.setItem(index,2,item2)
            item3 = QtGui.QTableWidgetItem(infmap_siz[index])
            self.table.setItem(index,3,item3)

            item4 = QtGui.QCheckBox('')
            item4.setChecked(True) # isChecked() == True?False?
            self.table.setCellWidget(index,4,item4)


        # Restore Training Status Graph
        log_file = "temp_log.csv"
        log_path = "./" + self.projectEdit.text() + "/" + log_file

        if( os.path.exists(log_path) == True):
            print("log_file %s" % log_path)

            subprocess.call(["cp",log_path,"./"])

            train_loss,train_acc,test_loss,test_acc = np.loadtxt(log_file, delimiter=',', skiprows=1,usecols=(1,2,5,6),unpack=True)
            self.canvas.push_data(train_acc,test_acc,train_loss,test_loss)
            self.canvas.refresh(int(self.n_trains_Edit.text()))

            subprocess.call(["rm","-rf",log_file])

        is_load_pretrain = 1
        self.bstart.setText('Continue Training')

        # Restore Global Variables
        global img_siz 
        img_siz = int(config['imgsiz'])
        global n_class 
        n_class = int(n_ou_fmaps[len(initial_options) - 1])
        
        print("[INFO] IMAGE SIZE %dx%d" % (img_siz,img_siz))
        print("[INFO] #CLASSES: %d" % (n_class))

        # update widgets
        self.update()

    # -----------------------------------------------------------------------
    # Set Feature Map Size
    # -----------------------------------------------------------------------
    def SetSize(self):
        global img_siz

        fsiz = 0
        for index in range(self.table.rowCount()):
            itm0 = self.table.cellWidget(index,0)
            itm3 = self.table.item(index,3) 
            
            if index == 0:
                fsiz = img_siz
                tbl_item = QtGui.QTableWidgetItem(str(fsiz))
                self.table.setItem(index,3,tbl_item)
                #fsiz = int(itm3.text())
            elif itm0.currentText() == 'Conv(Int)':
                tbl_item = QtGui.QTableWidgetItem(str(fsiz))
                self.table.setItem(index,3,tbl_item)

            elif itm0.currentText() == 'Conv(Bin)':
                tbl_item = QtGui.QTableWidgetItem(str(fsiz))
                self.table.setItem(index,3,tbl_item)

            elif itm0.currentText() == 'Max Pool':
                tbl_item = QtGui.QTableWidgetItem(str(fsiz))
                self.table.setItem(index,3,tbl_item)

                fsiz = fsiz / 2
                if fsiz < 1:
                    fsiz = 1

            elif itm0.currentText() == 'Ave Pool':
                tbl_item = QtGui.QTableWidgetItem(str(fsiz))
                self.table.setItem(index,3,tbl_item)

                fsiz = fsiz / 2
                if fsiz < 1:
                    fsiz = 1

            else: # Dense
                tbl_item = QtGui.QTableWidgetItem('1')
                self.table.setItem(index,3,tbl_item)


    # -----------------------------------------------------------------------
    # FileOpen Dialog for Training data selection
    # -----------------------------------------------------------------------
    def open_FileDialog(self):
        global n_dim
        global img_siz
        filename = QtGui.QFileDialog.getOpenFileName(self, 'File Open', './')
        self.td_label.setText(filename)

        # check dimension and size
        with open(filename, 'rb') as f:
            images = pickle.load(f)        
        
            print("[INFO] IMAGE SIZE %dx%d" % (images['train'].shape[3],images['train'].shape[3]))

            n_dim = images['train'].shape[1]
            img_siz = images['train'].shape[2]

            self.SetSize()

    def open_FileDialog_tl(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'File Open', './')
        self.tl_label.setText(filename)

        # check dimension and size
        with open(filename, 'rb') as f:
            global n_class
            labels = pickle.load(f)        
            label_set = labels['train'].astype(np.int8)
            max_idx = np.max(label_set) + 1 # includes '0' label
            print("[INFO] #CLASSES: %d" % max_idx)

            n_class = max_idx

            item3 = QtGui.QTableWidgetItem(str(n_class))
            self.table.setItem(self.table.rowCount()-1,2,item3)

    # -----------------------------------------------------------------------
    # Load PreDefined CNN
    # -----------------------------------------------------------------------
    def LoadConfig(self):
        template_name = self.combo1.currentText()

        self.combo_box_options = ["Conv(Int)","Conv(Bin)","Max Pool","Ave Pool","Dense"]
        if template_name == 'LeNet5':
            initial_options = [0,1,1,3,4]
            n_in_fmaps = [ '1','64','64','64','64']
            n_ou_fmaps = ['64','64','64','64','10']
            infmap_siz = ['28','28','28','28','1']
        elif template_name == 'TinyCNN':
            initial_options   = [0,1,1,2,3,4]
            n_in_fmaps = [ '3', '64','128','128','128','128']
            n_ou_fmaps = ['64','128','128','128','128', '10']
            infmap_siz = ['32', '32', '32', '32', '16',  '1']
        elif template_name == 'VGG9ave':
            initial_options = [0, 1,   2,    1,   1,   2,   1,   1,   2,   1,   1,   3,   4]
            n_in_fmaps = [ '3','64','64', '64','64','64','64','64','64','64','64','64','64']
            n_ou_fmaps = ['64','64','64', '64','64','64','64','64','64','64','64','64','10']
            infmap_siz = ['32','32','32', '16','16','16', '8', '8', '8', '4', '4', '4', '1']
        elif template_name == 'VGG11ave':
            initial_options = [0, 1,   2,    1,   1,   2,   1,   1,   2,   1,   1,   2,   1,   1,   3,   4]
            n_in_fmaps = [ '3','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','64']
            n_ou_fmaps = ['64','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','10']
            infmap_siz = ['32','32','32', '16','16','16', '8', '8', '8', '4', '4', '4', '2', '2', '2', '1']
        elif template_name == 'VGG16ave':
            initial_options = [0, 1,   2,    1,   1,   2,   1,   1,   1,   2,   1,   1,   1,   2,   1,   1,   1,   3,   4]
            n_in_fmaps = [ '3','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64']
            n_ou_fmaps = ['64','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','10']
            infmap_siz = ['64','64','64', '32','32','32','16','16','16','16', '8', '8', '8', '8', '4', '4', '4', '4', '1']
        elif template_name == 'VGG19ave':
            initial_options = [0, 1,   2,    1,   1,   2,   1,   1,   1,   1,   2,   1,   1,   1,   1,   2,   1,   1,   1,   1,   3,   4]
            n_in_fmaps = [ '3','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64']
            n_ou_fmaps = ['64','64','64', '64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','64','10']
            infmap_siz = ['64','64','64', '32','32','32','16','16','16','16','16', '8', '8', '8', '8', '8', '4', '4', '4', '4', '4', '1']
        else: # VGG11
            initial_options   = [0,1,2,1,1,2,1,1,2,1,1,2,4,4,4]
            n_in_fmaps = [ '3','64','64', '64','128','128','128','256','256','256','256','256','4096','1024','1024']
            n_ou_fmaps = ['64','64','64','128','128','128','256','256','256','256','256','256','1024','1024',  '10']
            infmap_siz = ['32','32','32', '16', '16', '16',  '8',  '8',  '8',  '8',  '8',  '8',   '1',   '1',   '1']

        # set output #neurons (that is, #classifications)
        global n_class
        n_ou_fmaps[len(n_ou_fmaps) - 1] = str(n_class)

        self.table.setRowCount(len(initial_options))
        for index in range(len(initial_options)):
            combo = QtGui.QComboBox()
            for t in self.combo_box_options:
                combo.addItem(t)
            combo.setCurrentIndex(initial_options[index])
            self.table.setCellWidget(index,0,combo)
            item1 = QtGui.QTableWidgetItem(n_in_fmaps[index])
            self.table.setItem(index,1,item1)
            item2 = QtGui.QTableWidgetItem(n_ou_fmaps[index])
            self.table.setItem(index,2,item2)
            item3 = QtGui.QTableWidgetItem(infmap_siz[index])
            self.table.setItem(index,3,item3)

            item4 = QtGui.QCheckBox('')
            item4.setChecked(True) # isChecked() == True?False?
            self.table.setCellWidget(index,4,item4)

        # Re-setting feature map size
        self.SetSize()

# -----------------------------------------------------------------------
# Plot Training Process (Train value, Test value)
# -----------------------------------------------------------------------
class Canvas(FigureCanvas):
    def __init__(self):
        FigureCanvas.__init__(self,Figure())
        self.ax = self.figure.add_subplot(111)
        self.train_acc=[0]*100
        self.test_acc=[0]*100
        self.train_loss=[0]*100
        self.test_loss=[0]*100
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("Accuracy[%]")
        self.ax.set_ylim(0,100)

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Loss")

        self.refresh(100)

    def refresh(self,xrange):
        self.ax = self.figure.add_subplot(111)
        self.ax.clear()
        self.ax.plot(range(0,len(self.train_acc)),np.ones(len(self.train_acc))*100.0 - self.train_acc,label='Accuracy(Train)',color="blue")
        self.ax.plot(range(0,len(self.test_acc)),np.ones(len(self.test_acc))*100.0 - self.test_acc,label='Accuracy(Test)',color="red")

        self.ax.annotate('Accuracy(Test)', 
            xy=(xrange - 1, 100.0 - self.test_acc[len(self.test_acc) - 1]), xycoords='data',
            xytext=(-100, -20), 
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->")
            )

        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("Accuracy[%]")
        self.ax.set_ylim(0,100)
        self.ax.set_xlim(0,xrange)
        self.ax.grid()

        self.ax2.clear()
        self.ax2.plot(range(0,len(self.train_loss)),self.train_loss,label='Loss(Train)',color="mediumslateblue")
        self.ax2.plot(range(0,len(self.test_loss)),self.test_loss,label='Loss(Test)',color="hotpink")

        self.ax2.annotate('Loss(Test)', 
            xy=(xrange - 1, self.test_loss[len(self.test_loss) - 1]), xycoords='data',
            xytext=(-80, 20), 
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->")
            )

        self.ax2.set_ylim(0,max(self.train_loss)*1.1)
        self.ax2.set_xlim(0,xrange)
        self.ax2.set_ylabel("Loss")

        self.draw()

    def push_data(self,train_acc,test_acc,train_loss,test_loss):
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.train_loss = train_loss
        self.test_loss = test_loss

###########################################################################################
# Main
###########################################################################################
def main():
    app = QtGui.QApplication(sys.argv)
    ex = Layout()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

###########################################################################################
# END OF PROGRAM
###########################################################################################
