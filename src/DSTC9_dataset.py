import os
import re
import sys
import json
import random
import numpy as np
import torch
import configparser
from torch.autograd import Variable
import config

USE_CUDA = True

class DatasetDSTC9(object):
    '''
    data container for woz dataset
    '''

    def __init__(self, percentage=1.0):
        # setup
        # 		feat_file = config['DATA']['feat_file']
        # 		text_file = config['DATA']['text_file']
        # #		dataSplit_file = config['DATA']['dataSplit_file']
        # 		vocab_file = config['DATA']['vocab_file']
        # 		template_file = config['DATA']['template_file']

        self.vocab_file = "../dstc9/dstc9_vocab.txt"
        self.train_file = "../dstc9/train.json"
        self.val_file = "../dstc9/val.json"
        self.test_file = "../dstc9/test.json"
        # feat_file = "../resource/woz3/feat.json"
        # text_file = "../resource/woz3/text.json"
        # template_file = "../resource/woz3/template.txt"
        # dataSplit_file = "../resource/woz3/data_split/Boo_ResDataSplitRand0925.json"
        # TODO: Look at the code.


        self.data = {'train': [], 'valid': [], 'test': []}
        self._load_data()
        #		self.data   = {'train':[],'valid':[],'test_seen':[], 'test_unseen':[]}
        #		self.data_index  = {'train': 0, 'valid': 0, 'test_seen': 0, 'test_unseen': 0} # index for accessing data

    def _load_data(self):
        with open(self.train_file, 'r') as myfile:
            data = myfile.read()
        train = json.loads(data)
        self.data["train"] = train

        with open(self.val_file, 'r') as myfile:
            data = myfile.read()
        valid = json.loads(data)
        self.data["valid"] = valid

        with open(self.test_file, 'r') as myfile:
            data = myfile.read()
        test = json.loads(data)
        self.data["test"] = test
