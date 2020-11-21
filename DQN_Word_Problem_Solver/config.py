'''
Author: jirufengyu
Date: 2020-11-21 10:40:40
LastEditTime: 2020-11-21 10:59:43
LastEditors: jirufengyu
Description: Nothing
FilePath: /Signal-1/DQN_Word_Problem_Solver/config.py
'''
#import pydevd
#pydevd.settrace('121.49.110.81', port=30000, stdoutToServer=True, stderrToServer=True)
#import numpy as np
#
#array1 = np.array((10,10,10))
from gold_tree import *
import os
import json
from parse import *
import numpy as np

class Config:
    def __init__(self):
        self.parse_prefix_path = "/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/ai2data/"
        self.max_nodes_num = 6
        self.wp_total_num = 0
        self.parse_dict = self.parse_process_data(self.parse_prefix_path) 
        self.gold_ans_file_name = "/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/ai2gold.data"
        self.gold_trees = self.get_gold_ans(self.gold_ans_file_name)
        self.train_num = int(self.wp_total_num*0.8)
        #self.tain_num = 316
        self.validate_num = (self.wp_total_num - self.train_num)
        #self.validate_num = 79
        #self.train_list, self.validate_list = self.seperate_date_set()
        self.train_list = []
        self.validate_list = []
        self.index_file_name = "ai2index.data"
        self.ana_filename = "/home/stu2/Signal-1/DQN_Word_Problem_Solver/test/analysis"
        self.reject = self.read_reject_json()
        self.picks = self.read_pick('/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/pick2.json')

    def read_pick(self,filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def read_reject_json(self, filename = "/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/reject.json"):
        with open(filename, 'r') as f:
            reject = json.load(f)
        return reject    

    def read_wp_parse_from_json(self,prefix_path, name):
        filename = os.path.join(prefix_path, name)
        with open(filename, 'r') as f:
            data = json.load(f)
            return data

    def parse_process_data(self, prefix_path):
        file_len = len(os.listdir(prefix_path))
        self.wp_total_num = file_len
        parse_dict = {}
        for i in range(file_len):
            parse_per_wp = self.read_wp_parse_from_json(prefix_path, str(i)+'.json')
            parse_dict[i] = Parsing(parse_per_wp, i)
        return parse_dict

    def find_num_from_equstr(self,equ_str):
        state = 0
        temp = ''
        num_list = []
        for i in range(len(equ_str)):
            if equ_str[i].isdigit() or equ_str[i] == '.':
                state = 1
                temp += equ_str[i] 
                if i == len(equ_str)-1:
                    num_list.append(temp)
            elif state == 1:
                state = 0
                num_list.append(temp)
                num_list.append(equ_str[i])
                temp = ''
            else:
                num_list.append(equ_str[i])
        return num_list

    def get_gold_ans(self, filename):
        gold_list = []
        with open(filename, 'r') as f:
            for elem in f:
                a = elem[:elem.find(' ')]
                a = a[a.find("[u'")+3: elem.find(']')-1] 
                a = a[a.find("=")+1:]
                equ_str_l = self.find_num_from_equstr(a)
                ans = elem[elem.find(' ')+1:elem.find('\n')]
                gold_ans = ans[ans.find('[')+1:ans.find(']')]
                gtree = GoldTree(gold_ans, equ_str_l)
                #gtree.post_order(gtree.root)
                gold_list.append(gtree)
        return gold_list

    def seperate_date_set(self, index):
        with open("/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/train_fold_"+index+".json", 'r') as f:
             train_list = json.load(f)
        with open("/home/stu2/Signal-1/DQN_Word_Problem_Solver/data/test_fold_"+index+".json", 'r') as f:
             validate_list = json.load(f)
        return train_list,  validate_list 

