#from subprocess import Popen, PIPE
from math import *
#import random
import random as pr
import numpy as np
from copy import deepcopy
##from types import ListType, TupleType, StringTypes
#import itertools
import time
#import math
import argparse
#import subprocess

#from rdkit.Chem.QED import qed
from load_model import loaded_model
#from keras.preprocessing import sequence
#from rdkit import Chem
#from rdkit.Chem import Draw
#from rdkit.Chem import Descriptors
#import sys
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node
from pygmo import hypervolume
import copy

import os
import json
#import traceback
import errno

import argparse

from joblib import Parallel, delayed
import pdb

class chemical:

    def __init__(self):

        self.position=['&']
        self.num_atom=8
        self.vl=['\n', '&', 'C', '(', 'c', '1', 'o', '=', 'O', 'N', 'F', '[C@@H]',
        'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]',
        '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '[N-]', '[n+]', '[S@@]', '[S-]',
        'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]',
        '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '[S@@+]']

    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class pareto:

    def __init__(self, front=[], size=0, avg=[], compounds=[]):
        self.front=front
        self.size=size
        self.avg=avg
        self.compounds=compounds

    def Dominated(self,m):
        if len(self.front) == 0:
            return False
        
        for p in self.front:
            flag = True
            for i in range(len(p)):
                if m[i]>=p[i]:
                    flag = False
            if(flag):
                return True
        
        return False

    def Update(self,scores,compound):
        del_list = []
        for k in range(len(self.front)):
            flag = True
            for i in range(len(self.front[k])):
                if(self.front[k][i]>=scores[i]):
                    flag = False
            if(flag):
                del_list.append(k-len(del_list))
        for i in range(len(del_list)):
            del self.front[del_list[i]]
            del self.compounds[del_list[i]]
        self.front.append(scores)
        self.compounds.append(compound)
        f = open(dataDir+"present/output.txt", 'a')
        
        print("pareto size:",len(self.front),file=f)
        print("Updated pareto front",self.front, file=f)
        print("Pareto Ligands",self.compounds,file=f)
        print("Time;",time.asctime( time.localtime(time.time()) ),file=f)
        f.close()
       
        print("pareto size:",len(self.front))
        print("Updated pareto front",self.front)
        
        self.avgcal()

    def avgcal(self):
        for i in range(len(self.avg)):
            self.avg[i] = 0
        for i in range(len(self.front)):
            for j in range(len(self.avg)):
                self.avg[j]+=self.front[i][j]/len(self.front)
    
    @staticmethod
    def from_dict(_filename):
        # should check _filename exists
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_pareto = pareto(front = _set_json['front'], size=_set_json['size'], avg=_set_json['avg'], compounds=_set_json['compounds'])
        _set_file.close()
        print("Loaded Pareto Fronts")
        return new_pareto

class Node:

    def __init__(self, position = None,  parent = None, state = None, childNodes=[], child=None, wins=[0,0,0], visits=0, nonvisited_atom=None, type_node= [], depth=0):
        #super().__init__()
        #self.__dict__ = self
        self.position = position
        self.parentNode = parent
        self.childNodes = childNodes
        self.child=child
        self.wins = wins
        self.visits = visits
        self.nonvisited_atom=state.Getatom() if nonvisited_atom is None else nonvisited_atom
        self.type_node=type_node
        self.depth=depth


    def Selectnode(self,pareto_front):##

        #s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + 0.8*sqrt(2*log(self.visits)/c.visits))[-1]
        #s=random.choice(self.childNodes)
        w=[]
        for i in range(len(self.childNodes)):
            ##ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+sqrt(2)*sqrt(2*log(self.visits)/self.childNodes[i].visits))
            ucb=[]
            for win in self.childNodes[i].wins:
                ucb.append(win/self.childNodes[i].visits+sqrt(2*log(self.visits)/self.childNodes[i].visits))
            w.append(self.childNodes[i].wcal(pareto_front,ucb))
        m = np.amax(w)
        indices = np.nonzero(w == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]

        return s

    def wcal(self,pareto,ucb):## cal W(s,a)
        dominated = pareto.Dominated(self.wins)
        hv = self.hvcal(pareto,ucb)
        if dominated:
            return hv - self.distance(pareto,ucb)
        else:
            return hv

    def distance(self, pareto, ucb):## get a approximate distance
        avg = pareto.avg
        distance = 0
        for i in range(len(avg)):
            distance += pow(avg[i]-ucb[i])

        return sqrt(distance)


    


    def hvcal(self,pareto,ucb):## cal hypervolume indicator
        if len(pareto.front) == 0:
            return 0
        _pareto_temp = copy.deepcopy(pareto.front)
        _pareto_temp.append(ucb)
        for i in range(len(_pareto_temp)):
            for j in range(len(_pareto_temp[0])):
                if(_pareto_temp[i][j]>0):
                    _pareto_temp[i][j] = -_pareto_temp[i][j]
                else:
                    _pareto_temp[i][j] = -0.00000000000000001
        hv = hypervolume(_pareto_temp)
        ref_point = [0,0,0]
        hvnum = 0
        try:
            hvnum = hv.compute(ref_point)
        except:
            f = open("./data/present/hverror_output.txt", 'a')
            print(time.asctime( time.localtime(time.time()) ),file=f)
            print(pareto.front,file=f)
            f.close()
        ##print(hvnum)
        #pdb.set_trace()
        return hvnum

    def Addnode(self, m, s):

        n = Node(position = m, parent = self, state = s)
        if not n in self.childNodes:
            self.childNodes.append(n) #TODO: need comment out? from 96h
            pass

    def simulation(self,state):
        predicted_smile=predict_smile(model,state)
        input_smile=make_input_smile(predicted_smile)
        logp,valid_smile,all_smile=logp_calculation(input_smile)

        return logp,valid_smile,all_smile

    #@staticmethod
    def preprocess_todict(self):
        """
        preprocess to dictional serialize:
            deleting parentNode field (for avoiding circle ref.)
        """
        self.parentNode = None
        for cn in self.childNodes:
            print(cn)
            print(cn.depth)
            print(cn.childNodes)
            if self != cn:
                cn.preprocess_todict()
            else:
                self.childNodes.remove(cn)
                pass
        return self

    #@staticmethod
    def preprocess_fromdict(self):
        """
        preprocess from dictional serialize:
            adding parentNode field (for having avoided circle ref.)
        """
        for cn in self.childNodes:
            cn.parentNode = self
            cn.preprocess_fromdict()
    

    def Update(self, result):

        self.visits += 1
        for i in range(len(self.wins)):
            self.wins[i]+=result[i]
        return self

    @staticmethod
    def from_dict(_filename):
        #must check if _filename file exists
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_root = Node(position =_set_json['position'], parentNode=None, childNodes=None, child=_set_json['child'], visits=_set_json['visits'], nonvisited_atom=_set_json['nonvisited_atom'], type_node=_set_json['type_node'], depth=_set_json['depth'])
        _set_file.close()
        while True:
            new_root.childNodes #TODO:
        print("Loaded Pareto Fronts")
        return new_pareto
        self.position = position
        self.parentNode = parent
        self.childNodes = childNodes
        self.child=child
        self.wins = wins
        self.visits = visits
        self.nonvisited_atom=state.Getatom() if nonvisited_atom is None else nonvisited_atom
        self.type_node=type_node
        self.depth=depth

def _sigmoidnormalize(score:float)-> float:
    threshold = -5
    return 1 / (1+ np.exp(score - threshold))

def _linearnormalize(score: float)-> float:
    return -score

def _sbmolgennormalize(score:float)-> float:
    base_dock_score = 0
    return -round(((score - base_dock_score)*0.1)/(1+abs((score - base_dock_score)*0.1)),3)

def MCTS(root, pareto=pareto(), time_limit_sec=3600*240):
    # initial time-limit is 240h
    """initialization of the chemical trees and grammar trees"""
    run_time=time.time()+time_limit_sec
    rootnode = Node(state = root)
    state = root.Clone()
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    all_simulated_compound=[]
    desired_compound=[]
    max_logp=[]
    desired_activity=[]
    depth=[]
    min_score=1000
    score_distribution=[]
    min_score_distribution=[]
    dock_score=[]
    sascore=[]
    qedscore=[]
    default_reward = [[0,0,0]]
    penalty_reward = [-1. , -1. , -1.]
    
    """
    get reward function from python cfg
    """
    if os.path.exists(dataDir+'/input/python_config.json') :
        config = json.load(open(dataDir+'input/python_config.json'))
        REWARD = config['reward'] #normal,sigmoid,nonormal

    """----------------------------------------------------------------------"""

    while time.time()<=run_time:

        node = rootnode # important !    this node is different with state / node is the tree node
        state = root.Clone() # but this state is the state of the initialization .  too important !!!
        """selection step"""
        node_pool=[]
        
        while node.childNodes!=[]:
            #pdb.set_trace()
            if not int(pow(node.visits +1, 0.5))==int(pow(node.visits, 0.5)):
                break
            new_node = node.Selectnode(pareto)
            if new_node == node:
                node = new_node
                break
            node = new_node
            state.SelectPosition(node.position)
        #print("state position:,",state.position)

        ## Check in next test
        
        if node.position == '\n':
            
            print("end with \\n")
            while node != None:
                node.Update(penalty_reward)
                node = node.parentNode
            continue
        if len(state.position)>= 70:
            
            print("position bigger than 70")
            while node != None:
                node.Update(penalty_reward)
                node = node.parentNode
            continue
        

        """------------------------------------------------------------------"""
        
        """expansion step"""
        expanded=expanded_node(model,state.position,val)
        nodeadded=node_to_add(expanded,val)
        all_posible=chem_kn_simulation(model,state.position,val,nodeadded)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)


        """"simulation"""
        node_index,scores,valid_smile=check_node_type(new_compound,dataDir)
        f = open(dataDir+"present/ligands.txt", 'a')
        for p in valid_smile:
            print(p,file=f)
        f.close()
        
        f = open(dataDir+"present/scores.txt", 'a')
        for s in scores:
            print(s,file=f)
        f.close()
        if len(node_index)==0:
            
            while node != None:
                node.Update(default_reward[0])
                node = node.parentNode
            continue
        re=[]
        for i in range(len(node_index)):
            m=node_index[i]
            newflag = True
            for j in range(len(node.childNodes)):
                if(node.childNodes[j].position == nodeadded[m]):
                    newflag = False
                    node_pool.append(node.childNodes[j])
            if newflag:
                node.Addnode(nodeadded[m],state)##
                if len(node.childNodes) >0:
                    node_pool.append(node.childNodes[-1])
            
            ##node_pool.append(node.childNodes[i])
            f = open(dataDir+"present/depth.txt", 'a')
            print(len(state.position),file=f)
            ##depth.append(len(state.position))
            ##print("current minmum score",min_score)
            ## old
            ##if rdock_score[i]<=min_score:
            ##    min_score_distribution.append(rdock_score[i])
            ##    min_score=rdock_score[i]
            ##else:
            ##    min_score_distribution.append(min_score)

            ##re.append((-0.8*rdock_score[i])/(1+0.8*abs(rdock_score[i])))

            ## new
            
            ##min_score_distribution.append(scores[i])
            
            ##re.append(scores[i])## todo: reward fucntion
            ##dock_score.append(scores[i][0])
            ##sascore.append(scores[i][1])
            ##qedscore.append(scores[i][2])
            '''scores[0] Docking Score'''
            base_dock_score = 0## need set for every compound
            match REWARD:
                case "normal":
                    scores[i][0]= _sbmolgennormalize(scores[i][0])
                case "sigmoid":
                    scores[i][0]= _sigmoidnormalize(scores[i][0])
                case "nonormal":
                    scores[i][0]= _linearnormalize(scores[i][0])
            '''scores[1] QED'''
            ##scores[i][1]=round(1-scores[i][1]/10,3)## For SA score
            #logpcenter= 1.4
            #scores[i][1] = 1 - pow((0.5*(scores[i][1]-logpcenter)),2) ## For logP
            #if scores[i][1]<0:
            #    scores[i][1]=0
            '''scores[2] etoxpred'''

            if pareto.Dominated(scores[i]) == False:
                pareto.Update(scores[i],valid_smile[i])
                print("Time: ",time.asctime( time.localtime(time.time()) ))

            re.append(scores[i])
            
        """backpropation step"""

        for i in range(len(node_pool)):

            node=node_pool[i]
            while node != None:
                node.Update(re[i])
                node = node.parentNode
                    




        """check if found the desired compound"""
        print("End Search Epoch: ", time.asctime( time.localtime(time.time()) ))
        pareto_file = open(dataDir+'present/pareto.json', 'w')
        json.dump(pareto.__dict__,pareto_file, indent=4, separators=(',', ': '))
        pareto_file.close()

        """ mct_file = open(dataDir+'present/tree.json', 'w')
        savenodes = copy.deepcopy(rootnode)
        savenodes = savenodes.preprocess_todict()
        json.dump(savenodes.__dict__, mct_file, indent=4, separators=(',', ': '))
        mct_file.close() """

    #print "all valid compounds:",valid_compound
    #print "all active compounds:",desired_compound
    ##print("dock_score",dock_score)
    ##print("sa_score",sascore)
    ##print("qed_score",qedscore)
    ##print("num valid_compound:",len(valid_compound))
    ##print("valid compounds",valid_compound)
    ##print("depth",depth)
    ##print("min_score",min_score_distribution)
    print("pareto front",pareto.compounds)
    print("pareto front scores",pareto.front)


    return valid_compound


def UCTchemical(time_limit_sec=3600*240):
    one_search_start_time=time.time()
    time_out=one_search_start_time+60*10
    state = chemical()
    pareto_front = pareto() if isLoadTree is False else pareto.from_dict(pareto_locate)
    best = MCTS(root = state,pareto=pareto_front, time_limit_sec = time_limit_sec)


    return best


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='search molecular')

    parser.add_argument('dataDir',help='path to data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    if os.path.exists(dataDir+'/input/python_config.json') :
        config = json.load(open(dataDir+'input/python_config.json'))
        isLoadTree = config['isLoadTree']
        pareto_locate = dataDir+'present/pareto.json'
        hours = config['limitTimeHours']
        minutes = config['limitTimeMinutes']
        seconds = config['limitTimeSeconds']
        rnnModelDir = config['whereisRNNmodelDir']
    else :
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),dataDir+'/input/python_config.json')
        #try: 
        #except FileNotFoundError:
        #    traceback.print_exc()
    smile_old=zinc_data_with_bracket_original()
    val,smile=zinc_processed_with_bracket(smile_old)
    #print(val)
    model=loaded_model(rnnModelDir)
    valid_compound=UCTchemical(time_limit_sec=hours*3600+minutes*60+seconds)
