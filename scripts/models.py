import random
import pandas as pd
import pickle
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import seaborn as sns


from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

from molecular_graph import ConstructMolecularGraph

device = "cuda" if torch.cuda.is_available() else "cpu"

class MessagePassing(nn.Module):
    
    def __init__(self, node_dim, edge_dim,T):
        super(MessagePassing, self).__init__()
        self.node_dim= node_dim
        self.edge_dim = edge_dim

        self.T = T

        self.U_0 = nn.Linear(2*self.node_dim + self.edge_dim,self.node_dim)
        self.U_1 = nn.Linear(2*self.node_dim + self.edge_dim,self.node_dim)
        self.U_2 = nn.Linear(2*self.node_dim + self.edge_dim,self.node_dim)
        
        self.M_0 = nn.Linear(2*self.node_dim , self.node_dim)
        self.M_1 = nn.Linear(2*self.node_dim , self.node_dim)
        self.M_2 = nn.Linear(2*self.node_dim , self.node_dim)


    def message_pass(self,g,h,k):
        message_list = []
        for v in g.keys():
            neighbors = g[v]
            reshaped_list = []
            for neighbor in neighbors:
                e_vw = neighbor[0] # feature variable
                w = neighbor[1]
                reshaped = torch.cat((h[v].view(1,-1), h[w].view(1,-1), e_vw.view(1,-1)), 1)
                if k == 0:
                    reshaped_list.append(self.U_0(reshaped))
                elif k == 1:
                    reshaped_list.append(self.U_1(reshaped))
                elif k == 2:
                    reshaped_list.append(self.U_2(reshaped))
            message_list.append(torch.sum(torch.stack(reshaped_list),0))
        
        i = 0
        for v in g.keys():
            if k == 0:
                h[v] = F.relu(self.M_0(torch.cat([h[v].view(1,-1),message_list[i]],1)))
            elif k == 1:
                h[v] = F.relu(self.M_1(torch.cat([h[v].view(1,-1),message_list[i]],1)))
            elif k == 2:
                h[v] = F.relu(self.M_2(torch.cat([h[v].view(1,-1),message_list[i]],1)))
            i += 1
                
    def forward(self,edge_features, node_features):
        self.edge_features = edge_features
        self.node_features =  node_features
        for k in range(0,self.T):
            self.message_pass(self.edge_features,self.node_features,k)
        
        return self.edge_features,self.node_features

class ReadoutLayer(nn.Module):
    def __init__(self, node_dim, edge_dim,mem_dim):
        super(ReadoutLayer, self).__init__()
        self.edge_dim = edge_dim
        self.node_dim =  node_dim
        self.mem_dim =  mem_dim
        self.transform_layer = nn.Linear(2*node_dim,2*node_dim)
        
    def forward(self,v0,v1):

        catted_reads = torch.cat([v0,v1], 1)
        activated_reads = F.relu( self.transform_layer (catted_reads))
        readout = torch.zeros(1, 2*self.node_dim).to(device)

        for read in activated_reads:
            readout = readout + read

        return readout

class Cigin(nn.Module):
    def __init__(self, node_dim=40, edge_dim=10, T=3):
        super(Cigin, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.T = T
        self.LSTM_t = 2
        self.solute_pass = MessagePassing(self.node_dim, self.edge_dim, self.T)
        self.solvent_pass = MessagePassing(self.node_dim, self.edge_dim, self.T)

        self.lstm_solute = torch.nn.LSTM(2*self.node_dim,self.node_dim)
        self.lstm_solvent = torch.nn.LSTM(2*self.node_dim,self.node_dim)
        
        self.lstm_gather_solute = torch.nn.LSTM(2*2*2*self.node_dim,2*2*self.node_dim)
        self.lstm_gather_solvent = torch.nn.LSTM(2*2*2*self.node_dim,2*2*self.node_dim)
        
        self.first_layer = nn.Linear(16*self.node_dim,360)
        self.second_layer = nn.Linear(360,200)
        self.third_layer = nn.Linear(200,120)
        self.fourth_layer = nn.Linear(120,1)
    
    def set2set(self,tensor,no_of_features,no_of_steps,lstm):
        ##### input format ########   no_of_atoms X timesteps X lengthof feature vector
        n = tensor.shape[0]
        tensor=tensor.transpose(0,1)
        q_star = torch.zeros(n,2*no_of_features).to(device)
        hidden = (torch.zeros(1, n, no_of_features).to(device),
              torch.zeros(1, n, no_of_features).to(device))
        for i in range(no_of_steps):
            q,hidden = lstm(q_star.unsqueeze(0),hidden)
            e = torch.sum(tensor*q,2)
            a = F.softmax(e,dim=0)
            r = a.unsqueeze(2)*tensor
            r=  torch.sum(r,0)
            q_star = torch.cat([q.squeeze(0),r],1)
        return q_star

    
    def forward(self,solute,solvent):
        
        #Construct molecular graph for solute.
        solute = Chem.MolFromSmiles(solute)
        solute = Chem.AddHs(solute)
        solute = Chem.MolToSmiles(solute)        
        edges_solute_0, nodes_solute_0 = ConstructMolecularGraph(solute)
        self.edges_solute_0 = deepcopy(edges_solute_0)
        self.nodes_solute_0 =  deepcopy(nodes_solute_0)
        
        #Message Passing for solute
        self.edges_solute_t, self.nodes_solute_t = self.solute_pass(edges_solute_0,nodes_solute_0)
        
        #Gather phase for solute
        self.node_features_0 = torch.stack([self.nodes_solute_0[i] for i in self.nodes_solute_0]).reshape(len(self.nodes_solute_0),self.node_dim)
        self.node_features_t = torch.stack([self.nodes_solute_t[i] for i in self.nodes_solute_t]).reshape(len(self.nodes_solute_0),self.node_dim)
        set2set_input_solute = torch.stack([self.node_features_0,self.node_features_t],1)
        gather_solute = self.set2set(set2set_input_solute,self.node_dim,self.LSTM_t,self.lstm_solute) #A
        
        
        #Construct molecular graph for solute.
        solvent = Chem.MolFromSmiles(solvent)
        solvent = Chem.AddHs(solvent)
        solvent = Chem.MolToSmiles(solvent)
        edges_solvent_0, nodes_solvent_0 = ConstructMolecularGraph(solvent)
        self.edges_solvent_0 = deepcopy(edges_solvent_0)
        self.nodes_solvent_0 = deepcopy(nodes_solvent_0)
        
        #Message passing for solvent
        self.edges_solvent_t, self.nodes_solvent_t = self.solvent_pass(edges_solvent_0,nodes_solvent_0)


        #Gather phase for solvent
        self.node_features_0 = torch.stack([self.nodes_solvent_0[i] for i in self.nodes_solvent_0]).reshape(len(self.nodes_solvent_0),self.node_dim)
        self.node_features_t = torch.stack([self.nodes_solvent_t[i] for i in self.nodes_solvent_t]).reshape(len(self.nodes_solvent_0),self.node_dim)
        set2set_input_solvent = torch.stack([self.node_features_0,self.node_features_t],1)
        gather_solvent = self.set2set(set2set_input_solvent,self.node_dim,self.LSTM_t,self.lstm_solvent) #B
        
        #Interaction phase
        
        combined_features_no_of_features=2*self.node_dim
        n = len(self.nodes_solute_t) # no of atoms in solute
        m = len(self.nodes_solvent_t) # no of atoms in solvent
        
        interaction_map = torch.zeros(n,m)
        interaction_map_2 = torch.zeros(n,m)
        
        for i,solute_row in enumerate(gather_solute):
            for j,solvent_row in enumerate(gather_solvent):
                interaction_map[i, j] = torch.sum(torch.mul(solute_row, solvent_row))
                interaction_map_2[i, j] = torch.sum(torch.mul(solute_row, solvent_row))

        
        interaction_map_2 = torch.tanh(interaction_map_2) #I
        
        
        solute_after_interaction = torch.mm(interaction_map_2, gather_solvent) #A'
        solvent_after_interaction = torch.mm(interaction_map_2.t(), gather_solute) #B'

        #Prediction phase
        combined_features_solute_features = self.set2set(torch.cat([solute_after_interaction, gather_solute],1).unsqueeze(0),
                                     2*combined_features_no_of_features,2,self.lstm_gather_solute) #A''
        combined_features_solvent_features = self.set2set(torch.cat([solvent_after_interaction, gather_solvent],1).unsqueeze(0),
                                      2*combined_features_no_of_features,2,self.lstm_gather_solvent) #B''
        
        combined_features = torch.cat([combined_features_solute_features,combined_features_solvent_features], 1)
        combined_features = F.relu(self.first_layer(combined_features))
        combined_features = F.relu(self.second_layer(combined_features))
        combined_features = F.relu(self.third_layer(combined_features))
        combined_features = self.fourth_layer(combined_features)
        
        return combined_features, interaction_map.detach()
        