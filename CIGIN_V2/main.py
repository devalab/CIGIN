# python imports
import pandas as pd
import warnings
import os
import argparse

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase


# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local imports
from model import *
from train import *
from molecular_graph import *

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin', help="The name of the current project: default: CIGIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                                          "tanh-general", default='dot')
parser.add_argument('--max_epochs', required=False, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=32, help="The batch size for training")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.isdir("runs/run-" + str(project_name)):
    os.makedirs("./runs/run-" + str(project_name))
    os.makedirs("./runs/run-" + str(project_name) + "/models")


def collate(samples):
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes)
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes)
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        solute = self.dataset.loc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)

        solvent = self.dataset.loc[idx]['SolventSMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)

        solvent_graph = get_graph_from_smile(solvent)
        delta_g = self.dataset.loc[idx]['DeltaGsolv']
        return [solute_graph, solvent_graph, [delta_g]]


def main():
    train_df = pd.read_csv('data/train.csv', sep=";")
    valid_df = pd.read_csv('data/valid.csv', sep=";")

    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)

    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)

    model = CIGINModel(interaction=interaction)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)


if __name__ == '__main__':
    main()
