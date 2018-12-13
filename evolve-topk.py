import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import cma
from scipy.stats import norm

from utils.data import make_generators_DF_cifar, make_generators_DF_MNIST
from utils.loading import net_from_args
from utils.scheduler import ListScheduler
from utils.train_val import train_net_evol
from models.PResNetTopK import PResNetTopK18


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--device', type=str)

# Defining the model/evol:
parser.add_argument('--within_block_act', default='relu', type=str)
parser.add_argument('--after_block_act', default=None, type=str)
parser.add_argument('--MAX_GENERATIONS', default=20, type=int)
parser.add_argument('--NPOPULATION', default=20, type=int)

# training params
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dataset', default='CIFAR', type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--IM_SIZE', default=32, type= int)
args = parser.parse_args()


def main(args):
    """Evolve to find the optimal top k-pooling for each layer / block"""
    print('CUDA VERSION:', torch.version.cuda)
    batch_size, num_workers, IM_SIZE, epochs= int(args.batch_size), int(args.num_workers), int(args.IM_SIZE), int(args.epochs)
    device = torch.device(args.device)

    NPARAMS = 4 # there are 4 blocks for topk
    NPOPULATION = int(args.NPOPULATION)  # population size
    MAX_GENERATIONS = int(args.MAX_GENERATIONS)  # number of generations
    
    within_block_act, after_block_act = str(within_block_act), str(after_block_act)
    model_name = 'PResNetTopK-'+str(within_block_act)+'_'+str(after_block_act)

    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with open(args.files_df_loc, 'rb') as f:
        files_df = pickle.load(f)
        
    epochs = 60
    group_list = [1, 1, 1, 1]
    
    # solutions generated from N(0, 1). later transformed [0, 1] with inv cdf
    es = cma.CMAEvolutionStrategy(NPARAMS * [0], 1)

    history = {}
    history['xbest'] = []
    history['fbest'] = []
    history['xfavorite'] = []
    history['NPOPULATION'] = NPOPULATION
    history['MAX_GENERATIONS'] = MAX_GENERATIONS

    for j in tqdm(range(MAX_GENERATIONS)):
        solutions = es.ask()
        fitness_list = np.zeros(es.popsize)

         # ??? Make generators for each generation. Does this matter? or just do it once???
        if args.dataset == 'CIFAR10':
            dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=IM_SIZE, 
                                                    path_colname='path', adv_path_colname=None, return_loc=False)
        elif args.dataset == 'MNIST':
            dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                    path_colname='path', adv_path_colname=None, return_loc=False, normalize=True)

        # evaluate each set of learning rates, using new model each time:
        for i in range(es.popsize):
            # convert the normal to a topk probability:
            topk_list = [norm.cdf(x) for x in solutions]
            
            # Create a model with this topk and train it:
            model = PResNetTopK18(block, num_blocks, within_block_act=within_block_act, after_block_act=after_block_act, 
                                  frac_list=topk_list, group_list=group_list, num_classes=10)
            model = model.to(device)
            metrics = train_net_evol(model, dataloaders, batch_size, epochs, device)
            
            # the fitness is the best validation accuracy *-1, because it tries to minimize
            fitness_list[i] = -1 * metrics['best_val_acc']
        es.tell(solutions, fitness_list)
        # es.logger.add()
        es.disp()
        result = es.result
        history['xbest'].append(result.xbest)
        history['fbest'].append(result.fbest)
        history['xfavorite'].append(result.xfavorite) # this is a weird one, maybe try it out
        print("fitness at generation", (j+1), result[1])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])

    print('es.result_pretty-------------------')
    es.result_pretty()
        
    pickle.dump(history, open(str(SAVE_PATH)+'/'+str(model_name)+'_bs_'+str(batch_size)+'_nGen_'+str(MAX_GENERATIONS)+'_nPop_'+str(NPOPULATION)+'_ep_'+str(NPARAMS)+'_history.pkl', "wb"))


if __name__ == '__main__':
    print(args)
    with torch.cuda.device(torch.device(args.device).index):
        main(args)
