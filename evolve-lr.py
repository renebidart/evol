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
# from es import CMAES

from utils.data import make_generators_DF_cifar, make_generators_DF_MNIST
from utils.loading import net_from_args
from utils.scheduler import ListScheduler
from utils.train_val import train_net


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--device', type=str)

# Defining the model/evol:
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=18, type=int, help='depth of model')
parser.add_argument('--MAX_GENERATIONS', default=20, type=int)
parser.add_argument('--NPOPULATION', default=20, type=int)

# training params
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--IM_SIZE', default=28, type= int)
args = parser.parse_args()


def main(args):
    batch_size, num_workers = int(args.batch_size), int(args.num_workers)
    IM_SIZE = int(args.IM_SIZE)
    device = torch.device(args.device)

    NPARAMS = int(args.epochs)  # one learning rate (parameter) per epoch
    NPOPULATION = int(args.NPOPULATION)  # population size
    MAX_GENERATIONS = int(args.MAX_GENERATIONS)  # number of generations

    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with open(args.files_df_loc, 'rb') as f:
        files_df = pickle.load(f)


    es = cma.CMAEvolutionStrategy(NPARAMS * [-2], 1) # solutions generated from N(-2, 1), but transformed to 10^sol

    history = []
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
            model, model_name = net_from_args(args, num_classes=10, IM_SIZE=IM_SIZE)
            model = model.to(device)

            # convert the exponenet to a learning rate:
            lr_list = np.power(10, solutions[i]).tolist()
            # Train it for the given lr list:
            metrics = train_net(model, dataloaders, lr_list, batch_size, device)
            # the fitness is the best validation accuracy *-1, because it tries to minimize
            fitness_list[i] = -1 * metrics['best_val_acc']
        es.tell(solutions, fitness_list)
        # es.logger.add()
        es.disp()
        result = es.result # first element is the best solution, second element is the best fitness
        history.append(result)
        print("fitness at generation", (j+1), result[1])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])

    print('es.result_pretty-------------------')
    es.result_pretty()
        
    pickle.dump(history, open(str(SAVE_PATH)+'/'+str(model_name)+'_bs_'+str(batch_size)+'_nGen_'+str(MAX_GENERATIONS)+'_nPop_'+str(NPOPULATION)+'_metrics.pkl', "wb"))


if __name__ == '__main__':
    print(args)
    with torch.cuda.device(torch.device(args.device).index):
        main(args)
