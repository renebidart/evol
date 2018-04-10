from models import*
from genetic import*
import sys
import argparse

# Assume runnign on GPU so cuda will be enabled

def main(args):
    import os
    import sys
    import glob
    import random
    import numpy as np
    import pandas as pd
    from PIL import Image
    import pickle

    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data
    import torch.utils.data.sampler as sampler

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print('torch.cuda.current_device(): ', torch.cuda.current_device())
    print('torch.cuda.get_device_name(0): ', torch.cuda.get_device_name(0))

    def run_genetic(data_loc, out_loc, generations, num_schedules, epochs, batch_size, momentum, kwargs):
        """Run the genetic algorithm on schedules for number of generations. Save best model at each generation and final best 5 models
        """
    
        # initialize schedules:
        population = create_population(num_schedules=num_schedules, epochs=epochs)

        # Using the same training split for all, create data loaders
        indexes = list(range(60000))
        random.shuffle(indexes)
        valid_frac = .2
        train_samples_index = indexes[int(valid_frac*len(indexes)):]
        valid_samples_index = indexes[0:int(valid_frac*len(indexes))]
        train_loader, valid_loader, test_loader = CFIAR_data_loaders(data_loc, train_samples_index, valid_samples_index, kwargs, batch_size=batch_size)

        # Create network
        model = SmallNet()
        model.cuda()

        # Store the top schedule and accuracy (tuples) as elements in a list.
        history=[]

        # Evolve the generation.
        for i in range(generations):
            print('Running generation: ', i)
            pop_perf = model.get_population_perf(population, train_loader, valid_loader, test_loader, momentum)
            pop_perf = [x for x in sorted(pop_perf, key=lambda x: x[0], reverse=True)]
            history.append(pop_perf[0])
            
            # print average accuracy, best accuracy, and best schedule
            perf_only = [x[0] for x in pop_perf]
            avg = sum(perf_only)/len(perf_only)
            print('Avg acc: ', avg, 'best acc: ', pop_perf[0][0])
            print('Schedule: ',[ '%.5f' % elem for elem in pop_perf[0][1]])

            # Evolve
            population = evolve(pop_perf)
        
        # get final accuracy, and print the top 5 sorted
        pop_perf = model.get_population_perf(population, train_loader, valid_loader, test_loader, momentum)
        pop_perf = [x for x in sorted(pop_perf, key=lambda x: x[0], reverse=True)]

        # Print out the top 5 networks.
        print('Final Results: ', pop_perf[:5])

        # save history as a pickle file
        out_file = os.path.join(out_loc, 'evol_cifar_gener_'+str(generations)+'_numsch_'+str(epochs)+'_epochs_'+str(epochs))
        pickle.dump(history, open(out_file, 'wb'))

    run_genetic(args.data_loc, args.out_loc, args.generations, args.num_schedules, args.epochs, args.batch_size, args.momentum, kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm on Learning Rate Schedule')
    parser.add_argument('--data_loc', type=str)
    parser.add_argument('--out_loc', type=str)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--num_schedules', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true', default=True) # not used
    args = parser.parse_args()

    main(args)

