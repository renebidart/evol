import math
import random
import numpy as np
from random import randint

# All functions are specific to genetic algorithm on learning rate schedule
# thanks: "https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164"

def create_population(num_schedules, epochs=10):
    """ Create Learning rate schedules, called population 
    
    thanks: "https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164"
    Creates learning rate schedules by randomly sampling between 10e1 and 10e-6
    Generate a random float in [0, -6], take exp of this
    
    Args: num_schedules (int): number of random schedules to create
          epochs (int): Number of epochs for learning rate schedule
    Returns: learning rate schedule
    """
    pop = []
    for _ in range(0, num_schedules):
        # Create schedule
        exponents = np.random.uniform(-6, 0, epochs)
        lr_schedule = np.power(10, exponents).tolist()
        pop.append(lr_schedule)
    return pop

def breed(sch1, sch2, way='average', num_children=2):
    """Make two children as parts of their parents.
    
    Args:
        sch1 (list): lr_schedule
        sch2 (list): lr_schedule
    """
    children = []
    for _ in range(2):
        child = []

        # Loop through the parameters and pick params for the kid.
        for idx in range(len(sch1)):
            if(way=='random'):
                child.append(random.choice([sch1[idx], sch2[idx]]))
            elif(way=='mean'):
                child.append(np.mean(np.array([sch1[idx], sch2[idx]])))
        children.append(child)
    return children

def mutate(lr_schedule):
    """Randomly mutate one learning rate randomly
    
    Args:
        lr_schedule (list): lr schedule to mutate
    """
    # Choose a random key.
    idx = randint(0, len(lr_schedule)-1)
    
    # Mutate one of the params. Multiply by at most a factor of 10
    lr_schedule[idx] = lr_schedule[idx]*math.pow(10, random.uniform(-1, 1))
    return lr_schedule 


def evolve(pop_perf):
    """Evolve a population of learning rates. 
    Args:
        pop (list): A list of learning rates
    
    Process:
    1. Tests schedules, and then keeps the top 25%, as well as 10% chance of keeping a poor schedule.
    2. Randomly mutate kept networks with 50% prob
    3. Fill the ramaining slots in population with children, created by randomly combining the parents 
        (50%/50% change of averaging a parameter, or randomly selecting one)
        
    """
    # Sort on the scores.
    pop = [x[1] for x in sorted(pop_perf, key=lambda x: x[0], reverse=True)]

    # keep the best 25%
    retain_length = int(len(pop)*.25)

    # The parents are every network we want to keep.
    parents = pop[:retain_length]

    # Randomly mutate the networks we're keeping, and add these
    # This preserves the already good networks, so we don't lose out. 
    mutated = []
    for index, individual in enumerate(parents):
        mutated.append(mutate(parents[index]))
    parents.extend(mutated)

    # For those we aren't keeping, randomly add one to increase variance. Also mutate it??
    parents.append(mutate(random.choice(pop[retain_length:])))

    # Now find out how many spots we have left to fill. (how many children to make, little under 50% of full pop)
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []


    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:

        # Get a random mom and dad.
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]

            # pick breeding method:
            if random.random() > .5:
                way = 'mean'
            else:
                way = 'random'
                
            # Breed them.
            babies = breed(male, female, way, num_children=2)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)
    parents.extend(children)
    return parents

