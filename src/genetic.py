import math
import random
import numpy as np
from random import randint

# All functions are specific to genetic algorithm on learning rate schedule
# thanks: "https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164"

def create_population(num_schedules, size=10, rate_range=(-6, 0)):
    """ Create Learning rate schedules, called population 
    
    Args: num_schedules (int): number of random schedules to create
          size (int): Number of points in learning rate schedule
          top, bottom = range of sampling, exp oth these will be used
          mom_range = range of momentum (default = .9)
    Returns: learning rate schedule
    """
    pop = []
    for _ in range(0, num_schedules):
        exponents = np.random.uniform(rate_range[0], rate_range[1], size)
        schedule = np.power(10, exponents).tolist()
        pop.append(schedule)
    return pop


def breed_r_m(sch1, sch2, way='average'):
    """Make two children as parts of their parents. 
    
    Args:
        sch1 (list): Any list of parameters to breed together. Most have same order.
        sch2 (list): Any list of parameters to breed together. Most have same order.
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

def breed_slice(sch1, sch2, way='average'):
    """Make two children as parts of their parents. 
    
    Args:
        sch1 (list): Any list of parameters to breed together. Most have same order.
        sch2 (list): Any list of parameters to breed together. Most have same order.
    """
    children = []
    for _ in range(2):
        child = []

        # part before idx goes from sch1 to sch2.
        idx = randint(0, len(sch1)-1)
        tmp = sch1[:idx]
        sch1[:idx] = sch2[:idx]
        sch2[:idx] = tmp
    return [sch1, sch2]


def mutate(lr_schedule):
    """Randomly mutate a single parameter by multiplying by a 10^N(0,1)
    
    Args:
        lr_schedule (list): Any list of parameters to mutate
    """
    for i in range(2): # mutate two to increase variance
        # Choose a random key.
        idx = randint(0, len(lr_schedule)-1)
        # Mutate one of the params. Will be within (1/10, 10)*Old param 68% of the time
        lr_schedule[idx] = lr_schedule[idx]*math.pow(10, random.normalvariate(0, 1.5)) 
    return lr_schedule 


def evolve(pop_perf, breed_method):
    """Evolve a population of learning rates. 
    Args:
        pop_perf (list of tuples): A list of tuples. First element is performance. Second is the parameter list.
    
    Process:
    1. Keeps the top 25% of schedules, as well as 10% chance of keeping a poor schedule.
    2. Mutate all kept networks
    3. Fill the ramaining slots in population with children, created by randomly combining the parents 
        (50%/50% change of averaging a parameter, or randomly selecting one)
    The returned schedules will be:
    25% good
    25% mutated good
    10% mutated bad
    40% Children of these above networks
        
    note that when breeding we should have at least 3 unique parents, plus the less unique mutations and 
    """
    # Sort on the scores.
    pop = [x[1] for x in sorted(pop_perf, key=lambda x: x[0], reverse=True)]

    # keep the best 25%
    retain_length = 2 #int(np.ceil(len(pop)*.25))

    # The parents are every network we want to keep.
    parents = pop[:retain_length]

    # Randomly mutate the networks we're keeping, and add these
    # This preserves the already good networks, so we don't lose out.
    mutated = []
    for index, individual in enumerate(parents):
        mutated.append(mutate(parents[index]))
    parents.extend(mutated)

    # For those we aren't keeping, randomly add 10% of population to increase variance. Mutate them individually, then add. 
    # Mutation because we already know they are bad, should try something else. Something like that.
    num_poor = 2#int(math.ceil(len(pop)*.1))
    poor_keeping = random.sample(pop[retain_length:], num_poor)
    for poor_sch in poor_keeping:
        parents.append(mutate(poor_sch))

    # Now find out how many spots we have left to fill. (how many children to make, about 40% of full pop)
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
            babies = breed_method(male, female, way)

            # children.append(babies[desired_length:])
            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)
    parents.extend(children)
    return parents