from random import randint, random
import numpy as np
from operator import xor
import math
import matplotlib.pyplot as plt
import sys

plt.style.use('fivethirtyeight')


class MOP():
    def __init__(self, pop_shape, pc=0.8, pm=0.005, max_round=100, chrom_l=[0, 0], low=[0, 0], high=[0, 0]):
        self.pop_shape = pop_shape
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.chrom_l = chrom_l
        self.low = low
        self.high = high

    def initialization(self):  # initialize first population
        pop = np.random.randint(
            low=0, high=2, size=self.pop_shape)  # random number 0,1
        return pop

    def crossover(self, ind_0, ind_1):  # cross over for two individual (one point crossover)
        new_0, new_1 = [], []
        # check two individuals have same lenght
        assert(len(ind_0) == len(ind_1))
        p_pc = np.random.random_sample(1)
        if p_pc < self.pc:  # doing crossover
            point = np.random.randint(len(ind_0))
            new_0 = list(np.hstack((ind_0[:point], ind_1[point:])))
            new_1 = list(np.hstack((ind_1[:point], ind_0[point:])))
        else:  # Transfer without crossover
            new_0 = list(ind_0)
            new_1 = list(ind_1)
        # check two new childs have same lenght
        assert(len(new_0) == len(new_1))

        return new_0, new_1

    def mutation(self, pop):
        # Calculate the number of bits that must mutation
        num_mut = math.ceil(self.pm * pop.shape[0] * pop.shape[1])
        for m in range(0, num_mut):
            i = np.random.randint(0, pop.shape[0])
            j = np.random.randint(0, pop.shape[1])
            pop[i][j] = xor(pop[i][j], 1)
        return pop

    def fitnessFunc(self, real_val, pareto):
        w1 = 0.5
        w2 = 1 - w1
        fit1 = 2 + pow(real_val[0]-2, 2) + pow(real_val[1]-1, 2)
        fit2 = (9*real_val[0])-pow(real_val[1]-1, 2)
        fitness_val = 1/(w1 * fit1 + w2 * fit2)
        penalty = self.penalty(real_val)
        fitnessPenalty = fitness_val-penalty
        if pareto == 1:
            print("fit1:",fit1,"\tfit2:",fit2)
        if fitnessPenalty < 0:
            return 0.00000000000001
        else:
            return fitnessPenalty

    def penalty(self, real_val):
        penaltySumation = 0
        c1, c2, k1, k2 = 1.7, 1.3, 2, 2
        v1 = np.abs(pow(real_val[0], 2) + pow(real_val[1], 2) - 225)
        v2 = np.abs(real_val[0]-(3*real_val[1])+10)
        if v1 > 0:
            penaltySumation += c1*pow(v1,k1)
        elif v2 > 0:
            penaltySumation += c2*pow(v2,k2)
        return penaltySumation

    def b2d(self, list_b):  # convert binary number to decimal number
        l = len(list_b)
        sum = 0
        for i in range(0, l):
            p = ((l-1)-i)
            sum += (pow(2, p) * list_b[i])
        return sum

    def d2r(self, b2d, lenght_b, m):  # Change the decimal number to fit in the range of problem variables
        norm = b2d/(pow(2, lenght_b) - 1)
        match m:
            case 0:
                real = self.low[0] + (norm * (self.high[0] - self.low[0]))
                return real
            case 1:
                real = self.low[1] + (norm * (self.high[1] - self.low[1]))
                return real

    # decoding the chromosome value for calculate fitness
    def chromosomeDecode(self, pop):
        gen = []
        for i in range(0, pop.shape[0]):
            l1 = pop[i][0:self.chrom_l[0]]
            l2 = pop[i][self.chrom_l[0]:]
            gen.append(self.d2r(self.b2d(list(l1)), len(l1), 0))
            gen.append(self.d2r(self.b2d(list(l2)), len(l2), 1))
        return np.array(gen).reshape(pop.shape[0], 2)

    def roulette_wheel_selection(self, population):
        chooses_ind = []
        population_fitness = sum([self.fitnessFunc(population[i],0)
                                 for i in range(0, population.shape[0])])
        chromosome_fitness = [self.fitnessFunc(population[i],0)
                              for i in range(0, population.shape[0])]
        # Calculate the probability of selecting each chromosome based on the fitness value
        chromosome_probabilities = [
            chromosome_fitness[i]/population_fitness for i in range(0, len(chromosome_fitness))]

        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(chromosome_probabilities))], p=chromosome_probabilities))  # Chromosome selection based on their probability of selection
        return chooses_ind  # return selected individuals

    def selectInd(self, chooses_ind, pop):  # Perform crossover on the selected population
        new_pop = []
        for i in range(0, len(chooses_ind), 2):
            a, b = self.crossover(pop[chooses_ind[i]], pop[chooses_ind[i+1]])
            new_pop.append(a)
            new_pop.append(b)
        npa = np.asarray(new_pop, dtype=np.int32)
        return npa

    def bestResult(self, population):  # calculate best fitness, avg fitness
        population_fitness = [1/self.fitnessFunc(
            population[i],1) for i in range(0, population.shape[0])]
        population_best_fitness = np.sort(population_fitness, kind="heapsort")[:5]
        agents_index = np.argsort(population_fitness, kind="heapsort")[:5]
        agents = population[agents_index]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population_fitness, agents

    def run(self):  # start algorithm
        avg_population_fitness = []
        population_best_fitness = []
        population_fitness = []
        agents = []
        ga = MOP((100, 40), chrom_l=[20, 20],
                low=[-20, -20], high=[20, 20])
        n_pop = ga.initialization()  # initial first population
        for i in range(0, self.max_round):
            chrom_decoded = ga.chromosomeDecode(n_pop)
            b_f, p_f, p, a = ga.bestResult(chrom_decoded)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            population_fitness.append(p)
            agents.append(a)
            selected_ind = ga.roulette_wheel_selection(chrom_decoded)
            new_child = ga.selectInd(selected_ind, n_pop)
            new_pop = ga.mutation(new_child)
            n_pop = new_pop  # Replace the new population
        return population_best_fitness, avg_population_fitness, agents, chrom_decoded

    def plot(self, population_best_fitness, avg_population_fitness, agents, chrom_decoded):
        fig, ax = plt.subplots()
        ax.plot(avg_population_fitness, linewidth=2.0, label="avg_fitness")
        ax.plot(population_best_fitness, linewidth=2.0 ,label=["best_fitness 1","best_fitness 2","best_fitness 3","best_fitness 4","best_fitness 5"], linestyle=':')
        plt.legend(loc="upper right")
        print(f"best solution: {population_best_fitness[-1]}")
        print(
            f"best solution agents: {agents[-1]}")
        plt.show()
