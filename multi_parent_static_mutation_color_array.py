from PIL import Image, ImageDraw
from multiprocessing import Process, Queue

import aggdraw
import numpy as np
import random
import os
from datetime import datetime
import copy

from base_classes import BaseUtils, PROCESSING, create_dir_name, save_metadata, Point

from single_parent_static_mutation_random_color import DNA

"""
    - New DNA is created from one parent
    - Mutation Rate for Colors and Points is equal and static
    - Colors are picked random from all colors existing in the original image
"""

NAME = "multi_parent_static_mutation_color_array"
DIR = create_dir_name(NAME)

SAVE_BEST_DNA_FREQUENCY = 1000

reference, IMAGE_SIZE, colors = BaseUtils.get_image_values("../mona_lisa.jpg")
reference_r, reference_g, reference_b = reference

POINTS_AMOUNT = 20

# Tuple x, y
distance_between_points = BaseUtils.get_distance_between_points(POINTS_AMOUNT, IMAGE_SIZE)

TRIANGLE_AMOUNT = (POINTS_AMOUNT - 1) * 2 * (POINTS_AMOUNT - 1)

MUTATION_RATE = 0.005

def new_fitness_func(ref_data, data):
    return np.sum(np.maximum(8000 * np.power(1.06, -np.abs(ref_data - data)) - 1, 0)) * 0.000001

BaseUtils.fitness_func = new_fitness_func

class Utils(BaseUtils):
    def __init__(self, colors, points_amount):
        super().__init__(colors, points_amount)

    def generate_color(self):
        return random.choice(colors)

utils = Utils(colors, POINTS_AMOUNT)

class DNAColorArray(DNA):
    def __init__(self, dna=None, color=None):
        self.fitness = 0
        self.points = []
        self.colors = []

        if dna and color:
            self.points = dna
            self.colors = color
            return

        for i in range(POINTS_AMOUNT * POINTS_AMOUNT):
            p = Point(i, POINTS_AMOUNT, distance_between_points)
            # p.mutate()

            self.points.append(p)

        for i in range(TRIANGLE_AMOUNT):
            self.colors.append(tuple(utils.generate_color()))

    @staticmethod
    def crossover(mom, dad, mutation_propabilities):
        new_points = [None] * len(mom.points)
        
        for i in range(len(mom.points)):
            if random.random() < 0.5:
                new_points[i] = mom.points[i]
            else:
                new_points[i] = dad.points[i]
            
            if random.random() < mutation_propabilities:
                new_points[i].mutate()
                
        new_colors = [None] * len(mom.colors)

        for i in range(len(new_colors)):
            if random.random() < 0.5:
                new_colors[i] = mom.colors[i]
            else:
                new_colors[i] = dad.colors[i]
            
            if random.random() < mutation_propabilities:
                new_colors[i] = tuple(utils.generate_color())

        return DNAColorArray(new_points, new_colors)

class Population:
    def __init__(self, mutation_rate=MUTATION_RATE):
        self.fitness = [0] * (PROCESSING + 1)
        self.best_dna = None
        self.best_fitness = 0
        self.mutation_rule = mutation_rate
        self.population = [None] * PROCESSING 

        self.population = [DNAColorArray() for _ in range(PROCESSING)]
    
    def run_epoch(self):
        max, min = 0, np.Infinity
        best = None
        processes = [None] * PROCESSING
        queues = [Queue()] * PROCESSING

        for i in range(len(self.population)):
            processes[i] = Process(target=self.population[i].get_fitness, args=(i, queues[i],))
            processes[i].start()

        for i in range(len(self.population)):
            fitness, id = queues[i].get()

            self.fitness[id] = fitness

            if fitness > max:
                max = fitness
                best = self.population[id]
            if fitness < min:
                min = fitness

        for process in processes:
            process.join()

        if max > self.best_fitness:
            self.best_dna = best
            self.best_fitness = max
            self.fitness[PROCESSING] = max

        self.max = max
        self.min = min

        self.generate_new_population()

        return self.best_dna, self.best_fitness, min

    def create_gene_pool(self, max, min, exclude=-1):
        normalized_fitness_index = []

        if max == min:
            min -= 0.001

        for fitness_index in range(len(self.fitness)):
            if fitness_index == exclude:
                continue

            amount = 1 + ((1000 - 1) / (max - min)) * (self.fitness[fitness_index] - min)

            for _ in range(int(amount)):
                normalized_fitness_index.append(fitness_index)

        return normalized_fitness_index

    def generate_new_population(self):
        new_population = []
        gene_pool = self.create_gene_pool(self.best_fitness, self.min)

        self.population.append(self.best_dna)

        for _ in range(PROCESSING):
            mom_index = random.choice(gene_pool)
            dad_index = random.choice(self.create_gene_pool(self.best_fitness, self.min, mom_index))

            new_population.append(DNAColorArray.crossover(self.population[mom_index], self.population[dad_index], self.mutation_rule))

        self.population = new_population


if __name__ == "__main__":
    population = Population()
    epoch = 0
    best_dna = None

    all_time_max = 0

    os.mkdir(DIR)

    save_metadata(DIR, {
        "mutation_rate": MUTATION_RATE,
        "points_amount": {
            "x": POINTS_AMOUNT,
            "y": POINTS_AMOUNT
        },
        "description": [
            "Two parents",
            "Static mutation Rate",
            "Pick color random from color array containing all colors of image"
        ]
    })

    with open(DIR + "/log.csv", "a") as file:
        file.write(f"epoch, max, min\n")

    while True:
        best_dna, max, min = population.run_epoch()

        if max > all_time_max + 25:
            all_time_max = max
            best_dna.draw().save(f"{DIR}/{epoch}_{max}.png")

        print(f"{str(epoch).ljust(10)} | {str(round(max, 5)).ljust(20)} | {str(round(min, 5)).ljust(20)}")

        with open(DIR + "/log.csv", "a") as file:
            file.write(f"{epoch}, {max}, {min}\n")

        if epoch % SAVE_BEST_DNA_FREQUENCY == 0:
            best_dna.draw().save(f"{DIR}/{epoch}.png")

        epoch += 1
        