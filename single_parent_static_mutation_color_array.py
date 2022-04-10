from PIL import Image, ImageDraw
from multiprocessing import Process, Queue

import aggdraw
import numpy as np
import random
import os
from datetime import datetime
import copy

from base_classes import BaseUtils, PROCESSING, create_dir_name, save_metadata, Point

from single_parent_static_mutation_random_color import Population, DNA

"""
    - New DNA is created from one parent
    - Mutation Rate for Colors and Points is equal and static
    - Colors are picked random from all colors existing in the original image
"""

NAME = "single_parent_static_mutation_new_fitness_color_array"
DIR = create_dir_name(NAME)

SAVE_BEST_DNA_FREQUENCY = 1000

reference, IMAGE_SIZE, colors = BaseUtils.get_image_values("../mona_lisa.jpg")
reference_r, reference_g, reference_b = reference

POINTS_AMOUNT = 50

# Tuple x, y
distance_between_points = BaseUtils.get_distance_between_points(POINTS_AMOUNT, IMAGE_SIZE)

TRIANGLE_AMOUNT = (POINTS_AMOUNT - 1) * 2 * (POINTS_AMOUNT - 1)

POINT_MUTATION = 0.005
COLOR_MUTATION = 0.00075

def new_fitness_func(ref_data, data):
    return np.maximum(np.sum(8000 * np.power(1.06, -np.abs(ref_data - data)) - 1), 0) * 0.000001

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
    def crossover(mom, mutation_propabilities):
        p_mutations = 0
        c_mutations = 0

        new_points = copy.deepcopy(mom.points)
        
        for _ in range(len(new_points)):
            if random.random() < POINT_MUTATION:
                p_mutations += 1
                new_points[random.randint(0, len(new_points) - 1)].mutate()

        new_colors = copy.deepcopy(mom.colors)

        for _ in range(len(new_colors)):
            if random.random() < COLOR_MUTATION:
                c_mutations += 1
                new_colors[random.randint(0, len(new_colors) - 1)] = tuple(utils.generate_color())

        # print("MUTATED", p_mutations, c_mutations)

        return DNAColorArray(new_points, new_colors)

class PopulationColorArray(Population):
    def __init__(self, best_dna=None, mutation_rate=POINT_MUTATION):
        super().__init__(best_dna, mutation_rate)

        if best_dna == None:
            self.population = [DNAColorArray() for _ in range(PROCESSING)]

    def generate_new_population(self, best_dna):
        new_population = []

        for _ in range(len(self.population) - 1):
            new_population.append(DNAColorArray.crossover(best_dna, self.mutation_rule))

        new_population.append(best_dna)

        self.population = new_population


if __name__ == "__main__":
    population = PopulationColorArray()
    epoch = 0
    best_dna = None
    all_time_best = 0

    os.mkdir(DIR)

    save_metadata(DIR, {
        "mutation": {
            "point": POINT_MUTATION,
            "color": COLOR_MUTATION
        },
        "points_amount": {
            "x": POINTS_AMOUNT,
            "y": POINTS_AMOUNT
        },
        "description": [
            "Single parent",
            "New fitness function",
            "Static mutation Rate",
            "Pick color random from color array containing all colors of image"
        ]
    })

    while True:
        population = PopulationColorArray(best_dna)
        best_dna, max, min = population.run_epoch()

        if max > all_time_best + 50:
            all_time_best = max
            best_dna.draw().save(f"{DIR}/{epoch}_{max}.png")

        print(f"{str(epoch).ljust(10)} | {str(round(max, 5)).ljust(20)} | {str(round(min, 5)).ljust(20)}")

        with open(DIR + "/log.csv", "a") as file:
            file.write(f"{epoch}, {max}, {min}\n")

        if epoch % SAVE_BEST_DNA_FREQUENCY == 0:
            best_dna.draw().save(f"{DIR}/{epoch}.png")

        epoch += 1
        