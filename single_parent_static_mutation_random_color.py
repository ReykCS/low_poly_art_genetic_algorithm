from PIL import Image, ImageDraw
from multiprocessing import Process, Queue

import aggdraw
import numpy as np
import random
import os
from datetime import datetime
import copy

from base_classes import BaseUtils, PROCESSING, create_dir_name, Point

"""
    - New DNA is created from one parent
    - Mutation Rate for Colors and Points is equal and static
    - Colors are picked random in range 0x333333 - 0xCCCCCC with varying R, G and B values
"""

NAME = "single_parent_static_mutation_random_color"
DIR = create_dir_name(NAME)

SAVE_BEST_DNA_FREQUENCY = 100

reference, IMAGE_SIZE, colors = BaseUtils.get_image_values("../mona_lisa.jpg")
reference_r, reference_g, reference_b = reference

POINTS_AMOUNT = 50

# Tuple x, y
distance_between_points = BaseUtils.get_distance_between_points(POINTS_AMOUNT, IMAGE_SIZE)

TRIANGE_AMOUNT = (POINTS_AMOUNT - 1) * 2 * (POINTS_AMOUNT - 1)

MUTATION_RATE = 0.05

Utils = BaseUtils(colors, POINTS_AMOUNT)

class DNA:
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

            self.points.append(p)

        for i in range(TRIANGE_AMOUNT):
            self.colors.append(tuple(Utils.generate_color()))

    def draw(self):
        image = Image.new("RGB", IMAGE_SIZE, 0xFFFFFF)
        draw = aggdraw.Draw(image)

        draw_triangle = lambda i, j, x, color_index: draw.polygon(
            np.array(
                [self.points[i].coord(), self.points[j].coord(), self.points[x].coord()]
            ).flatten(), 
            aggdraw.Brush(self.colors[color_index])
        )

        for i in range(len(self.points)):
            x_index = self.points[i].x_index
            y_index = self.points[i].y_index

            if x_index <= 0 or y_index <= 0:
                continue

            color_index = Utils.get_color_index(x_index, y_index)

            draw_triangle(i, i - 1, i - POINTS_AMOUNT - 1, color_index)
            draw_triangle(i, i - POINTS_AMOUNT, i - POINTS_AMOUNT - 1, color_index + 1)

        draw.flush()

        return image

    def get_fitness(self, id=0, queue=None):
        image = self.draw()

        img_arr = np.asarray(image).flatten()

        data_r, data_g, data_b = img_arr.reshape((-1, 3)).transpose()

        r_fitness = BaseUtils.fitness_func(reference_r, data_r)
        g_fitness = BaseUtils.fitness_func(reference_g, data_g)
        b_fitness = BaseUtils.fitness_func(reference_b, data_b)

        self.fitness = r_fitness + g_fitness + b_fitness

        if queue:
            queue.put((self.fitness, id))

        return self.fitness

    @staticmethod
    def crossover(mom, mutation_propabilities):
        new_points = copy.deepcopy(mom.points)
        
        for i in range(len(new_points)):
            if random.random() < mutation_propabilities:
                new_points[i].mutate()

        new_colors = copy.deepcopy(mom.colors)

        for i in range(len(new_colors)):
            if random.random() < mutation_propabilities:
                new_colors[i] = tuple(Utils.generate_color())

        return DNA(new_points, new_colors)

class Population:
    def __init__(self, best_dna=None, mutation_rate=MUTATION_RATE):
        self.fitness = [0] * PROCESSING
        self.best_dna = None
        self.mutation_rule = mutation_rate
        self.population = [None] * PROCESSING 

        if best_dna:
            self.best_dna = best_dna
            self.generate_new_population(best_dna)
            return

        self.population = [DNA() for _ in range(PROCESSING)]

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

        self.best_dna = best

        # self.generate_new_population(self.best_dna)

        return self.best_dna, max, min

    def generate_new_population(self, best_dna):
        new_population = []

        for _ in range(len(self.population)):
            new_population.append(DNA.crossover(best_dna, self.mutation_rule))

        self.population = new_population


if __name__ == "__main__":
    population = Population()
    epoch = 0
    best_dna = None

    os.mkdir(DIR)

    while True:
        population = Population(best_dna)
        best_dna, max, min = population.run_epoch()

        print(f"{str(epoch).ljust(10)} | {str(round(max, 5)).ljust(20)} | {str(round(min, 5)).ljust(20)}")

        with open(DIR + "/log.csv", "a") as file:
            file.write(f"{epoch}, {max}, {min}\n")

        if epoch % SAVE_BEST_DNA_FREQUENCY == 0:
            best_dna.draw().save(f"{DIR}/{epoch}.png")

        epoch += 1
