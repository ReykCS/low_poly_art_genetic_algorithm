from PIL import Image, ImageDraw
from multiprocessing import Process, Queue

import aggdraw
import numpy as np
import random
import os
from datetime import datetime
import copy

from base_classes import BaseUtils, create_dir_name, save_metadata, Point

from single_parent_static_mutation_color_array import PopulationColorArray, DNAColorArray, utils

"""
    - New DNA is created from one parent
    - Mutation Rate for Colors and Points is equal and static
    - Colors are picked random from all colors existing in the original image
"""

NAME = "single_parent_dynamic_mutation_only_colors_color_array"
DIR = create_dir_name(NAME)

SAVE_BEST_DNA_FREQUENCY = 50

reference, IMAGE_SIZE, colors = BaseUtils.get_image_values("../mona_lisa.jpg")
reference_r, reference_g, reference_b = reference

POINTS_AMOUNT = 20

# Tuple x, y
distance_between_points = BaseUtils.get_distance_between_points(POINTS_AMOUNT, IMAGE_SIZE)

TRIANGLE_AMOUNT = (POINTS_AMOUNT - 1) * 2 * (POINTS_AMOUNT - 1)

def new_crossover(mom, mutation_props):
    new_points = copy.deepcopy(mom.points)
    new_colors = copy.deepcopy(mom.colors)

    for i in range(len(new_colors)):
        if random.random() < mutation_props:
            new_colors[random.randint(0, len(new_colors) - 1)] = tuple(utils.generate_color())

    return DNAColorArray(new_points, new_colors)

DNAColorArray.crossover = new_crossover

if __name__ == "__main__":
    population = PopulationColorArray(mutation_rate=random.random())
    epoch = 0
    best_dna = None

    os.mkdir(DIR)

    save_metadata(DIR, {
        "mutation_rate": "dynamic",
        "points_amount": {
            "x": POINTS_AMOUNT,
            "y": POINTS_AMOUNT
        },
        "description": [
            "Single parent",
            "Dynamic mutation Rate",
            "Max amount of mutations per generation",
            "Pick color random from color array containing all colors of image"
        ]
    })

    while True:
        population = PopulationColorArray(best_dna, random.random())
        best_dna, max, min = population.run_epoch()

        print(f"{str(epoch).ljust(10)} | {str(round(max, 5)).ljust(20)} | {str(round(min, 5)).ljust(20)}")

        with open(DIR + "/log.csv", "a") as file:
            file.write(f"{epoch}, {max}, {min}, {population.mutation_rule}\n")

        if epoch % SAVE_BEST_DNA_FREQUENCY == 0:
            best_dna.draw().save(f"{DIR}/{epoch}.png")

        epoch += 1
        