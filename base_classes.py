from PIL import Image, ImageDraw
from multiprocessing import Process, Queue

import aggdraw
import numpy as np
import random
import os
from datetime import datetime
import copy
import json
import math

PROCESSING = 8
ISOLATION_CYCLES = 1

GRADIENT_THRESHHOLD = 75

NOW = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

# np.sum(np.abs(ref_data - data) / 255)

create_dir_name = lambda name: "./" + name + "_" + NOW

def save_metadata(dir, metadata):
    with open(dir + "/metadata.json", "w") as file:
        json.dump(metadata, file)

class BaseUtils:
    def __init__(self, colors, points_amount):
        self.colors = colors
        self.points_amount = points_amount

    def generate_color(self):
        # return list(random.choice(colors)) + [generate_color_entry()] * 4
        return [BaseUtils.generate_color_entry() for i in range(3)]

    def get_color_index(self, x, y):
        return (x - 1) * 2 + (y - 1) * (self.points_amount - 1) * 2

    @staticmethod
    def generate_color_entry():
        return random.randint(0x33, 0xcc)

    @staticmethod
    def fitness_func(ref_data, data):
        return np.sum(np.maximum(50 * np.power(1.02, -np.abs(ref_data - data)), 0)) * 0.000001

    @staticmethod
    def get_image_values(image_path):
        reference = Image.open(image_path)
        image_size = reference.size
        reference_data = reference.getdata()

        colors = list(set(reference_data))

        return np.array(reference_data).transpose(), image_size, colors

    @staticmethod
    def get_distance_between_points(points_amount, image_size):
        return (
            image_size[0] / (points_amount - 1), 
            image_size[1] / (points_amount - 1)
        )

    @staticmethod
    def reduce_color_amount(colors):
        colors.sort(key=lambda c: c[2] + c[1] * 1000 + c[0] * 1000000)
        
        clusters = []
        last_cluster_index = 0
        last_color_split = colors[0]

        for i in range(len(colors) - 1):
            from_color = last_color_split #colors[i]
            to_color = colors[i + 1]

            gradient = math.sqrt((from_color[0] - to_color[0]) ** 2 + (from_color[1] - to_color[1]) ** 2 + (from_color[2] - to_color[2]) ** 2)

            if gradient > GRADIENT_THRESHHOLD:
                clusters.append(np.array(colors[last_cluster_index:i]))
                last_cluster_index = i
                last_color_split = to_color

        color_values = []

        print(len(clusters))

        for c in clusters:
            mean_color = list(map(int, list(np.mean(np.array(c), axis=0))))

            color_values.append(mean_color)

        return color_values

class Point:
    def __init__(self, index, points_amount, distance_between_points):
        self.index = index
        self.points_amount = points_amount
        self.distance_between_points = distance_between_points

        self.x_index = index % self.points_amount
        self.y_index = int(index / self.points_amount)

        self.base_pos_x = self.x_index * self.distance_between_points[0]
        self.base_pos_y = self.y_index * self.distance_between_points[1]

        self.distance_between_points = [0.8 * i for i in self.distance_between_points]

        self.pos_x = self.base_pos_x
        self.pos_y = self.base_pos_y

    def mutate(self):
        if not self.x_index == self.points_amount - 1 and not self.x_index == 0:
            self.pos_x = (
                self.base_pos_x 
                - (self.distance_between_points[0] / 2) 
                + random.uniform(0, self.distance_between_points[0])
            )

        if not self.y_index == self.points_amount - 1 and not self.y_index == 0:
            self.pos_y = (
                self.base_pos_y 
                - (self.distance_between_points[1] / 2) 
                + random.uniform(0, self.distance_between_points[1])    
            )

    def coord(self):
        return (self.pos_x, self.pos_y)