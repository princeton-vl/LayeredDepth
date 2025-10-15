import math

import bpy
import mathutils
import numpy as np
from numpy.random import normal as N
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3 - math.sqrt(5))  # Golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # Radius at y
        theta = phi * i  # Calculate angle
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points


class DiscoLightFactory(AssetFactory):
    def __init__(self, factory_seed):
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params = {
                "Wattage": U(100, 2000),
                "Radius": U(0.02, 0.03),
                "numbers": np.random.choice([RI(50, 100), RI(800, 2000)]),
                "variance": U(0.1, 0.3),
            }

    def create_placeholder(self, **_):
        cube = butil.spawn_cube(size=2)
        cube.scale = (self.params["Radius"],) * 3
        butil.apply_transform(cube)
        return cube

    def create_asset(self, **_) -> bpy.types.Object:
        bpy.ops.object.empty_add(type="PLAIN_AXES")
        parent = bpy.context.object

        directions = fibonacci_sphere(self.params["numbers"])
        lights = []

        base_r = U(0.05, 0.5)
        base_g = U(0.05, 0.5)
        base_b = U(0.05, 0.5)
        variance = self.params["variance"]
        for direction in directions:
            # Create a new spotlight
            bpy.ops.object.light_add(type="SPOT")
            light = bpy.context.object
            light.data.energy = self.params["Wattage"]
            light.data.color = (
                N(base_r, variance),
                N(base_g, variance),
                N(base_b, variance),
            )
            light.data.spot_size = (
                N(360 / self.params["numbers"] / 10, 2) * math.pi / 180
            )

            # Set the light's position and rotation
            light.rotation_euler = (
                mathutils.Vector(direction).to_track_quat("Z", "Y").to_euler()
            )
            light.parent = parent

        return parent
