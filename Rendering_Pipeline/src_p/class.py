import bpy
import math as m


class Class:
    def __init__(self, name, surface_sphere, clouds_sphere, atmos_sphere, background_plane, light):
        self.name = name
        self.surface_sphere = surface_sphere
        self.clouds_sphere = clouds_sphere
        self.atmos_sphere = atmos_sphere
        self.background_plane = background_plane
        self.light = light
        self.light_axis = bpy.data.objects['Axis_Light']

        self.surface_textures = None
        self.surface_texture_index = -1
        self.clouds_textures = None
        self.clouds_texture_index = -1
        self.colors = None
        self.color_index = -1
        self.scales = None
        self.scale_index = -1
        self.light_directions = None
        self.light_direction_index = -1
        self.light_energies = None
        self.light_energy_index = -1
        self.backgrounds = None
        self.background_index = -1

    def next_scale(self):
        self.scale_index += 1
        if self.scale_index >= len(self.scales):
            self.scale_index = -1
            return False
        new_scale = self.scales[self.scale_index]
        self.surface_sphere.scale = (new_scale, new_scale, new_scale)
        self.clouds_sphere.scale = (new_scale, new_scale, new_scale)
        self.atmos_sphere.scale = (new_scale, new_scale, new_scale)
        return True

    def next_clouds_texture(self):
        self.clouds_texture_index += 1
        if self.clouds_texture_index >= len(self.clouds_textures):
            self.clouds_texture_index = -1
            return False
        self.clouds_sphere.data.materials[0] = self.clouds_textures[self.clouds_texture_index][0]
        return True

    def next_surface_textures(self):
        self.surface_texture_index += 1
        if self.surface_texture_index >= len(self.surface_textures):
            self.surface_texture_index = -1
            return False
        self.surface_sphere.data.materials[0] = self.surface_textures[self.surface_texture_index][0]
        return True

    def next_color(self):
        self.color_index += 1
        if self.color_index >= len(self.colors):
            self.color_index = -1
            return False
        for surface_texture in self.surface_textures:
            surface_texture[1].elements[0].color = self.colors[self.color_index]
        return True

    def next_background(self):
        self.background_index += 1
        if self.background_index >= len(self.backgrounds):
            self.background_index = -1
            return False
        self.background_plane.data.materials[0] = self.backgrounds[self.background_index]
        return True

    def next_lighting(self):
        self.light_energy_index += 1
        if self.light_energy_index >= len(self.light_energies):
            self.light_energy_index = -1
            return False
        self.light.data.energy = self.light_energies[self.light_energy_index]
        return True

    def next_light_direction(self):
        self.light_direction_index += 1
        if self.light_direction_index >= len(self.light_directions):
            self.light_direction_index = -1
            return False
        self.light_axis.rotation_euler.z = m.radians(self.light_directions[self.light_direction_index])
        return True
