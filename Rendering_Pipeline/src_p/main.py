import os
import shutil
from enum import Enum
from importlib import util

import bpy
import numpy as np


def import_file(full_name, path):
    """
    Imports a python file.

    :param full_name: The name of the file
    :param path: The path to the file
    :return: The module
    """

    spec = util.spec_from_file_location(full_name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_path = 'C:/Users/elias/Desktop/bachelorthesis/Rendering_Pipeline/src_p'

class_module = import_file('class', os.path.join(source_path, 'class.py'))
camera_module = import_file('camera', os.path.join(source_path, 'camera.py'))


class Attribute(Enum):
    """
    An Enum of all the available attributes which can be used as trace and ndea.
    """

    SCALE = 1
    COLOR = 2
    SURFACE_TEXTURE = 3
    CLOUDS_TEXTURE = 4
    LIGHT_DIRECTION = 5
    LIGHTING = 6
    BACKGROUND = 7


class Render:
    """
    The class Render contains everything necessary to create a synthetic dataset with traces. Before the method
    render is used the method initialize_classification_objects should be called. If the usage of a GPU is desired the
    enable_gpus method must be invoked before the rendering process starts. Alternatively the GPUs can be activated in
    Blender.
    """

    def __init__(self, filepath, class_names, traces, ndea, camera):
        """
        Initializes an instance of Render. The traces and ndea should not have the same elements otherwise
        they are used as ndea.

        :param filepath: The path where the dataset should be saved
        :param class_names: A list of the class names which define the amount of classes
        :param traces: A list of the attributes wanted as traces
        :param ndea: A list of the attributes wanted as non dataset extending attributes
        :param camera: The camera of the scene
        """

        self.filepath = filepath
        self.class_names = class_names
        self.traces = traces
        self.ndea = ndea
        self.functions = []
        self.ndea_functions = []

        self.scene = bpy.context.scene
        self.camera = camera

        self.classes = []

        self.scene.render.engine = 'CYCLES'
        self.scene.render.resolution_percentage = 100

        self.scene.cycles.samples = 50
        self.scene.render.resolution_x = 224
        self.scene.render.resolution_y = 224

        self.image_counter = 0
        self.current_path = None
        self.current_class = None
        self.ndea_function_index = 0
        self.ndea_start = 0
        self.skip = 0

        self.total_images = 0

    def initialize_classification_objects(self, colors, scales, light_energies, light_directions):
        """
        Initializes the classification objects accordingly to the given traces. The backgrounds and surface textures
        are materials and retrieved from Blender. The materials are used as surface texture if they have 'surface' in
        their name and used as background if the name contains 'background'. Only materials used for the surface must
        contain a color ramp node.

        :param colors: A list of colors (color is a tuple of the form: (R, G, B, Alpha))
        :param scales: A list of scales (scale is a tuple of the form: (x, y, z))
        :param light_energies: A list of light energies
        :param light_directions: A list of light directions
        :return: False if no Traces were given and user don't want to continue and True otherwise
        :rtype: bool
        """

        if len(self.traces) <= 0 and len(self.class_names) > 1:
            user_input = input('\nNo traces given all classes will be the same [Stop(S), Continue(C)]: ')
            if user_input != 'C':
                return False

        if Attribute.BACKGROUND in self.ndea:
            self.ndea_functions.append(class_module.Class.next_background)
        else:
            self.functions.append(class_module.Class.next_background)
        if Attribute.SCALE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_scale)
        else:
            self.functions.append(class_module.Class.next_scale)
        if Attribute.SURFACE_TEXTURE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_surface_textures)
        else:
            self.functions.append(class_module.Class.next_surface_textures)
        if Attribute.COLOR in self.ndea:
            self.ndea_functions.append(class_module.Class.next_color)
        else:
            self.functions.append(class_module.Class.next_color)
        if Attribute.LIGHTING in self.ndea:
            self.ndea_functions.append(class_module.Class.next_lighting)
        else:
            self.functions.append(class_module.Class.next_lighting)
        if Attribute.CLOUDS_TEXTURE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_clouds_texture)
        else:
            self.functions.append(class_module.Class.next_clouds_texture)
        if Attribute.LIGHT_DIRECTION in self.ndea:
            self.ndea_functions.append(class_module.Class.next_light_direction)
        else:
            self.functions.append(class_module.Class.next_light_direction)

        light = bpy.data.objects['Light']
        surface_sphere = bpy.data.collections['Planet'].all_objects['Surface']
        clouds_sphere = bpy.data.collections['Planet'].all_objects['Clouds']
        atmos_sphere = bpy.data.collections['Planet'].all_objects['Atmos']
        background_plane = bpy.data.objects['Background_Plane']

        for class_name in self.class_names:
            self.classes.append(class_module.Class(
                class_name, surface_sphere, clouds_sphere, atmos_sphere, background_plane, light))

        surface_textures = []
        clouds_textures = []
        backgrounds = []

        for material in bpy.data.materials:
            color_ramp = None
            for node in material.node_tree.nodes:
                if node.name == 'planet_color':
                    color_ramp = node.color_ramp
            if 'surface' in material.name:
                surface_textures.append((material, color_ramp))
            elif 'clouds' in material.name:
                clouds_textures.append((material, color_ramp))
            elif 'background' in material.name:
                backgrounds.append(material)

        self.__distribute_traces__(surface_textures, clouds_textures, backgrounds, colors, scales,
                                   light_energies, light_directions)
        self.__set_total_images__()
        return True

    def __distribute_traces__(self, surface_textures, clouds_textures, backgrounds, colors, scales,
                              light_energies, light_directions):
        """
        Initializes the classification objects accordingly to the given traces.

        :param surface_textures: A list of all the surface textures
        :param clouds_textures: A list of all the clouds textures
        :param backgrounds: A list of all the background textures
        :param colors: A list of all the colors
        :param scales: A list of all the scales
        :param backgrounds: A list of all the backgrounds
        :param light_energies: A list of all the light energies
        :param light_energies: A list of all the light directions
         """

        for index, c_class in enumerate(self.classes):
            if Attribute.LIGHTING in self.traces:
                c_class.light_energies = [light_energies[index]]
            else:
                c_class.light_energies = light_energies
            if Attribute.SURFACE_TEXTURE in self.traces:
                c_class.surface_textures = [surface_textures[index]]
            else:
                c_class.surface_textures = surface_textures
            if Attribute.CLOUDS_TEXTURE in self.traces:
                c_class.clouds_textures = [clouds_textures[index]]
            else:
                c_class.clouds_textures = clouds_textures
            if Attribute.COLOR in self.traces:
                c_class.colors = [colors[index]]
            else:
                c_class.colors = colors
            if Attribute.SCALE in self.traces:
                c_class.scales = [scales[index]]
            else:
                c_class.scales = scales
            if Attribute.LIGHT_DIRECTION in self.traces:
                c_class.light_directions = [light_directions[index]]
            else:
                c_class.light_directions = light_directions
            if Attribute.BACKGROUND in self.traces:
                c_class.backgrounds = [backgrounds[index]]
            else:
                c_class.backgrounds = backgrounds

    def __set_total_images__(self):
        """
        Sets total_images to the total amount of images the dataset will contain. Must be called after the classes
         are fully initialized.
        """

        total_images = 1
        for c_class in self.classes:
            if Attribute.BACKGROUND not in self.ndea:
                total_images *= len(c_class.backgrounds)
            if Attribute.SCALE not in self.ndea:
                total_images *= len(c_class.scales)
            if Attribute.SURFACE_TEXTURE not in self.ndea:
                total_images *= len(c_class.surface_textures)
            if Attribute.COLOR not in self.ndea:
                total_images *= len(c_class.colors)
            if Attribute.LIGHTING not in self.ndea:
                total_images *= len(c_class.light_energies)
            if Attribute.LIGHT_DIRECTION not in self.ndea:
                total_images *= len(c_class.light_directions)
            if Attribute.CLOUDS_TEXTURE not in self.ndea:
                total_images *= len(c_class.clouds_textures)
            self.total_images += total_images * self.camera.number_of_positions
            total_images = 1

    def __delete_contents__(self):
        """
        Deletes all contents at filepath.

        :return: True when deletion was successful and False otherwise
        :rtype: bool
        """

        for filename in os.listdir(self.filepath):
            file_path = os.path.join(self.filepath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                return False
        return True

    def __create_folder_structure__(self):
        """
        Creates the needed folder structure at the filepath location. If the needed structure is already existent the
        user can resume a unfinished dataset or delete all contents at the filepath location.

        :return: True if the data structure was successfully created and the dataset is not already complete and False
            is returned otherwise
        :rtype: bool
        """

        if os.path.exists(self.filepath) and os.path.isdir(self.filepath):
            contents = os.listdir(self.filepath)
            if contents:
                if self.class_names.sort() == contents.sort():
                    user_input = input(
                        '\nFolder ' + self.filepath + ' is not empty. But order structure was found '
                                                      '[Stop(S), Use existing Content(C), Delete Content(D)]:  ')
                    if user_input == 'C':
                        num_files = sum([len(files) for _, _, files in os.walk(self.filepath)])
                        if num_files >= self.total_images:
                            print("\nDataset is already complete")
                            return False
                        print('\nProgram will start render at image: ' + str(num_files + 1) + '\\' + str(
                            self.total_images) + '.')
                        self.skip = num_files
                        return True
                else:
                    user_input = input(
                        '\nFolder ' + self.filepath + ' is not empty [Stop(S), Delete Content(D)]:  ')
                if user_input == 'D':
                    if not self.__delete_contents__():
                        print('\nDeletion was not successful.')
                        return False
                else:
                    return False
        else:
            os.mkdir(self.filepath)
        for class_name in self.class_names:
            os.mkdir(os.path.join(self.filepath, class_name))
        return True

    def enable_gpus(self, device_type):
        """
        Enables all GPUs of a specific device type and sets tile size.

        :param device_type: One of the following: ('CUDA', 'OPTIX', 'HIP', 'ONEAPI')
        """

        self.scene.cycles.device = 'GPU'
        self.scene.cycles.tile_size = 256
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = device_type
        bpy.context.preferences.addons['cycles'].preferences.get_devices()

    def __ndea_function_caller__(self, i):
        """
        Creates every combination of ndea values in the order of the ndea_function_index.

        :param i: offset for the current ndea_function_index

        :return: True if the next ndea_function_index is needed and False otherwise
        :rtype: bool
        """

        if i >= len(self.ndea_functions):
            return True

        else:
            if not self.ndea_functions[(self.ndea_function_index + i) % len(self.ndea_functions)](
                    self.current_class):
                self.ndea_functions[(self.ndea_function_index + i) % len(self.ndea_functions)](self.current_class)
                return self.__ndea_function_caller__(i + 1)
            return False

    def __render_class__(self, functions):
        """
        Renders all remaining images of the current class and saves the images as the current value of image_counter.

        :param functions: a list of all DEA functions and trace functions
        """

        if len(functions) > 0:
            while functions[0](self.current_class):
                self.__render_class__(functions[1:])
        else:
            while self.camera.next_perspective():
                if len(self.ndea_functions) > 0:
                    if not self.ndea_functions[self.ndea_function_index](self.current_class):
                        self.ndea_functions[self.ndea_function_index](self.current_class)
                        if len(self.ndea_functions) > 1:
                            if self.__ndea_function_caller__(1):
                                self.ndea_function_index = (self.ndea_function_index + 1) % len(self.ndea_functions)

                if self.skip <= 0:
                    self.scene.render.filepath = os.path.join(self.current_path, str(self.image_counter))
                    bpy.ops.render.render(write_still=True)
                    self.image_counter += 1
                else:
                    self.skip -= 1

    def render(self):
        """
        Prepares the needed folder structure and renders all remaining images for the dataset.

        :return: True if the dataset was successfully crated and False otherwise
        :rtype: bool
        """

        if not self.__create_folder_structure__():
            return False
        user_input = input(
            '\nImages to render: ' + str(self.total_images - self.skip) + '. [Stop(S), Continue(C)]  ')
        if user_input != 'C':
            return False

        self.image_counter = self.skip
        for c_class in self.classes:
            # initialize all ndea which are not initialized indirectly
            for ndea_function in self.ndea_functions[1:]:
                ndea_function(c_class)
            self.current_class = c_class
            self.current_path = os.path.join(self.filepath, c_class.name)
            self.__render_class__(self.functions)
            self.ndea_function_index = 0
        return True


def load_random_attribute_values(filepath):
    rav = np.load(filepath)
    return rav['heights'], rav['betas'], rav['gammas'], rav['scales'], rav['light_energies'], rav['light_directions']


if __name__ == '__main__':
    heights, betas, gammas, scales, light_energies, light_directions = load_random_attribute_values(
        'C:/Users/elias/Desktop/bachelorthesis/Rendering_Pipeline/resources_p/random_attributes_values_no_clouds.npz')

    camera = camera_module.Camera(heights=heights,
                                  betas=betas,
                                  gammas=gammas)

    render = Render('G:/Datasets/Planet/LightDirection_No_Clouds',
                    class_names=['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'],
                    traces=[Attribute.LIGHT_DIRECTION],
                    ndea=[Attribute.COLOR, Attribute.SURFACE_TEXTURE],
                    camera=camera)

    if render.initialize_classification_objects(
            colors=[(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1)],
            scales=scales,
            light_energies=light_energies,
            light_directions=light_directions,
    ):
        # Remove if no GPU is available else set the correct device_type
        render.enable_gpus('OPTIX')

        if render.render():
            print('\nDataset finished')
        else:
            print('\nCreation of dataset was terminated.')
    else:
        print('\nInitialization was not successful.')
