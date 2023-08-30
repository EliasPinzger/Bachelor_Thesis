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


source_path = 'C:/Users/elias/Desktop/bachelorthesis/Rendering_Pipeline/src_g'

class_module = import_file('class', os.path.join(source_path, 'class.py'))
camera_module = import_file('camera', os.path.join(source_path, 'camera.py'))


class Attribute(Enum):
    """
    An Enum of all the available attributes which can be used as trace and ndea.
    """

    BACKGROUND = 1
    SHAPE = 2
    SCALE = 3
    TEXTURE = 4
    COLOR = 5
    LIGHTING = 6


class Render:
    """
    The class Render contains everything necessary to create a synthetic dataset with traces. Before the method
    render is used the method initialize_classes should be called. If the usage of GPUs is desired the
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
        self.shapes = None

        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = 100

        self.scene.render.resolution_x = 224
        self.scene.render.resolution_y = 224

        self.image_counter = 0
        self.current_path = None
        self.current_class = None
        self.ndea_function_index = 0
        self.ndea_start = 0
        self.skip = 0

        self.total_images = 0

    def initialize_classes(self, colors, scales, light_energies):
        """
        Initializes the classification objects accordingly to the given traces. The backgrounds and surface textures
        are materials and retrieved from Blender. The materials are used as surface texture if they have 'surface' in
        their name and used as background if the name contains 'background'. Only materials used for the surface must
        contain a Color Ramp Node.

        :param colors: A list of colors (color is a tuple of the form: (R, G, B, Alpha))
        :param scales: A list of scales (scale is a tuple of the form: (x, y, z))
        :param light_energies: A list of light energies
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
        if Attribute.SHAPE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_shape)
        else:
            self.functions.append(class_module.Class.next_shape)
        if Attribute.SCALE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_scale)
        else:
            self.functions.append(class_module.Class.next_scale)
        if Attribute.TEXTURE in self.ndea:
            self.ndea_functions.append(class_module.Class.next_texture)
        else:
            self.functions.append(class_module.Class.next_texture)
        if Attribute.COLOR in self.ndea:
            self.ndea_functions.append(class_module.Class.next_color)
        else:
            self.functions.append(class_module.Class.next_color)
        if Attribute.LIGHTING in self.ndea:
            self.ndea_functions.append(class_module.Class.next_lighting)
        else:
            self.functions.append(class_module.Class.next_lighting)

        lights = bpy.data.collections['Lights'].all_objects
        background_planes = bpy.data.collections['Background Planes'].all_objects
        for class_name in self.class_names:
            self.classes.append(class_module.Class(class_name, lights, background_planes))

        self.shapes = np.array(bpy.data.collections['Shapes'].all_objects)
        backgrounds = []
        textures = []

        for material in bpy.data.materials:
            color_ramp = None
            for node in material.node_tree.nodes:
                if node.name == 'Color':
                    color_ramp = node.color_ramp
            if 'Texture' in material.name:
                textures.append((material, color_ramp))
            elif 'Background' in material.name:
                backgrounds.append(material)
        self.__distribute_traces__(textures, colors, scales, backgrounds, light_energies)
        self.__set_total_images__()
        return True

    def __distribute_traces__(self, textures, colors, scales, backgrounds, light_energies):
        """
        Initializes the classes accordingly to the given traces.

        :param textures: A list of all the textures
        :param colors: A list of all the colors
        :param scales: A list of all the scales
        :param backgrounds: A list of all the background textures
        :param light_energies: A list of all the light energies
         """

        for index, c_class in enumerate(self.classes):
            if Attribute.BACKGROUND in self.traces:
                c_class.backgrounds = [backgrounds[index]]
            else:
                c_class.backgrounds = backgrounds
            if Attribute.SHAPE in self.traces:
                c_class.shapes = [self.shapes[index]]
            else:
                c_class.shapes = self.shapes
            if Attribute.SCALE in self.traces:
                c_class.scales = [scales[index]]
            else:
                c_class.scales = scales
            if Attribute.TEXTURE in self.traces:
                c_class.textures = [textures[index]]
            else:
                c_class.textures = textures
            if Attribute.COLOR in self.traces:
                c_class.colors = [colors[index]]
            else:
                c_class.colors = colors
            if Attribute.LIGHTING in self.traces:
                c_class.light_energies = [light_energies[index]]
            else:
                c_class.light_energies = light_energies

    def __set_total_images__(self):
        """
        Sets total_images to the total amount of images the dataset will contain. Must be called after the classes
        are fully initialized.
        """

        total_images = 1
        for c_class in self.classes:
            if Attribute.BACKGROUND not in self.ndea:
                total_images *= len(c_class.backgrounds)
            if Attribute.SHAPE not in self.ndea:
                total_images *= len(c_class.shapes)
            if Attribute.SCALE not in self.ndea:
                total_images *= len(c_class.scales)
            if Attribute.TEXTURE not in self.ndea:
                total_images *= len(c_class.textures)
            if Attribute.COLOR not in self.ndea:
                total_images *= len(c_class.colors)
            if Attribute.LIGHTING not in self.ndea:
                total_images *= len(c_class.light_energies)
            self.total_images += total_images * self.camera.number_of_positions
            total_images = 1

    def __delete_contents__(self):
        """
        Deletes all contents at the filepath.

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
        user can resume an unfinished dataset or delete all contents at the filepath location.

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

    def __hide_all_shapes__(self):
        """
        Hides indirectly all shapes of all classes.
        """

        for shape in self.shapes:
            shape.hide_render = True

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
            if not self.ndea_functions[(self.ndea_function_index + i) % len(self.ndea_functions)](self.current_class):
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
                    if self.__ndea_function_caller__(0):
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
            self.__hide_all_shapes__()
            # initialize all ndea which are not initialized indirectly
            for ndea_function in self.ndea_functions[1:]:
                ndea_function(c_class)
            self.current_class = c_class
            self.current_path = os.path.join(self.filepath, c_class.name)
            self.__render_class__(self.functions)
            self.ndea_function_index = 0
        return True


def load_random_attribute_values(filepath):
    """
    Loads the values of the file at filepath.

    :param filepath: Path of the file which should be loaded

    :return: a tuple of heights, betas, gammas, scales and light energies
    :rtype: tuple of numbers
    """

    rav = np.load(filepath)
    return rav['heights'], rav['betas'], rav['gammas'], rav['scales'], rav['light_energies']


if __name__ == '__main__':
    heights, betas, gammas, scales, light_energies = load_random_attribute_values(
        'C:/Users/elias/Desktop/bachelorthesis/Rendering_Pipeline/resources_g/random_attributes_values.npz')

    camera = camera_module.Camera(heights=heights,
                                  betas=betas,
                                  gammas=gammas)

    render = Render('G:/Datasets/Geometric2/Shape_Texture',
                    class_names=['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'],
                    traces=[Attribute.SHAPE, Attribute.TEXTURE],
                    ndea=[Attribute.COLOR],
                    camera=camera)

    if render.initialize_classes(
            colors=[(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1)],
            scales=scales,
            light_energies=light_energies):

        # Remove if no GPU is available else set the correct device_type
        render.enable_gpus('OPTIX')

        if render.render():
            print('\nDataset finished')
        else:
            print('\nCreation of dataset was not finished.')
    else:
        print('\nInitialization was not successful.')
