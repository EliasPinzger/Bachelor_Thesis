class Class:
    """
    This class contains for a class to classify all needed values of the attributes and is used to modify the scene
    based on those values. Initially all shapes should be hidden.
    """

    def __init__(self, name, lights, background_planes):
        """
        Initializes an instance of Class.

        :param name: The name of the class
        :param lights: The lights for which the light energies should be set
        :param background_planes: A list of planes for which the background texture should be set
        """

        self.name = name
        self.lights = lights
        self.background_planes = background_planes

        self.shapes = None
        self.shape_index = -1
        self.textures = None
        self.texture_index = -1
        self.colors = None
        self.color_index = -1
        self.scales = None
        self.scale_index = -1
        self.backgrounds = None
        self.background_index = -1
        self.light_energies = None
        self.light_energy_index = -1

    def next_background(self):
        """
        Sets the next background texture. If the method is called and there is no next background the method will be
        reset.

        :return: False when there is no next background and otherwise True
        :rtype: bool
        """

        self.background_index += 1
        if self.background_index >= len(self.backgrounds):
            self.background_index = -1
            return False
        for background_plane in self.background_planes:
            background_plane.data.materials[0] = self.backgrounds[self.background_index]
        return True

    def next_shape(self):
        """
        Disables hide for the next shape and hides the last shape. If the method is called and there is  no next
        shape the method will be reset.

        :return: False when there is no next shape and otherwise True
        :rtype: bool
        """

        self.shapes[self.shape_index].hide_render = True
        self.shape_index += 1
        if self.shape_index >= len(self.shapes):
            self.shape_index = -1
            return False
        self.shapes[self.shape_index].hide_render = False
        return True

    def next_scale(self):
        """
        Sets the next scale and adjusts the position. If the method is called and there is no next scale the method
        will be reset.

        :return: False when there is no next scale and otherwise True
        :rtype: bool
        """

        self.scale_index += 1
        if self.scale_index >= len(self.scales):
            self.scale_index = -1
            return False
        new_scale = self.scales[self.scale_index]
        for shape in self.shapes:
            shape.location.z = self.shapes[self.shape_index].dimensions.z / shape.scale.z * new_scale / 2
            shape.scale = (new_scale, new_scale, new_scale)
        return True

    def next_texture(self):
        """
        Sets the next texture. If the method is called and there is no next texture the method will be reset.

        :return: False when there is no next texture and otherwise True
        :rtype: bool
        """

        self.texture_index += 1
        if self.texture_index >= len(self.textures):
            self.texture_index = -1
            return False
        for shape in self.shapes:
            shape.data.materials[0] = self.textures[self.texture_index][0]
        return True

    def next_color(self):
        """
        Sets the next color. If the method is called and there is no next color the method will be reset.

        :return: False when there is no next color and otherwise True
        :rtype: bool
        """

        self.color_index += 1
        if self.color_index >= len(self.colors):
            self.color_index = -1
            return False
        for texture in self.textures:
            texture[1].elements[0].color = self.colors[self.color_index]
        return True

    def next_lighting(self):
        """
        Sets the next lighting. If the method is called and there is no next lighting the
        method will be reset.

        :return: False when there is no next lighting and otherwise True
        :rtype: bool
        """

        self.light_energy_index += 1
        if self.light_energy_index >= len(self.light_energies):
            self.light_energy_index = -1
            return False
        for light in self.lights:
            light.data.energy = self.light_energies[self.light_energy_index]
        return True

