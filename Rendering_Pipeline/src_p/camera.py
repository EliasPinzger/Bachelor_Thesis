import bpy
import math as m


class Camera:
    def __init__(self, heights, betas, gammas):
        self.camera = bpy.context.scene.camera
        self.camera.rotation_euler = (m.radians(90), m.radians(0), m.radians(0))
        self.camera.location.x = -0.2
        self.camera.location.y = -12
        self.axis = bpy.data.objects['Axis_Camera']
        self.axis.rotation_euler.y = m.radians(0)

        self.heights = heights
        self.betas = betas
        self.gammas = gammas

        self.number_of_positions = len(heights) * len(betas) * len(gammas)

        self.current_height = 0
        self.current_beta = 0
        self.current_gamma = -1

    def next_perspective(self):
        self.current_gamma += 1
        if self.current_gamma >= len(self.gammas):
            self.current_gamma = 0
            self.current_beta += 1
            if self.current_beta >= len(self.betas):
                self.current_beta = 0
                self.current_height += 1
                if self.current_height >= len(self.heights):
                    self.current_height = 0
                    self.current_gamma = -1
                    return False
        self.camera.location.z = self.heights[self.current_height]
        self.axis.rotation_euler.x = m.radians(self.betas[self.current_beta])
        self.axis.rotation_euler.z = m.radians(self.gammas[self.current_gamma])
        return True
