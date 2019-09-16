import numpy as np
import math

class Scene:
    """Scene class, represent the scene"""
    def __init__(self, scene_params):
        super(Scene, self).__init__()
        stick_width = scene_params['stick_width']
        stick_height = scene_params['stick_height']
        self.z_boundaries = []
        self.y_funcs = []
        if scene_params['shape'] == 'rectangle':
            side_length = scene_params['side_length']
            self.z_boundaries.append(stick_height - side_length/2)
            f1 = lambda z: stick_width
            self.y_funcs.append(f1)
            self.z_boundaries.append(stick_height + side_length/2)
            f2 = lambda z: side_length/2
            self.y_funcs.append(f2)

        elif scene_params['shape'] == 'circle':
            radius = scene_params['radius']
            self.z_boundaries.append(stick_height - math.sqrt(radius * radius - stick_width * stick_width))
            f1 = lambda z: stick_width
            self.y_funcs.append(f1)
            self.z_boundaries.append(stick_height + radius)
            f2 = lambda z: math.sqrt(math.pow(radius, 2) - math.pow(z - stick_height, 2))
            self.y_funcs.append(f2)

        elif scene_params['shape'] == 'triangle':
            side_length = scene_params['side_length']
            self.z_boundaries.append(stick_height - side_length / (2*math.sqrt(3)))
            f1 = lambda z: stick_width
            self.y_funcs.append(f1)
            self.z_boundaries.append(stick_height + side_length / math.sqrt(3))
            f2 = lambda z: side_length/3 + stick_height/math.sqrt(3) - z/math.sqrt(3)
            self.y_funcs.append(f2)

        else:
            raise TypeError("The scene does not support this sign shape")


    def query_object(self, x, y, z):
        # Query if there is object at point (x, y, z)
        if x == 0:
            # Query on the sign plane
            max_y = self._find_max_y_sign_plane(z)
            if abs(y) <= max_y:
                return True
            else:
                return False
        elif z == 0:
            # Query on the ground plane
            return True
        else:
            raise ValueError("Invalid query point to the scene: outside two planes")

    def _find_max_y_sign_plane(self, z):
        if z < 0:
            raise ValueError("Invalid query point to the scene: negative z")
        else:
            for i in range(len(self.z_boundaries)):
                if z < self.z_boundaries[i]:
                    max_y = self.y_funcs[i](z)
                    return max_y
            return -1



class Scene:
    """Scene class, represent the scene
    Support query into the scene to get intensity value at a point
    Support query to get intensity range in a specific shape
    """
    def __init__(self, scene_params):
        super(Scene, self).__init__()
        self.line_width = scene_params['line_width']
        self.road_width = scene_params['road_width']
        self.shade_width = scene_params['shade_width']
        self.side_line_intensity = SIDE_LINE_INTENSITY
        self.center_line_intensity = CENTER_LINE_INTENSITY
        self.road_intensity = ROAD_INTENSITY
        self.sky_intensity = SKY_INTENSITY

        self.x1 = self.line_width/2 - self.shade_width # centerline inner edge
        self.x2 = self.line_width/2 + self.shade_width # centerline outer edge
        self.x3 = self.road_width - self.line_width/2 - self.shade_width # side line left outer edge
        self.x4 = self.road_width - self.line_width/2 + self.shade_width # side line left inner edge
        self.x5 = self.road_width + self.line_width/2 - self.shade_width # side line right inner edge
        self.x6 = self.road_width + self.line_width/2 + self.shade_width # side line right outer edge
        # check the road is reasonable
        s = [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6]
        assert sorted(range(len(s)), key=lambda k: s[k]) == [0,1,2,3,4,5], "Scene dimensions not reasonable, change the paramters"
        self.critical_points = [-self.x6, -self.x5, -self.x4, -self.x3, -self.x2, -self.x1, self.x1, self.x2, self.x3, self.x4, self.x5, self.x6]

    def get_intensity_at_point(self, x, y):
        if x<0:
            x = -x # Scene is symmetric
        if x >= self.x6 or (x>=self.x2 and x<=self.x3):
            return self.road_intensity
        elif x > self.x5:
            return self._interpolate(self.x5, self.side_line_intensity, self.x6, self.road_intensity, x)
        elif x > self.x4:
            return self.side_line_intensity
        elif x > self.x3:
            return self._interpolate(self.x3, self.road_intensity, self.x4, self.side_line_intensity, x)
        elif x > self.x1:
            return self._interpolate(self.x1, self.center_line_intensity, self.x2, self.road_intensity, x)
        else:
            return self.center_line_intensity

    def get_intensity_range(self, x_min, x_max):
        # Get the range for intensity within (x_min, x_max)
        # Get the set of critical points within (x_min, x_max)
        critical_points = [x_min, x_max]
        for x in self.critical_points:
            if (x_min < x) and (x < x_max):
                critical_points.append(x)
        # Read intensity of each critical points, take max and min values
        critical_intensities = [self.get_intensity_at_point(x,0) for x in critical_points]
        return (min(critical_intensities), max(critical_intensities))

    def get_sky_intensity(self):
        return self.sky_intensity

    def _interpolate(self, x1, i1, x2, i2, x):
        return ((x2-x)*i1 + (x-x1)*i2)/(x2-x1)
