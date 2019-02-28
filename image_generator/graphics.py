import numpy as np

SIDE_LINE_INTENSITY = 1.0
CENTER_LINE_INTENSITY = 0.7
ROAD_INTENSITY = 0.3
SKY_INTENSITY = 0.0

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

    def get_sky_intensity(self):
        return self.sky_intensity

    def _interpolate(self, x1, i1, x2, i2, x):
        return ((x2-x)*i1 + (x-x1)*i2)/(x2-x1)



class Viewer:
    """Viewer class, represent the camera.
    Support given a offset and angle, return an image taken
    Support given a offset and angle and range, return an image together with pixel value ranges.
    """
    def __init__(self, camera_params, scene):
        super(Viewer, self).__init__()
        self.height = camera_params['height']
        self.focal_length = camera_params['focal_length']
        self.pixel_num = camera_params['pixel_num'] # n-by-n image
        self.pixel_size = camera_params['pixel_size'] # edge length for single pixel
        self.scene = scene

    def take_picture(self, offset, angle):
        # offset is x for the camera, angle is positive for counterclockwise viewing down z axis
        pixel_matrix = np.zeros((self.pixel_num, self.pixel_num))
        image_to_world_transform = self._compute_transform(offset, angle)
        # for each pixel, get pixel center coordinate using offset and angle
        for i in range(self.pixel_num):
            for j in range(self.pixel_num):
                if i < self.pixel_num/2:
                    pixel_matrix[i][j] = self.scene.get_sky_intensity()
                else:
                    # Find intersection point of the ray with the world
                    intersection_point = self._image_to_world(i, j, image_to_world_transform)
                    # Query intensity value from Scene
                    pixel_matrix[i][j] = self.scene.get_intensity_at_point(intersection_point[0], intersection_point[1]) # i is row index, j is column index

        pixel_matrix = self._quantize_image(pixel_matrix)
        return pixel_matrix

    def _compute_transform(self, offset, angle):
        # Compute the transform matrix from image coordinate to world coordinate
        theta = np.deg2rad(angle)
        common_term = self.pixel_size/2 - self.pixel_num*self.pixel_size/2
        # Pixel coordinate to coordinate centered at focal point
        pixel_to_focal = np.array([[0,self.pixel_size,common_term],[0,0,self.focal_length],[-self.pixel_size,0,-common_term],[0,0,1]])
        rotation = np.array([[np.cos(theta), -np.sin(theta),0,0], [np.sin(theta), np.cos(theta),0,0], [0,0,1,0], [0,0,0,1]]) # Rotation in the focal coordinate
        projection = np.array([[-self.height,0,0,0], [0,-self.height,0,0], [0,0,-self.height,0], [0,0,1,0]]) # Project each pixel to the ground plane
        focal_to_world = np.eye(4)
        focal_to_world[0,3] = offset
        focal_to_world[2,3] = self.height
        transform_mtrx = np.matmul(focal_to_world, np.matmul(projection, np.matmul(rotation, pixel_to_focal)))
        return transform_mtrx

    def _image_to_world(self, i, j, transform):
        # Transform the image coordinate to world coordinate
        image_coord = np.array([i,j,1])
        world_homo = np.matmul(transform, image_coord)
        world_coord = world_homo[0:3]/world_homo[3]
        return world_coord

    def _quantize_image(self, continuous_image):
        # Transform an image in (0,1) to [0,255] discrete
        return np.clip(np.floor(continuous_image*256),0,255)
