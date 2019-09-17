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


class Sensor:
    """docstring for Sensor."""
    def __init__(self, sensor_params, scene, noise_mode='none', noise_scale=0.0):
        super(Sensor, self).__init__()
        self.height = sensor_params['height']
        self.ray_num = sensor_params['ray_num'] # n-by-n ray matrix
        self.focal_length = sensor_params['focal_length']
        self.pixel_size = sensor_params['pixel_size']
        self.max_distance = sensor_params['max_distance'] # Max distance that Lidar can sense
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale * self.max_distance # noise_scale is fraction of maximum range
        self.scene = scene

    def take_measurement(self, angle, distance):
        # angle, distance with respect to center of sign, polar coordinate
        # angle is counterclockwise viewing down z axis
        # distance is w.r.t. the center of the sign, in x-y plane
        theta = np.deg2rad(angle)
        pixel_matrix = np.zeros((self.ray_num, self.ray_num))
        for i in range(self.ray_num):
            for j in range(self.ray_num):
                px, py, pz = self._compute_ray_direction_vec(i, j, theta)
                d, y, z = self._compute_intersect_sign_plane(px, py, pz, theta, distance)
                if z>=0 and self.scene.query_object(0, y, z):
                    pixel_matrix[i][j] = d
                else:
                    if i < self.ray_num/2:
                        pixel_matrix[i][j] = self.max_distance
                    else:
                        d, x, y = self._compute_intersect_ground_plane(px, py, pz, theta, distance)
                        if self.scene.query_object(x, y, 0):
                            pixel_matrix[i][j] = min(d, self.max_distance)
                        else:
                            raise ValueError("Cannot hit object in scene, unexpected")

        if self.noise_mode != 'none':
            pixel_matrix = self._add_noise(pixel_matrix)
        return pixel_matrix

    def _compute_ray_direction_vec(self, i, j, theta):
        common_term = - self.pixel_size/2 + self.ray_num*self.pixel_size/2
        px = - self.focal_length * np.cos(theta) + (common_term - j * self.pixel_size ) * np.sin(theta)
        py = - self.focal_length * np.sin(theta) - (common_term - j * self.pixel_size ) * np.cos(theta)
        pz = common_term - i * self.pixel_size
        norm_len = math.sqrt(px*px + py*py + pz*pz)
        return px/norm_len, py/norm_len, pz/norm_len

    def _compute_intersect_sign_plane(self, px, py, pz, theta, r):
        t = - r * np.cos(theta) / px
        assert t >= 0, "returned distance is not positive"
        y = r * np.sin(theta) + t * py
        z = self.height + t * pz
        return t, y, z

    def _compute_intersect_ground_plane(self, px, py, pz, theta, r):
        t = - self.height / pz
        assert t >= 0, "returned distance is not positive"
        x = r * np.cos(theta) + px * t
        y = r * np.sin(theta) + py * t
        return t, x, y

    def _add_noise(self, continuous_image):
        if self.noise_mode == 'uniform':
            noise = np.random.rand(*np.shape(continuous_image)) * (2*self.noise_scale) - self.noise_scale # Rescale to [-noise_scale, noise_scale]
            noise[continuous_image == self.max_distance] = 0
            return np.clip(continuous_image+noise, 0.0, self.max_distance)
        elif self.noise_mode == 'gaussian':
            noise = np.random.randn(*np.shape(continuous_image)) * self.noise_scale # Rescale to zero mean and std noise_scale
            noise[continuous_image == self.max_distance] = 0
            return np.clip(continuous_image+noise, 0.0, self.max_distance)
        else:
            raise Exception('Noise mode not supported')


class Viewer:
    """Viewer class, represent the camera.
    Support given a offset and angle, return an image taken
    Support given a offset and angle and range, return an image together with pixel value ranges.
    """
    def __init__(self, camera_params, scene, noise_mode='none', noise_scale=0.0):
        super(Viewer, self).__init__()
        self.height = camera_params['height']
        self.focal_length = camera_params['focal_length']
        self.pixel_num = camera_params['pixel_num'] # n-by-n image
        self.pixel_size = camera_params['pixel_size'] # edge length for single pixel
        self.scene = scene
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.n_sigma = 4 # Only for noise_mode==gaussian, where to truncate
        self.angle_to_x_axis = self._compute_angle_to_x_axis()

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

        if self.noise_mode != 'none':
            pixel_matrix = self._add_noise(pixel_matrix)
        pixel_matrix = self._quantize_image(pixel_matrix)
        return pixel_matrix

    def take_picture_with_range(self, offset, angle, delta_x, delta_phi):
        # offset plus_or_minus delta_x, angle plus_or_minus delta_phi is the small box of interest.
        # Return: an image, and a matrix of lower bound of each pixel and another of higher bound of each pixel
        image_matrix = np.zeros((self.pixel_num, self.pixel_num))
        lower_bound_matrix = np.zeros((self.pixel_num, self.pixel_num))
        upper_bound_matrix = np.zeros((self.pixel_num, self.pixel_num))
        image_to_world_transform_center = self._compute_transform(offset, angle)

        image_to_world_transform_criticals = []
        image_to_world_transform_criticals.append(self._compute_transform(offset-delta_x, angle-delta_phi))
        image_to_world_transform_criticals.append(self._compute_transform(offset+delta_x, angle-delta_phi))
        image_to_world_transform_criticals.append(self._compute_transform(offset-delta_x, angle+delta_phi))
        image_to_world_transform_criticals.append(self._compute_transform(offset+delta_x, angle+delta_phi))

        # for each pixel, get pixel center coordinate using offset and angle
        for j in range(self.pixel_num):
            critical_transforms = deepcopy(image_to_world_transform_criticals)
            if ((angle - delta_phi) < self.angle_to_x_axis[j]) and ((angle + delta_phi) > self.angle_to_x_axis[j]):
                critical_transforms.append(self._compute_transform(offset-delta_x, self.angle_to_x_axis[j]))
                critical_transforms.append(self._compute_transform(offset+delta_x, self.angle_to_x_axis[j]))
            for i in range(self.pixel_num):
                if i < self.pixel_num/2:
                    image_matrix[i][j] = self.scene.get_sky_intensity()
                    lower_bound_matrix[i][j] = image_matrix[i][j]
                    upper_bound_matrix[i][j] = image_matrix[i][j]
                else:
                    # Find intersection point of the ray with the world
                    intersection_point = self._image_to_world(i, j, image_to_world_transform_center)
                    # Query intensity value from Scene
                    image_matrix[i][j] = self.scene.get_intensity_at_point(intersection_point[0], intersection_point[1]) # i is row index, j is column index

                    # Find the critical intersection points with the world
                    critical_intersections = [self._image_to_world(i, j, transform) for transform in critical_transforms]
                    xs_critical = [p[0] for p in critical_intersections]
                    (min_val, max_val) = self.scene.get_intensity_range(min(xs_critical), max(xs_critical))
                    lower_bound_matrix[i][j] = min_val
                    upper_bound_matrix[i][j] = max_val
        if self.noise_mode != 'none':
            image_matrix = self._add_noise(image_matrix)
            lower_bound_matrix = self._extend_bound(lower_bound_matrix, 'lower')
            upper_bound_matrix = self._extend_bound(upper_bound_matrix, 'upper')
        image_matrix = self._quantize_image(image_matrix)
        lower_bound_matrix = self._quantize_image(lower_bound_matrix)
        upper_bound_matrix = self._quantize_image(upper_bound_matrix)
        return image_matrix, lower_bound_matrix, upper_bound_matrix

    def _compute_angle_to_x_axis(self):
        # For each column of pixels, compute the angle that needs for it to rotate to align with x axis. This is used for computing critical points in determining the intensity range
        angle_to_x_axis = []
        common_term = self.pixel_size/2 - self.pixel_num*self.pixel_size/2
        for j in range(self.pixel_num):
            if j < (self.pixel_num/2):
                angle_to_x_axis.append(90 + np.degrees(np.arctan((common_term + j*self.pixel_size)/self.focal_length)))
            else:
                angle_to_x_axis.append(-90 + np.degrees(np.arctan((common_term + j*self.pixel_size)/self.focal_length)))
        return angle_to_x_axis

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

    def _add_noise(self, continuous_image):
        if self.noise_mode == 'uniform':
            noise = np.random.rand(*np.shape(continuous_image)) * (2*self.noise_scale) - self.noise_scale # Rescale to [-noise_scale, noise_scale]
            return np.clip(continuous_image+noise, 0.0, 1.0)
        elif self.noise_mode == 'gaussian':
            noise = np.random.randn(*np.shape(continuous_image)) * self.noise_scale # Rescale to zero mean and std noise_scale
            return np.clip(continuous_image+noise, 0.0, 1.0)
        else:
            raise Exception('Noise mode not supported')

    def _quantize_image(self, continuous_image):
        # Transform an image in (0,1) to [0,255] discrete
        return np.clip(np.floor(continuous_image*256),0,255)

    def _extend_bound(self, continuous_image, mode):
        if self.noise_mode == 'uniform':
            if mode == 'lower':
                return np.clip(continuous_image - self.noise_scale, 0.0, 1.0)
            else:
                return np.clip(continuous_image + self.noise_scale, 0.0, 1.0)
        elif self.noise_mode == 'gaussian':
            if mode == 'lower':
                return np.clip(continuous_image - self.noise_scale * self.n_sigma, 0.0, 1.0)
            else:
                return np.clip(continuous_image + self.noise_scale * self.n_sigma, 0.0, 1.0)
        else:
            raise Exception('Noise mode not supported')
