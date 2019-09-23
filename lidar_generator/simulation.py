import numpy as np
import math
from copy import deepcopy

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
        self.n_sigma = 5 # Only for noise_mode==gaussian, where to truncate
        self.scene = scene
        self.angles_to_focal = self._compute_angle_to_focal()


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

        return self._scale(pixel_matrix)


    def take_measurement_with_range(self, distance, angle, delta_x, delta_angle):
        # distance plus_or_minus delta_x, angle plus_or_minus delta_angle is the small box of interest.
        # Return: an image, and a matrix of lower bound of each pixel and another of uppper bound of each pixel
        lower_bound_d_matrix = np.zeros((self.ray_num, self.ray_num))
        upper_bound_d_matrix = np.zeros((self.ray_num, self.ray_num))

        critical_points = []
        critical_points.append((distance - delta_x, np.deg2rad(angle - delta_angle)))
        critical_points.append((distance + delta_x, np.deg2rad(angle + delta_angle)))
        critical_points.append((distance - delta_x, np.deg2rad(angle + delta_angle)))
        critical_points.append((distance + delta_x, np.deg2rad(angle - delta_angle)))

        # For each pixel, get (hit_plane, distance) at each of the critical points (4 or 5), then decide distance range
        for j in range(self.ray_num):
            critical_pts = deepcopy(critical_points)
            if ((angle - delta_angle) < self.angles_to_focal[j]) and ((angle + delta_angle) > self.angles_to_focal[j]):
                critical_pts.append((distance - delta_x, np.deg2rad(self.angles_to_focal[j])))
            for i in range(self.ray_num):
                critical_distances = [self._get_intersection_d(i, j, distance, theta) for (distance, theta) in critical_pts]
                on_sign_planes = [p[0] for p in critical_distances]
                ds = [p[1] for p in critical_distances]
                upper_bound_d_matrix[i][j] = max(ds)
                if any(on_sign_planes):
                    additional_distances = [self._get_intersection_d_sign_plane(i, j, distance, theta) for (idx, (distance, theta)) in enumerate(critical_pts) if not on_sign_planes[idx]]
                    ds.extend(additional_distances)
                    lower_bound_d_matrix[i][j] = min(ds)
                else:
                    lower_bound_d_matrix[i][j] = min(ds)

        if self.noise_mode != 'none':
            lower_bound_d_matrix = self._extend_bound(lower_bound_d_matrix, 'lower')
            upper_bound_d_matrix = self._extend_bound(upper_bound_d_matrix, 'upper')

        lower_bound_matrix = self._scale(upper_bound_d_matrix)
        upper_bound_matrix = self._scale(lower_bound_d_matrix)

        return lower_bound_matrix, upper_bound_matrix

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

    def _scale(self, distance_image):
        return 1.0 - distance_image / self.max_distance

    def _compute_angle_to_focal(self):
        # For each column of pixels, compute the angle to the focal direction (rotating around z axis). This is used for computing critical points in determining the distance range
        angles_to_focal = []
        common_term = self.pixel_size/2 - self.ray_num*self.pixel_size/2
        for j in range(self.ray_num):
            angles_to_focal.append(np.degrees(np.arctan((common_term + j*self.pixel_size)/self.focal_length)))
        return angles_to_focal

    def _get_intersection_d(self, i, j, distance, theta):
        px, py, pz = self._compute_ray_direction_vec(i, j, theta)
        d, y, z = self._compute_intersect_sign_plane(px, py, pz, theta, distance)
        if z>=0 and self.scene.query_object(0, y, z):
            on_plane = True
        else:
            on_plane = False
            if i < self.ray_num/2:
                d = self.max_distance
            else:
                d, x, y = self._compute_intersect_ground_plane(px, py, pz, theta, distance)
                d = min(d, self.max_distance)

        return (on_plane, d)

    def _get_intersection_d_sign_plane(self, i, j, distance, theta):
        px, py, pz = self._compute_ray_direction_vec(i, j, theta)
        d, y, z = self._compute_intersect_sign_plane(px, py, pz, theta, distance)
        return d

    def _extend_bound(self, continuous_image, mode):
        if self.noise_mode == 'uniform':
            extension = np.zeros(np.shape(continuous_image)) + self.noise_scale
            extension[continuous_image == self.max_distance] = 0
            if mode == 'lower':
                return np.clip(continuous_image - extension, 0.0, self.max_distance)
            else:
                return np.clip(continuous_image + extension, 0.0, self.max_distance)
        elif self.noise_mode == 'gaussian':
            extension = np.zeros(np.shape(continuous_image)) + self.noise_scale * self.n_sigma
            extension[continuous_image == self.max_distance] = 0
            if mode == 'lower':
                return np.clip(continuous_image - extension, 0.0, self.max_distance)
            else:
                return np.clip(continuous_image + extension, 0.0, self.max_distance)
        else:
            raise Exception('Noise mode not supported')
