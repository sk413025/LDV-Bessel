import numpy as np
from scipy.optimize import minimize

class IntensityReconstructor:
    def __init__(self, params):
        self.params = params
        self.history = {'obj_values': []}

    def define_space_time_grid(self):
        x = np.linspace(self.params['x_range'][0], self.params['x_range'][1], self.params['x_points'])
        y = np.linspace(self.params['y_range'][0], self.params['y_range'][1], self.params['y_points'])
        t = np.linspace(self.params['t_range'][0], self.params['t_range'][1], self.params['t_points'])
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        return x, y, t, X, Y, T

    def define_measurement_region(self, X, Y, r=None, x0=None, y0=None):
        r = r if r is not None else self.params['r']
        x0 = x0 if x0 is not None else self.params['x0']
        y0 = y0 if y0 is not None else self.params['y0']
        distance = np.sqrt((X[:, :, 0] - x0)**2 + (Y[:, :, 0] - y0)**2)
        measurement_mask = distance <= r
        measurement_indices = np.where(measurement_mask)
        return measurement_mask, measurement_indices

    def calculate_phase_modulation(self, X, Y, T):
        kx = self.params['k_magnitude'] * np.cos(self.params['theta_incidence'])
        ky = self.params['k_magnitude'] * np.sin(self.params['theta_incidence'])
        omega = 2 * np.pi * self.params['f']
        phase = self.params['beta'] * np.cos(kx * X + ky * Y - omega * T)
        return phase

    def generate_interference_intensity(self, phase):
        A_ref = 1.0
        A_meas = 1.0
        intensity = A_ref**2 + A_meas**2 + 2 * A_ref * A_meas * np.cos(phase)
        return intensity

    def get_intensity_measurements(self, intensity, measurement_indices):
        y = intensity[measurement_indices].flatten()
        return y

    def construct_dictionary_intensity(self, X, Y, T, theta_list, measurement_indices):
        n_pixels = len(measurement_indices[0])
        n_times = len(T[0,0,:])
        n_directions = len(theta_list)
        Psi = np.zeros((n_pixels * n_times, n_directions))

        for idx, theta in enumerate(theta_list):
            kx = self.params['k_magnitude'] * np.cos(theta)
            ky = self.params['k_magnitude'] * np.sin(theta)
            phase = self.params['beta'] * np.cos(kx * X[measurement_indices] + ky * Y[measurement_indices] - 2 * np.pi * self.params['f'] * T[measurement_indices])
            intensity = 2 + 2 * np.cos(phase)
            Psi[:, idx] = intensity.flatten()
        return Psi / np.sqrt(n_pixels * n_times)

    def objective_intensity(self, s, Psi, y):
        residual = y - Psi @ s
        obj_value = 0.5 * np.sum(residual**2) + 1e-3 * np.sum(np.abs(s))
        self.history['obj_values'].append(obj_value)
        return obj_value

    def reconstruct_intensity_field(self, Psi, s, shape, measurement_indices):
        intensity = np.zeros(shape)
        intensity[measurement_indices] = (Psi @ s).reshape((len(measurement_indices[0]), -1))
        return intensity

    def reconstruct(self, y_measurements, theta_list=None):
        if theta_list is None:
            theta_list = np.linspace(self.params['theta_range'][0], self.params['theta_range'][1], self.params['theta_points'])
        
        x, y, t, X, Y, T = self.define_space_time_grid()
        measurement_mask, measurement_indices = self.define_measurement_region(X, Y)
        Psi = self.construct_dictionary_intensity(X, Y, T, theta_list, measurement_indices)

        s0 = np.zeros(len(theta_list))
        result = minimize(lambda s: self.objective_intensity(s, Psi, y_measurements),
                          s0,
                          method='L-BFGS-B',
                          options={'maxiter': 1000})

        s_est = result.x
        reconstructed_intensity = self.reconstruct_intensity_field(Psi, s_est, X.shape, measurement_indices)
        reconstruction_error = np.mean(np.abs(reconstructed_intensity[measurement_indices] - y_measurements.reshape(reconstructed_intensity[measurement_indices].shape)))
        reconstructed_theta = theta_list[np.argmax(np.abs(s_est))]

        return reconstructed_intensity, s_est, reconstruction_error, reconstructed_theta