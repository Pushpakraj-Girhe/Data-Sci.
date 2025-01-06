import numpy as np
from filterpy.kalman import KalmanFilter

class MultiSensorFusion:
    def __init__(self, num_sensors, state_dim, process_variance, measurement_variance):
        """
        Initialize the multisensor fusion system using Kalman Filters.

        Parameters:
        - num_sensors: Number of sensors contributing data.
        - state_dim: Dimensions of the state to estimate (e.g., one per sensor).
        - process_variance: Process noise covariance (system uncertainty).
        - measurement_variance: Measurement noise covariance for each sensor.
        """
        self.num_sensors = num_sensors
        self.state_dim = state_dim
        self.filters = []

        # Initialize individual Kalman Filters for each sensor
        for _ in range(num_sensors):
            kf = KalmanFilter(dim_x=state_dim, dim_z=1)  # dim_z=1 for scalar measurements
            kf.F = np.eye(state_dim)  # State transition matrix (identity for simplicity)
            kf.H = np.array([[1.]])  # Measurement function (1x1 matrix for scalar measurements)
            kf.Q = process_variance * np.eye(state_dim)  # Process noise covariance
            kf.R = np.array([[measurement_variance]])  # Measurement noise covariance (1x1 matrix)
            kf.P = np.eye(state_dim)  # Initial state covariance
            kf.x = np.zeros((state_dim, 1))  # Initial state estimate
            self.filters.append(kf)

    def preprocess_data(self, sensor_data):
        """
        Preprocess sensor data to align and normalize measurements.

        Parameters:
        - sensor_data: List of numpy arrays from sensors.

        Returns:
        - Preprocessed sensor data (aligned and normalized).
        """
        sensor_data = np.array(sensor_data, dtype=float)
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        data_min = np.min(sensor_data, axis=0)
        data_range = np.max(sensor_data, axis=0) - data_min + epsilon
        normalized_data = (sensor_data - data_min) / data_range
        return normalized_data

    def fuse(self, sensor_data):
        """
        Perform data fusion using Kalman Filters.

        Parameters:
        - sensor_data: List of numpy arrays representing sensor measurements.

        Returns:
        - Fused state estimate.
        """
        preprocessed_data = self.preprocess_data(sensor_data)

        # Apply Kalman update for each sensor's data
        for i, data in enumerate(preprocessed_data.T):  # Iterate over sensor columns
            for measurement in data:  # Process each measurement for this sensor
                self.filters[i].predict()
                # Reshape measurement to (1, 1) as required by filterpy
                measurement = np.array([[measurement]], dtype=float)
                self.filters[i].update(measurement)

        # Combine sensor estimates using Covariance Intersection
        combined_state = np.zeros((self.state_dim, 1))
        combined_covariance = np.zeros((self.state_dim, self.state_dim))

        for kf in self.filters:
            weight = np.linalg.inv(kf.P)  # Use the inverse of the covariance as weight
            combined_state += weight @ kf.x
            combined_covariance += weight

        combined_covariance = np.linalg.inv(combined_covariance)
        combined_state = combined_covariance @ combined_state

        return combined_state, combined_covariance

# Example Usage
if __name__ == "__main__":
    # Initialize multisensor fusion system
    num_sensors = 8  # Based on the data (8 sensors)
    state_dim = 1  # Single dimension per sensor
    process_variance = 0.1  # Reduced from 1.0 for more stable estimates
    measurement_variance = 0.5  # Reduced from 2.0 for more responsive updates

    fusion_system = MultiSensorFusion(num_sensors, state_dim, process_variance, measurement_variance)

    # Example sensor data (rows = samples, columns = sensors)
    sensor_data = [
        [2.26, 6.75, 2.154, 0.178, 2.2, 5.3, 0.278, 0.452],
        [3.5, 10.83, 3.365, 0.267, 3.64, 8.423, 0.588, 0.935],
        [0.796, 10.62, 0.893, 0.111, 0.968, 7.897, 0.3, 0.242],
        [4.021, 13.33, 4.255, 0.34, 4.538, 16.4, 0.842, 2.25]
    ]

    # Perform data fusion
    fused_state, fused_covariance = fusion_system.fuse(sensor_data)

    print("Fused State:", fused_state.flatten()[0])
    print("Fused Covariance:", fused_covariance[0][0])
    """
Output
    1. Fused State
        Represents the most accurate estimate of the state based on all sensor inputs.
        Calculated as a weighted average of individual sensor estimates, with more weight given to sensors with lower uncertainty.
    2. Fused Covariance
    Represents the confidence in the fused state.
    Lower covariance means higher confidence.

    """