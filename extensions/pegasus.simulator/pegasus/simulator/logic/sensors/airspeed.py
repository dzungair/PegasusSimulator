"""
| File: airspeed.py
| Author: Mohammadreza Mousaei
| License: [License Information]
| Description: Simulates an airspeed sensor.
"""

__all__ = ["Airspeed"]

import numpy as np
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor

class Airspeed(Sensor):
    """The class that implements the Airspeed sensor. This class inherits the base class Sensor.
    """
    def __init__(self, config={}):
        """Initialize the Airspeed class

        Args:
            config (dict): A dictionary that contains all the parameters for configuring the Airspeed sensor.
                It can be empty or only have some of the parameters used by the Airspeed sensor.
        """
        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="Airspeed", update_rate=config.get("update_rate", 50.0))

        # Airspeed noise constants
        self._airspeed_bias = 0.0
        airspeed_config = config.get("airspeed", {})
        self._airspeed_noise_density = airspeed_config.get("noise_density", 0.01)

        # Save the current state measured by the Airspeed sensor
        self._state = {
            "airspeed": 0.0,
        }

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        """Method that implements the logic of an Airspeed sensor. In this method, we simulate the airspeed measurement
        by adding noise to the true airspeed value.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """
        # Airspeed terms
        sigma_as_d = 1.0 / np.sqrt(dt) * self._airspeed_noise_density

        # Simulate airspeed noise processes and add them to the true airspeed.
        airspeed = state.airspeed + sigma_as_d * np.random.randn() + self._airspeed_bias

        # Add the airspeed value to the dictionary and return it
        self._state = {
            "airspeed": airspeed,
        }

        return self._state