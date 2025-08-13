from typing import Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from citylearn.building import DynamicsBuilding
from citylearn.dynamics import Dynamics
from citylearn.energy_model import HeatPump

from .model import TransformerDynamicsModel
from .definitions import TARGET_NAMES

class TransformerDynamics(Dynamics, nn.Module):
    """
    Clean Transformer dynamics class inheriting directly from base Dynamics class.
    """
    
    def __init__(self, filepath: Union[Path, str] = None):
        Dynamics.__init__(self)
        torch.nn.Module.__init__(self)
        
        if filepath is None:
            raise ValueError("filepath parameter is required")
            
        # Load checkpoint
        ckpt = torch.load(filepath, map_location="cpu")
        
        # Model setup
        arch = ckpt["arch"]
        self.model = TransformerDynamicsModel(**arch)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        # IO configuration from checkpoint
        self.input_observation_names = ckpt["input_observation_names"]
        self.lookback = int(ckpt["lookback"])
        self.n_features = len(self.input_observation_names)
        
        # Normalization parameters (critical - from training data)
        self.input_normalization_minimum = torch.tensor(ckpt["input_normalization_minimum"], dtype=torch.float32)
        self.input_normalization_maximum = torch.tensor(ckpt["input_normalization_maximum"], dtype=torch.float32)
        
        # Target scaling parameters
        self.target_names = TARGET_NAMES
        self.indoor_idx = self.target_names.index("indoor_dry_bulb_temperature")
        self.y_min = torch.tensor(ckpt["target_minimum"], dtype=torch.float32)
        self.y_max = torch.tensor(ckpt["target_maximum"], dtype=torch.float32)
        
        # Data buffer for sliding window
        self._model_input = np.zeros((self.n_features, self.lookback + 1))

    def forward(self, observations) -> float:
        """
        Main prediction method called by CityLearn.
        Now properly uses the observations parameter.
        """
        if isinstance(observations, torch.Tensor) and observations.dim() == 3:
            x = observations[0].to(dtype=torch.float32)

        # Prepare model input tensor (lookback window, exclude current timestep)
        x = torch.tensor(
            self._model_input[:, :-1].T,  # Shape: (lookback, n_features)
            dtype=torch.float32
        )

        # Normalize the input data    
        x_norm = (x - self.input_normalization_minimum) / (self.input_normalization_maximum - self.input_normalization_minimum + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)  # Keep in [0,1] range

        # Transformer prediction
        with torch.no_grad():
            model_input = x_norm.unsqueeze(0)  # Add batch dimension: [1, lookback, features]
            y_all_norm = self.model(model_input).squeeze(0)  # Remove batch dimension

        # Extract temperature prediction and denormalize
        if y_all_norm.dim() > 0:
            y_temp_norm = y_all_norm[self.indoor_idx]
        else:
            y_temp_norm = y_all_norm

        # Clamp prediction to [0,1] before denormalizing
        y_temp_norm = torch.clamp(y_temp_norm, 0.0, 1.0)

        # Denormalize
        y_min_val = self.y_min[self.indoor_idx].item() if isinstance(self.y_min, torch.Tensor) else self.y_min[self.indoor_idx]
        y_max_val = self.y_max[self.indoor_idx].item() if isinstance(self.y_max, torch.Tensor) else self.y_max[self.indoor_idx]
        
        y_temp_actual = y_min_val + y_temp_norm * (y_max_val - y_min_val)
        
        return float(y_temp_actual)

    
    def reset(self):
        """Reset the dynamics model state for new episode."""
        self._model_input = np.zeros((self.n_features, self.lookback + 1))
        self._debug_once = True
    
    def terminate(self):
        """Clean up resources when simulation ends."""
        pass


class TransformerDynamicsBuilding(DynamicsBuilding):
    """Class for building with transformer temperature dynamics model.

    Parameters
    ----------
    *args: Any
        Positional arguments in :py:class:`citylearn.building.Building`.
    dynamics: TransformerDynamics
        Indoor dry-bulb temperature transformer dynamics model.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize :py:class:`citylearn.building.Building` super class.
    """
    def __init__(self,  *args, dynamics: TransformerDynamics, **kwargs):
        super().__init__(*args, dynamics=dynamics, **kwargs)
        self.dynamics: TransformerDynamics

    @DynamicsBuilding.simulate_dynamics.getter
    def simulate_dynamics(self) -> bool:
        # Check if we have enough data to make predictions
        # After lookback steps, we should have sufficient history
        has_data = (self.time_step >= self.dynamics.lookback)
        return super().simulate_dynamics and has_data

    def update_indoor_dry_bulb_temperature(self):
        """Update indoor temperature using transformer dynamics model."""
        
        # Prepare input tensor
        model_input_tensor = torch.tensor(self.get_dynamics_input().T)
        model_input_tensor = model_input_tensor[np.newaxis, :, :]
        
        # Call transformer (no hidden state needed)
        indoor_dry_bulb_temperature = self.dynamics(model_input_tensor.float())
        
        # Update temperature directly (already denormalized by transformer)
        self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature

    def get_dynamics_input(self) -> np.ndarray:
        """Get the input array for the next dynamic prediction."""
        model_input = []
        for i, k in enumerate(self.dynamics.input_observation_names):
            if k == 'indoor_dry_bulb_temperature':
                # indoor temperature values are t = (t - lookback - 1): t = (t - 1)
                #  i.e. use samples from previous time step to current time step
                model_input.append(self.dynamics._model_input[i][:-1])
            else:
                # other values are t = (t - lookback): t = (t)
                #  i.e. use samples from previous time step to current time step
                model_input.append(self.dynamics._model_input[i][1:])

        model_input = np.array(model_input, dtype='float32')

        return model_input
    
    def _update_dynamics_input(self):
        """Updates and returns the input time series for the dynamics prediction model.

        Updates the model input with the input variables for the current time step. 
        The variables in the input will have length of lookback + 1.
        """
        # Get relevant observations for the current time step
        observations = self.observations(include_all=True, normalize=False, periodic_normalization=True)

        # Ensure _model_input is a numpy array with correct shape
        if not isinstance(self.dynamics._model_input, np.ndarray):
            self.dynamics._model_input = np.zeros((self.dynamics.n_features, self.dynamics.lookback + 1))

        # Shift left and add new normalized observations
        self.dynamics._model_input[:, :-1] = self.dynamics._model_input[:, 1:]

        # Add current observations
        for i, obs_name in enumerate(self.dynamics.input_observation_names):
            raw_value = observations[obs_name]
            self.dynamics._model_input[i, -1] = raw_value

    def update_cooling_demand(self, action: float):
        """Cooling-only, demand-driven. No devices required."""
        EPS = 1e-6
        # 1) sanitize action to [0, 1]
        try:
            a = float(action)
        except Exception:
            a = 0.0
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

        # 2) optional: only allow cooling if above setpoint (+ deadband)
        deadband = getattr(self, "cooling_deadband", 0.2)  # °C
        try:
            Tin = float(self.energy_simulation.indoor_dry_bulb_temperature[self.time_step])
            Tsp = float(self.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.time_step])
        except Exception:
            Tin, Tsp = None, None

        allow_cooling = True
        if Tin is not None and Tsp is not None:
            allow_cooling = Tin > (Tsp + deadband * 0.5)

        # 3) map action to demand directly via a fixed capacity (kW)
        #    (no device object, no weather curves)
        max_cooling_power_kw = getattr(self, "max_cooling_power_kw", 5.0)  # pick a sensible value
        demand = float(a * max_cooling_power_kw) if allow_cooling and a > EPS else 0.0

        # 4) write demand and derive hvac_mode from demand (binary: 1=cool, 0=off)
        self.energy_simulation.cooling_demand[self.time_step] = demand
        self.energy_simulation.hvac_mode[self.time_step] = 1 if demand > EPS else 0


    def update_heating_demand(self, action: float):
        """Set heating demand from the heating device at the current time step.

        Compatible with Transformer-based dynamics: does NOT gate on self.simulate_dynamics.
        Demand is produced whenever the device/action is active and hvac_mode indicates heating.
        """
        # sanitize action
        try:
            a = float(action)
        except Exception:
            a = 0.0
        a = min(max(a, 0.0), 1.0)

        if ('heating_device' in self.active_actions or 'cooling_or_heating_device' in self.active_actions):
            # hvac_mode: 2=heating, 3=both (heating allowed)
            if self.energy_simulation.hvac_mode[self.time_step] in [2, 3]:
                electric_power = a * self.heating_device.nominal_power

                # if HeatPump, pass outdoor temp + heating=True; otherwise only max_electric_power
                try:
                    t_out = self.weather.outdoor_dry_bulb_temperature[self.time_step]
                except Exception:
                    t_out = None

                if isinstance(self.heating_device, HeatPump):
                    demand = self.heating_device.get_max_output_power(
                        t_out,
                        heating=True,
                        max_electric_power=electric_power
                    )
                else:
                    demand = self.heating_device.get_max_output_power(
                        max_electric_power=electric_power
                    )
            else:
                demand = 0.0

            self.energy_simulation.heating_demand[self.time_step] = float(max(0.0, demand))
        else:
            self.energy_simulation.heating_demand[self.time_step] = 0.0
