import numpy as np
import gymnasium as gym
from gymnasium import spaces
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch


class CityLearnSB3Wrapper(gym.Env):
    """
    Wrapper to make CityLearn compatible with Stable Baselines3
    """
    
    def __init__(self, env: CityLearnEnv):
        super().__init__()
        self.env = env
        
        # Define action and observation spaces
        self.num_buildings = len(env.buildings)
        
        # Flatten action space - SB3 works better with flattened spaces
        action_lows = []
        action_highs = []
        
        for building in env.buildings:
            action_space = building.action_space
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                # Handle Box action space
                action_lows.extend(action_space.low.flatten())
                action_highs.extend(action_space.high.flatten())
            else:
                # Fallback for other action space types
                action_lows.append(-1.0)
                action_highs.append(1.0)
        
        
        # Create flattened action space
        self.action_space = spaces.Box(
            low=np.array(action_lows, dtype=np.float32), 
            high=np.array(action_highs, dtype=np.float32), 
            dtype=np.float32
        )
        
        # Get observation space dimensions by getting sample observations
        sample_observations = []
        for building in env.buildings:
            # Get the actual observation dimension
            obs = building.observations()
            sample_observations.append(len(obs))
        
        total_obs_dim = sum(sample_observations)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # Store action dimensions for unflattening
        self.building_action_dims = []
        for building in env.buildings:
            if hasattr(building.action_space, 'shape'):
                self.building_action_dims.append(building.action_space.shape[0])
            else:
                self.building_action_dims.append(1)
        
        self.building_obs_dims = sample_observations
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        observations = self.env.reset()
        flat_obs = self._flatten_observations(observations)
        
        # SB3 expects (observation, info) tuple
        return flat_obs, {}
    
    def step(self, actions):
        # SB3 hands us a flat vector
        actions = np.asarray(actions, dtype=np.float32).ravel()
        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Convert for CityLearn
        if getattr(self.env, "central_agent", False):
            # One agent controlling all buildings: wrap once
            env_actions = [actions.tolist()]
        else:
            # Multi-agent style: split per building
            env_actions = self._unflatten_actions(actions)

        # Step raw CityLearn
        observations, rewards, terminated, truncated, info = self.env.step(env_actions)

        # Compute an "original" scalar reward (avg across buildings if list)
        raw_reward = (sum(rewards)/len(rewards)) if isinstance(rewards, list) else float(rewards)

        # Scale/clip for stability (your choice)
        scaled_reward = float(np.clip(raw_reward / 100.0, -10.0, 10.0))

        # Flatten obs for SB3
        flat_obs = self._flatten_observations(observations)

        # Attach raw reward for your callback prints
        if isinstance(info, dict):
            info.setdefault("original_reward", raw_reward)
        elif isinstance(info, list) and info:
            info[0] = {**info[0], "original_reward": raw_reward}

        done = bool(terminated or truncated)
        return flat_obs, scaled_reward, done, False, info
    
    def _flatten_observations(self, observations):
        """Flatten nested observation structure with normalization"""
        if isinstance(observations[0], list):
            # Multiple buildings
            flat_obs = []
            for obs in observations:
                flat_obs.extend(obs)
        else:
            # Single building or already flattened
            flat_obs = observations
        
        # Convert to numpy and handle any NaN/inf values
        flat_obs = np.array(flat_obs, dtype=np.float32)
        flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Simple normalization to help with training stability
        flat_obs = np.clip(flat_obs, -1e6, 1e6)
        
        return flat_obs
    
    def _unflatten_actions(self, flat_actions):
        """Convert flat actions back to CityLearn format"""
        unflat_actions = []
        action_idx = 0
        
        for building_idx, action_dim in enumerate(self.building_action_dims):
            building_actions = flat_actions[action_idx:action_idx + action_dim]
            unflat_actions.append(building_actions.tolist())
            action_idx += action_dim
            
        return unflat_actions


class SAC:
    """
    SAC Agent using Stable Baselines3 for CityLearn
    Compatible with existing CityLearn RLC interface
    """
    
    def __init__(self, env, **kwargs):
        self.env = env
        
        # Wrap CityLearn environment for SB3
        self.wrapped_env = CityLearnSB3Wrapper(env)
        self.vec_env = DummyVecEnv([lambda: self.wrapped_env])
        
        # SAC hyperparameters - tuned for CityLearn
        sac_kwargs = {
            'learning_rate': kwargs.get('learning_rate', 1e-4),  # Lower LR for stability
            'buffer_size': kwargs.get('buffer_size', 100000),
            'learning_starts': kwargs.get('learning_starts', 500),  # Start learning sooner
            'batch_size': kwargs.get('batch_size', 128),  # Smaller batch for limited data
            'tau': kwargs.get('tau', 0.01),  # Faster target network updates
            'gamma': kwargs.get('gamma', 0.95),  # Lower discount for shorter episodes
            'train_freq': kwargs.get('train_freq', (1, 'step')),  # Train every step
            'gradient_steps': kwargs.get('gradient_steps', 1),
            'target_update_interval': kwargs.get('target_update_interval', 1),
            'target_entropy': kwargs.get('target_entropy', 'auto'),
            'ent_coef': kwargs.get('ent_coef', 0.1),  # Fixed entropy coefficient
            'use_sde': kwargs.get('use_sde', False),
            'policy_kwargs': kwargs.get('policy_kwargs', dict(
                net_arch=[128, 128],  # Smaller networks
                activation_fn=torch.nn.ReLU,
                normalize_images=False
            )),
            'verbose': kwargs.get('verbose', 0),
            'device': kwargs.get('device', 'auto'),
            'tensorboard_log': kwargs.get('tensorboard_log', None),
        }
        
        # Initialize SAC agent
        self.model = SB3_SAC('MlpPolicy', self.vec_env, **sac_kwargs)
        
        # Tracking variables to match RLC interface
        self.time_step = 0
        self.actions = None
        self.episode = 0
        
    def learn(self, episodes=10):
        """
        Train the SAC agent for specified number of episodes
        Compatible with existing main() function
        """
        # Calculate total timesteps from episodes
        # Estimate timesteps per episode (use a reasonable default)
        timesteps_per_episode = getattr(self.env, 'episode_time_steps', 8760)
        total_timesteps = episodes * timesteps_per_episode
        
        print(f"Training SAC agent for {episodes} episodes ({total_timesteps} timesteps)...")
        
        # Create a simple progress callback with better debugging
        class ProgressCallback(BaseCallback):
            def __init__(self, total_episodes, timesteps_per_episode):
                super().__init__()
                self.total_episodes = total_episodes
                self.timesteps_per_episode = timesteps_per_episode
                self.episode_count = 0
                self.episode_rewards = []
                self.current_episode_reward = 0
                self.step_count = 0
                self.raw_rewards = []  # Track unscaled rewards
                
            def _on_step(self):
                # Track rewards (these are already scaled by our wrapper)
                reward = self.locals.get('rewards', [0])[0]
                self.current_episode_reward += reward
                self.step_count += 1
                
                # Get info for original reward if available
                infos = self.locals.get('infos', [{}])
                if infos and len(infos) > 0 and 'original_reward' in infos[0]:
                    self.raw_rewards.append(infos[0]['original_reward'])
                
                # Check if episode ended
                if self.locals.get('dones', [False])[0]:
                    self.episode_count += 1
                    self.episode_rewards.append(self.current_episode_reward)
                    
                    if self.episode_count % max(1, self.total_episodes // 10) == 0 or self.episode_count <= 3:
                        avg_reward = np.mean(self.episode_rewards[-5:])  # Last 5 episodes
                        recent_raw = np.mean(self.raw_rewards[-100:]) if self.raw_rewards else "N/A"
                        print(f"Episode {self.episode_count}/{self.total_episodes}")
                        print(f"  Scaled Reward: {self.current_episode_reward:.2f}")
                        print(f"  Avg Last 5: {avg_reward:.2f}")
                        print(f"  Recent Raw Reward Avg: {recent_raw}")
                        print(f"  Steps: {self.step_count}")
                    
                    self.current_episode_reward = 0
                    self.step_count = 0
                
                return True
        
        callback = ProgressCallback(episodes, timesteps_per_episode)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Use our custom progress tracking
        )
        
        print(f"Training completed!")
        
    def predict(self, observations, deterministic=False):
        """
        Get actions from the trained policy
        Compatible with existing CityLearn RLC interface
        """
        # Convert to SB3 format
        flat_obs = self.wrapped_env._flatten_observations(observations)
        
        # Get action from model
        action, _ = self.model.predict(flat_obs, deterministic=deterministic)
        
        # Convert back to CityLearn format
        unflat_actions = self.wrapped_env._unflatten_actions(action)
        
        self.actions = unflat_actions
        self.time_step += 1
        
        return unflat_actions
    
    def save(self, path):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model"""
        self.model = SB3_SAC.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
    
    # Compatibility methods for RLC interface
    def update(self, observations, actions, reward, next_observations, terminated, truncated):
        """
        Compatibility method for CityLearn's RLC interface
        This is handled internally by SB3 during training
        """
        pass
    
    def next_time_step(self):
        """Compatibility method"""
        pass
    
    @property 
    def action_space(self):
        """Return action space for compatibility"""
        return self.env.action_space
        
    @property
    def observation_space(self):
        """Return observation space for compatibility"""
        return self.env.observation_space
