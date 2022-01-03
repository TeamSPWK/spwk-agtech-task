import copy
import datetime
import logging
import os

import gym
import numpy as np
import yaml
from gym.spaces import Box
from pcse.base import ParameterProvider
from pcse.engine import Engine
from pcse.fileinput import CABOFileReader

from .const import ACTIONS, OBSERVATIONS
from .utils import (NASAPowerWeatherDataFetcher, plot_pcse_engine,
                    send_actions2engine)

pcse_data_dir = os.path.join(os.path.dirname(__file__), 'data')


def get_profit(state, action, done):
    """ Get profit from state, action and done state.

    Args:
        state (np.ndarray or tensor): State from PCSE environment
        action (np.ndarray or tensor): Action from Actor
        done (bool): Done state

    Returns:
        float: Profit (Income - Cost)
    """

    if done:
        price = state[3] * 279.34 / 1000  # wheat price is 279.34 USD / 1000 kg
    else:
        price = 0

    irrigation_cost = action[9] * 50 / 10  # water price is 50 USD / 1000m3
    npk_cost = (action[10] * 250 / 1000) + (action[11] * 460 /
                                            1000) + (action[12] * 370 / 1000)
    # N = 250 USD/kg, P = 460 USD/kg, K = 370 USD/kg
    cost = irrigation_cost + npk_cost
    return price - cost


def get_reward(prev_state, state, action, done):
    """ Get reward. In now, only use get_profit function.
        You can modify this funtion to help training agent.

    Args:
        prev_state (np.ndarray or tensor): Previous state
        state (np.ndarray or tensor): Current state
        action (np.ndarray or tensor): Action
        done (bool): Done state

    Returns:
        float: Reward
    """

    return get_profit(state, action, done)


class PcseEnv(gym.Env):
    """
    Description:
        A simulation environment for growing crops in the open field

        Current PCSE engine settings

        lat = 35
        long = 128
        crop_name = 'wheat'
        variety_name = 'winter-wheat'
        campaign_start_date = '1988-01-01'
        emergence_date = '1988-01-01'
        max_duration = 365

    Observations:
        Type: Box(11)
        'DVS':'crop DeVelopment Stage' 0 at emergence 1 at Anthesis (flowering) and 2 at maturity
        'LAI': 'Leaf Area Index' including stem and pod area
        'TAGP': 'Total Above Ground Production'
        'TWSO': 'Total dry Weight of Storage Organs'
        'TWLV': 'Total dry Weight of LeaVes'
        'TWST': 'Total dry Weight of STems'
        'TWRT': 'Total dry Weight of RooTs'
        'TRA': 'crop TRAnspiration RAte'
        'RD': 'Rooting Depth'
        'SM': 'Soil Moisture' root-zone, Volumetric moisture content in root zone
        'WWLOW': 'WWLOW = WLOW + W' Total amount of water in the soil profile

    Actions: Check const.py for detailed information
        Type: Box(13)
        'IRRAD': Incoming global radiaiton
        'TMIN': Daily minimum temperature
        'TMAX': Daily maximum temperature
        'VAP': Daily mean vapour pressure
        'RAIN': Daily total rainfall
        'E0': Penman potential evaporation from a free water surface
        'ES0': Penman potential evaporation from a moist bare soil surface
        'ET0': Penman or Penman-Monteith potential evaporation for a reference crop canopy
        'WIND': Daily mean wind speed at 2m height
        'IRRIGATE': Amount of irrigation in cm water applied on this day
        'N': Amount of N fertilizer in kg/ha applied on this day
        'P': Amount of P fertilizer in kg/ha applied on this day
        'K': Amount of K fertilizer in kg/ha applied on this day

    Note: Check const.py for detailed information about obseravations and actions

    Reward:
        Only terminal reward. [Profit = Income - Cost]
        Check get_profit function.

    Starting State:
        Now, it is fixed.

    Episode Termination:
        If 'DVS' > 2.
        If simulation ends (365 days).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 lat=35,
                 long=128,
                 crop_name='wheat',
                 variety_name='winter-wheat',
                 campaign_start_date='1988-01-01',
                 emergence_date='1988-01-01'):
        super().__init__()
        self.lat = lat
        self.long = long
        self.crop_name = crop_name
        self.variety_name = variety_name
        self.campaign_start_date = campaign_start_date
        self.emergence_date = emergence_date
        self.end_date = datetime.datetime.strftime(
            datetime.datetime.strptime(emergence_date, '%Y-%m-%d') +
            datetime.timedelta(days=365), '%Y-%m-%d')

        self.observation_space = Box(low = np.array([-1]*11, dtype=np.float32), \
                                    high = np.array([1]*11, dtype=np.float32))

        self.action_space = Box(low = np.array([-1]*13, dtype=np.float32), \
                                high = np.array([1]*13, dtype=np.float32))

        self.obs_min = np.array([v['min'] for v in OBSERVATIONS.values()],
                                dtype=np.float32)
        self.obs_max = np.array([v['max'] for v in OBSERVATIONS.values()],
                                dtype=np.float32)
        self.action_min = np.array([v['min'] for v in ACTIONS.values()],
                                   dtype=np.float32)
        self.action_max = np.array([v['max'] for v in ACTIONS.values()],
                                   dtype=np.float32)
        self.obs_name = list(OBSERVATIONS.keys())
        self.obs_unit = [v['unit'] for v in OBSERVATIONS.values()]

        self.ref_weather = NASAPowerWeatherDataFetcher(self.lat, self.long)
        self.profit = 0
        self.need_reset = True
        self.done = False

    def denorm(self, value, cat):
        if cat == 'act':
            denorm_value = (value * (self.action_max - self.action_min) +
                            self.action_min + self.action_max) / 2
            denorm_value = np.clip(denorm_value, self.action_min,
                                   self.action_max)
        elif cat == 'obs':
            denorm_value = (value * (self.obs_max - self.obs_min) +
                            self.obs_min + self.obs_max) / 2
        return denorm_value

    def norm(self, value, cat):
        if cat == 'act':
            norm_value = (2 * value - (self.action_max + self.action_min)) / (
                self.action_max - self.action_min)
        elif cat == 'obs':
            norm_value = (2 * value -
                          (self.obs_max + self.obs_min)) / (self.obs_max -
                                                            self.obs_min)
            norm_value = np.clip(norm_value, -1, 1)
        return norm_value

    def _module_init(self):
        self.weather = copy.deepcopy(self.ref_weather)
        self.agro_yaml = """
        - {start}:
            CropCalendar:
                crop_name: {cname}
                variety_name: {vname}
                crop_start_date: {startdate}
                crop_start_type: emergence
                crop_end_date: null
                crop_end_date: {end}
                crop_end_type: harvest
                max_duration: {maxdur}
            TimedEvents:
            -   event_signal: irrigate
                name: Irrigation application table
                comment: All irrigation amounts in cm
                events_table:
                - {start}: {{amount: 0, efficiency: 0.7}}
            -   event_signal: apply_npk
                name:  Timed N/P/K application table
                comment: All fertilizer amounts in kg/ha
                events_table:
                - {start}: {{N_amount : 0, P_amount: 0, K_amount: 0, N_recovery: 0.7, P_recovery: 0.7, K_recovery: 0.7}}
            StateEvents:
        """.format(cname=self.crop_name,
                   vname=self.variety_name,
                   start=self.campaign_start_date,
                   startdate=self.emergence_date,
                   end=self.end_date,
                   maxdur=365)

        self.crop = CABOFileReader(
            os.path.join(pcse_data_dir, 'wofost_npk.crop'))
        self.soil = CABOFileReader(
            os.path.join(pcse_data_dir, 'wofost_npk.soil'))
        self.site = CABOFileReader(
            os.path.join(pcse_data_dir, 'wofost_npk.site'))
        self.params = ParameterProvider(soildata=self.soil,
                                        cropdata=self.crop,
                                        sitedata=self.site)

    def _engine_init(self):

        self._module_init()
        self.agro = yaml.safe_load(self.agro_yaml)
        self.engine = Engine(self.params,
                             self.weather,
                             self.agro,
                             config=os.path.join(pcse_data_dir,
                                                 'Wofost71_NPK.conf'))
        self.current_date = self.engine.day

    def get_obs(self, raw_obs, obs_name):
        obs = np.array([raw_obs[x] for x in obs_name if x in raw_obs.keys()],
                       dtype=np.float32)
        obs = self.norm(obs, 'obs')
        return obs

    def reset(self):
        self.profit = 0
        self.need_reset = False
        self.done = False
        self._engine_init()
        obs = self.get_obs(self.engine.get_output()[-1], self.obs_name)
        self.obs = obs
        return obs

    def step(self, action):
        if (self.need_reset is True) | (self.done is True):
            logging.error('Needs reset')
            return None

        action = self.denorm(action, 'act')
        send_actions2engine(action, self.engine)
        self.engine.run(days=1)

        if self.engine.day - self.current_date == datetime.timedelta(0):
            self.done = True
        else:
            self.current_date = self.engine.day

        next_obs = self.get_obs(self.engine.get_output()[-1],
                                    self.obs_name)
        if self.denorm(next_obs, 'obs')[0] >= 2:
            self.done = True

        self.profit += get_profit(self.denorm(next_obs, 'obs'), action,
                                  self.done)
        reward = get_reward(self.denorm(self.obs, 'obs'),
                            self.denorm(next_obs, 'obs'), action, self.done)
        info = {}

        self.obs = next_obs
        return next_obs, reward, self.done, info

    def render(self, mode="human"):
        print(f'profit: {self.profit} USD/ha')
        try:
            fig = plot_pcse_engine(self.engine.get_output())
            return fig
        except AttributeError:
            logging.error(
                'Needs reset. You should first initialize environment.')
            return None

    def close(self):
        pass
