import os
import datetime
import yaml
import copy
import numpy as np

import pcse
from pcse.engine import Engine
from pcse.fileinput import CABOFileReader
from pcse.base import ParameterProvider

import gym
from gym.spaces import Box
from .utils import *
from .const import *

import logging

'''
hyperparameter

lat = 35
long = 128
crop_name = 'wheat'
variety_name = 'winter-wheat'
campaign_start_date = '1988-01-01'
emergence_date = "1988-01-01"
# max_duration = 365
'''

pcse_data_dir = os.path.join(os.path.dirname(__file__), "data")

def get_profit(state, action, done):
    
    if done:
        price = state[3] * 279.34 / 1000 # wheat price is 279.34 USD / 1000 kg
    else:
        price = 0
    
    irrigation_cost = action[9] * 50 / 10 # water price is 50 USD / 1000m3
    npk_cost = (action[10] * 250 / 1000) + (action[11] * 460 / 1000) + (action[12] * 370 / 1000)
    cost = irrigation_cost + npk_cost
    return price - cost

def get_reward(prev_state, state, action, done):
    
    init_factor = 0
    progress_factor = 0
    
    return get_profit(state, action, done)


class PCSE_Env(gym.Env):
   
    metadata = {'render.modes': ['human']}

    def __init__(self, lat = 35, long = 128, crop_name = 'wheat', variety_name = 'winter-wheat', campaign_start_date = '1988-01-01', emergence_date = "1988-01-01"):
        super(PCSE_Env, self).__init__()
        self.lat = lat
        self.long = long
        self.crop_name = crop_name
        self.variety_name = variety_name
        self.campaign_start_date = campaign_start_date
        self.emergence_date = emergence_date
        self.end_date = datetime.datetime.strftime(datetime.datetime.strptime(emergence_date, '%Y-%m-%d') + datetime.timedelta(days = 365), '%Y-%m-%d')
        
        self.action_space = Box(low = np.array([-1]*13, dtype=np.float32), \
                                high = np.array([1]*13, dtype=np.float32))

        self.observation_space = Box(low = np.array([-1]*11, dtype=np.float32), \
                                    high = np.array([1]*11, dtype=np.float32))
            
        self.action_min = np.array([v['min'] for v in ACTIONS.values()], dtype=np.float32)
        self.action_max = np.array([v['max'] for v in ACTIONS.values()], dtype=np.float32)
        self.obs_min = np.array([v['min'] for v in OBSERVATIONS.values()], dtype=np.float32)
        self.obs_max = np.array([v['max'] for v in OBSERVATIONS.values()], dtype=np.float32)
        self.obs_name = [k for k in OBSERVATIONS]
        self.obs_unit = [v['unit'] for v in OBSERVATIONS.values()]
    
        self.ref_weather = NASAPowerWeatherDataFetcher(self.lat, self.long)
        self.profit = 0
        self.need_reset = True
        self.done = False
    
    
    def denorm(self, obs_act, _type):
        if _type == "act":
            denorm_obs_act = (obs_act * (self.action_max - self.action_min) + self.action_min + self.action_max)/2
            denorm_obs_act = np.clip(denorm_obs_act, self.action_min, self.action_max)
        elif _type == "obs":
            denorm_obs_act = (obs_act * (self.obs_max - self.obs_min) + self.obs_min + self.obs_max)/2
        return denorm_obs_act
    
    
    def norm(self, obs_act, _type):
        if _type == "act":
            norm_obs_act = (2*obs_act - (self.action_max + self.action_min))/(self.action_max  - self.action_min)
        elif _type == 'obs':
            norm_obs_act = (2*obs_act - (self.obs_max + self.obs_min))/(self.obs_max  - self.obs_min)
            norm_obs_act = np.clip(norm_obs_act, -1, 1)
        return norm_obs_act
    
    
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
        """.format(cname=self.crop_name, vname=self.variety_name, 
                   start=self.campaign_start_date, startdate=self.emergence_date, 
                   end = self.end_date, maxdur=365)
        
        self.crop = CABOFileReader(os.path.join(pcse_data_dir, "wofost_npk.crop"))
        self.soil = CABOFileReader(os.path.join(pcse_data_dir, "wofost_npk.soil"))
        self.site = CABOFileReader(os.path.join(pcse_data_dir, "wofost_npk.site"))
        self.params = ParameterProvider(soildata=self.soil, cropdata=self.crop, sitedata=self.site)
        
        
    def _engine_init(self):
        
        self._module_init()
        self.agro = yaml.safe_load(self.agro_yaml)
        self.engine = Engine(self.params, self.weather, self.agro, config=os.path.join(pcse_data_dir, "Wofost71_NPK.conf"))
        self.current_date = self.engine.day
        
        
    def Obs_shaping(self, raw_obs, obs_name):
        obs = np.array([raw_obs[x] for x in obs_name if x in raw_obs.keys()], dtype=np.float32)
        obs = self.norm(obs, "obs")
        return obs

    
    def reset(self):
        self.profit = 0
        self.need_reset = False
        self.done = False
        self._engine_init()
        obs = self.Obs_shaping(self.engine.get_output()[-1], self.obs_name)
        self.obs = obs
        return obs
    
    
    def step(self, action):
        if (self.need_reset == True) | (self.done == True):
            logging.error("Needs reset")
            return None
            
        action = self.denorm(action, 'act')
        send_actions2engine(action, self.engine)
        self.engine.run(days = 1)
        
        if self.engine.day - self.current_date == datetime.timedelta(0):
            self.done = True    
        else:
            self.current_date = self.engine.day
        
        next_obs = self.Obs_shaping(self.engine.get_output()[-1], self.obs_name)
        if self.denorm(next_obs, 'obs')[0] >= 2:
            self.done = True
        
        self.profit += get_profit(self.denorm(next_obs, 'obs'), action, self.done)
        reward = get_reward(self.denorm(self.obs, 'obs'), self.denorm(next_obs, 'obs'), action, self.done)
        info = {}
        
        self.obs = next_obs
        return next_obs, reward, self.done, info
    
    
    def render(self, mode='human'):
        print(f"profit: {self.profit} USD/ha")
        try:
            plot_pcse_engine(self.engine.get_output())
        except AttributeError:
            loggin.error("Needs reset. You should first initialize environment.")
    
    
    def close(self):
        pass