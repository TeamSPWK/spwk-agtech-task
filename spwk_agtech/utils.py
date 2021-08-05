import types
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pcse.db import NASAPowerWeatherDataProvider
from .const import *


OUTPUT_VARNAME = {k: v['mean'] for k,v in OBSERVATIONS.items()}
OBS_UNIT = [v['unit'] for v in OBSERVATIONS.values()]

# Tick Setting
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
                     '%b',       # ticks are mostly months
                     '%d',       # ticks are mostly days
                     '%H:%M',    # hrs
                     '%H:%M',]  # mins
# these are mostly just the level above...
formatter.zero_formats = [''] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have
# month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
                            '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',]

COL = ['IRRAD', 'TMIN', 'TMAX', 'VAP', 'RAIN', 'E0', 'ES0', 'ET0', 'WIND', "IRRIGATE", "N", "P", "K"]


def NASAPowerWeatherDataFetcher(latitude, longitude, force_update = False, ETmodel = 'PM', fill = 'ffill'):
    
    '''
    WeatherDataProvider for using the NASA POWER database with PCSE

    :param latitude: latitude to request weather data for
    :param longitude: longitude to request weather data for
    :keyword force_update: Set to True to force to request fresh data from POWER website.
    :keyword ETmodel: "PM"|"P" for selecting penman-monteith or Penman
        method for reference evapotranspiration. Defaults to "PM".
    :keyword fill: "ffill"|"bfill"|"linear"| ... for filling holes in weather data. Defaults to "ffill".

    The NASA POWER database is a global database of daily weather data
    specifically designed for agrometeorological applications. The spatial
    resolution of the database is 0.5x0.5 degrees (as of 2018). It is
    derived from weather station observations in combination with satellite
    data for parameters like radiation.
    
    The weather data is updated with a delay of about 3 months which makes
    the database unsuitable for real-time monitoring, nevertheless the
    POWER database is useful for many other studies and it is a major
    improvement compared to the monthly weather data that were used with
    WOFOST in the past.
    
    For more information, check the docstring of [NASAPowerWetherDataProvider] functions!
    '''
    
    # Import weather from around the world at 0.5 X 0.5 resolution using NASA Power data
    weather = NASAPowerWeatherDataProvider(latitude, longitude, force_update, ETmodel)
    
    # Fill the missing data
    if weather.missing > 0:
        logging.debug("there are(is) %s missing value(s) in weather data" % weather.missing)
        df_weather = pd.DataFrame(weather.export())
        full_range = pd.date_range(start=df_weather.DAY.min(), end=df_weather.DAY.max())
        full_range_weather = df_weather.set_index('DAY').reindex(full_range).rename_axis('DAY').reset_index()
        if fill == ("ffill" or "bfill"):
            filled_weather = full_range_weather.fillna(method=fill, axis=0)
        else:
            filled_weather = full_range_weather.interpolate(method='linear', axis=0)
        weather._make_WeatherDataContainers(filled_weather.to_dict(orient="records"))
            
    elif weather.missing < 0:
        raise ValueError("WDP error")
    return weather

def pcse_runner(env, policy, test = True):
    '''
    Return actions and rewards (observations can be obtained directly from env.engine) after running environment
    
    :params env: environment 
    :params policy: fixed_policy (function) or trained (class) or optimized model (class)
    :keyword test: if True, no stochastic
    :return: list of actions and list of rewards
    '''
    actions = []
    rewards = []
    obs = env.reset()
    done = False
    while done != True:
        if isinstance(policy, types.FunctionType):
            act = policy(obs, env)
        else:
            act = policy.get_action(obs, test = test)

        obs, reward, done, info = env.step(act)
        actions.append(act)
        rewards.append(reward)
    return actions, rewards

def pcse_env_obs_show(env, policies:list, policy_name:list, test:list):
    '''
    Visualize observations from running environment using policies (for comparison)
    
    :params env: environment 
    :params policies: list of fixed_policy (function) or trained (class) or optimized model (class) to be compared
    :params policy_name: list of policy names to be visualized
    :keyword test: list of bool. if True, no stochastic
    '''
    actions, rewards = pcse_runner(env, policies[0], test = test[0])
    print("{0:15s}: return ({1:>9.3f}) / net profit ({2:>10.3f}) ".format(policy_name[0], np.sum(rewards), env.profit))
    output_df = pd.DataFrame(env.engine.get_output())
    output_df = output_df[['day'] + env.obs_name]
    output_df.set_index('day', inplace=True)
    output_df.rename(columns = OUTPUT_VARNAME, inplace = True)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,12))
    ax = axes.flatten()
    for ix, var in enumerate(output_df.columns):
        ax[ix].plot(output_df[var])
        ax[ix].xaxis.set_major_locator(locator)
        ax[ix].xaxis.set_major_formatter(formatter)
        ax[ix].set_title(var)
        
    for i, policy in enumerate(policies):
        if i == 0:
            continue
        actions, rewards = pcse_runner(env, policy, test[i])
        print("{0:15s}: return ({1:>9.3f}) / net profit ({2:>10.3f}) ".format(policy_name[i], np.sum(rewards), env.profit))
        output_df = pd.DataFrame(env.engine.get_output())
        output_df = output_df[['day'] + env.obs_name]
        output_df.set_index('day', inplace=True)
        output_df.rename(columns = OUTPUT_VARNAME, inplace = True)
        for ix, var in enumerate(output_df.columns):
            ax[ix].plot(output_df[var])
            
    ax[10].legend(policy_name, frameon = False, bbox_to_anchor = (1, 1), fontsize = 20)
    fig.delaxes(ax[11])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.plot()
    
def pcse_env_act_show(env, policies:list, policy_name:list, test:list):
    '''
    Visualize actions from running environment using policies (for comparison)
    
    :params env: environment 
    :params policies: list of fixed_policy (function) or trained (class) or optimized model (class) to be compared
    :params policy_name: list of policy names to be visualized
    :keyword test: list of bool. if True, no stochastic
    '''
    actions, rewards = pcse_runner(env, policies[0], test = test[0])
    denorm_act = [env.denorm(x, 'act') for x in actions]
    weather_col = ['IRRAD', 'TMIN', 'TMAX', 'VAP', 'RAIN', 'E0', 'ES0', 'ET0', 'WIND']
    WEATHER = pd.DataFrame(env.weather.export()).set_index('DAY')
    act_df = pd.DataFrame(denorm_act, index=pd.date_range(datetime.date(1988, 1, 2), periods=196).tolist(), columns = COL)
    act_df[weather_col] = WEATHER[weather_col]
    
    fig, axes = plt.subplots(4, 4, figsize = (16, 16))
    ax = axes.flatten()
    for ix in range(len(ax) - 3):
        ax[ix].plot(act_df[COL[ix]])
        ax[ix].set_title(COL[ix])
        ax[ix].xaxis.set_major_locator(locator)
        ax[ix].xaxis.set_major_formatter(formatter)
        
    for i, policy in enumerate(policies):
        if i == 0:
            continue
        actions, rewards = pcse_runner(env, policy, test[i])
        denorm_act = [env.denorm(x, 'act') for x in actions]
        act_df = pd.DataFrame(denorm_act, index=pd.date_range(datetime.date(1988, 1, 2), periods=196).tolist(), columns = COL)
        for ix in range(len(ax) - 3):
            ax[ix].plot(act_df[COL[ix]])
            ax[ix].set_title(COL[ix])   
            
    ax[12].legend(policy_name, frameon = False, bbox_to_anchor = (1, 1), fontsize = 20)
    for j in range(13, 16):
        fig.delaxes(ax[j])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.plot()
    
def pcse_env_plot(env, policies:list, policy_name:list, test=None):
    '''
    Visualize observations and actions from running environment using policies (for comparison)
    
    :params env: environment 
    :params policies: list of fixed_policy (function) or trained (class) or optimized model (class) to be compared
    :params policy_name: list of policy names to be visualized
    :keyword test: list of bool. if True, no stochastic
    '''
    if test:
        assert len(test) == len(policies)
    else:
        test = [True] * len(policies)
        
    pcse_env_obs_show(env, policies, policy_name, test)
    pcse_env_act_show(env, policies, policy_name, test)

def pcse_engine_plot(output:dict, output_varname:dict={None}):
    '''
    Directly visualize outputs from engine
    
    :params output: = wofost.get_output()
    :keyword output_varname: dictionary of key: abbr names of output and values: converted name (see the OUTPUT_VARNAME)
    '''
    if output_varname == {None}:
        output_varname = OUTPUT_VARNAME
        
    df = pd.DataFrame(output)
    df.set_index('day', inplace=True)
    df.rename(columns = output_varname, inplace = True)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6*3,6*4))
    ax = axes.flatten()
    for ix, var in enumerate((output_varname.values())):
        try:
            df[var].plot(ax = ax[ix], legend = var)
            ax[ix].xaxis.set_major_locator(locator)
            ax[ix].xaxis.set_major_formatter(formatter)
            ax[ix].set_ylabel("%s" % OBS_UNIT[ix])
            plt.setp(ax[ix].yaxis.get_label(), rotation=0, fontsize=10, position = (0, 1.02))
        except Exception as e:
            logging.warning(f"{var} can't be shown due to {e}")
            fig.delaxes(axes.flatten()[ix])
    fig.delaxes(axes.flatten()[-1])
    plt.show()