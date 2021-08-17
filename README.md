<h1 align="center">Spacewalk Agtech Engineer Task</h1>
<p align="center">Task for spacewalk agtech engineer applicants</p>

<p align="center"><a href="https://github.com/TeamSPWK/spwk-agtech-task/releases"><img src="https://img.shields.io/badge/release-v0.2.1-blue" alt="release state" /></a>
<a href="#description"><img src="https://img.shields.io/badge/env-PCSE--v0-blueviolet" alt="env" /></a>
<a href="https://github.com/TeamSPWK/spwk-agtech-task/blob/develop/LICENSE"><img src="https://img.shields.io/badge/license-GPL--v3-green" alt="env" /></a>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#install">Install</a> •
  <a href="#get-started">Get started</a> •
  <a href="#documentation">Documentation</a>
</p>

<h2 align="center">Description</h2>

This package contains tasks and instructions for spacewalk Agtech engineer applicants.

[PCSE](https://pcse.readthedocs.io/en/stable/) (Python Crop Simulation Environment) is a Python package for building crop simulation models, in particular the crop models developed in Wageningen (Netherlands). PCSE is no exception and its source code is open and licensed under the European Union Public License. 

In this task, we need to solve a simple agricultural reinforcement learning problem created by modified PCSE. 

Goal: `MAXIMIZE RETURN!!`

List of environments :

* `PCSE-v0`

<h2 align="center">Install</h2>

Install `spwk_agtech` package with : 

(It is recommended to install the package in virtual environment due to dependencies)

```console
pip install git+https://github.com/TeamSPWK/spwk-agtech-task.git
```

<h2 align="center">Get started</h2>

You can use the environment like any other gym environment :

```console
# runner
import spwk_agtech
import gym

env = gym.make("PCSE-v0") # there is just version 0 in now

return = 0
obs = env.reset()
step = 0
while True:
    step += 1
    action = env.action_space.sample()
    next_obs, rew, done, _  = env.step(action)
    return += rew
    obs = next_obs
    if done:
        break

```

<h2 align="center">Documentation</h2>

[Task Details](https://github.com/TeamSPWK/spwk-agtech-task/blob/develop/documents/Task_Details.md)
