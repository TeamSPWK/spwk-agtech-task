<h1 align="center">Task Details</h1>

<p align="center">
  <a href="#problem-descriptions">Problem Descriptions</a> •
  <a href="#environment">Environment</a> •
  <a href="#submission-guidelines">Submission Guidelines</a> •
  <a href="#scoring-criteria">Scoring Criteria</a> •
  <a href="#appendix">Appendix</a>
</p>

<h2 align="center">Problem Descriptions</h2>

Welcome to the documentation of `PCSE-v0`, the gym environment for cultivating crops.

<h3>MDP design</h3>

- Observation
  - Plant information (9 items)
  - Soil information (2 items)
- Action (`continuous`)
  - Weather (9 items)
  - Irrigation (1 items)
  - N/P/K application (3 items)
- Reward
  - It is calculated profit from yield and cost
  - In now, we only use "profit". You can modify the reward funtion to help training agent.
- Done
  - 1 episode will be terminated when `DVS` reachs to 2 or the simulation ends.


🔎 _you can check details of `observations` and `actions` in [const.py](https://github.com/TeamSPWK/spwk-agtech-task/blob/master/spwk_agtech/const.py)_

🔎 _you can check details of `reward` and `done` in [pcse_env.py](https://github.com/TeamSPWK/spwk-agtech-task/blob/develop/spwk_agtech/pcse_env.py)_

<h3>Goal</h3>

- Maximize `profit` (In now, sum of reward)
- High stability of training
- Fast convergence
- (optional) Make generialized agent
  - There are additional test PCSE setting that we do not offer to applicants
  - Good performance on those setting is not mandatory, but recommended
  - Extra points will given for performance on those test setting

<h2 align="center">Environment</h2>

It's a multi-steps environment with `continuous actions`

<h3>Allowed</h3>

- You can wrap the given environment with your custom environment
    - Depending on your method, current MDP design may not be sufficient to solve the problem. Given environment provides minimal information to solve the problem and it has large range of observations and actions. You can create a wrapper environment to generate additional information
    - If you are familiar with some libraries(like gym, etc.), you can use given environment as a library and create wrapper environment inherits framework you want to use
    - If you create a wrapper environment, you have to submit that as well
- You can modify the reward funtion to help training agent (It needs to be submintted as well)


<h3>Methods of PCSE-v0</h3>

- `env = gym.make("PCSE-v0")`

- `env.norm(self, obs_act, _type)` and `env.denorm(self, obs_act, _type))`
  - Normalize the observations from PCSE engine to -1 ~ 1
  - Denormalze the actions (-1 ~ 1) from agent to real value
  - args
    - `obs_act (ndarray)`: observations or actions
    - `_type (str)`: the type of `obs_act`. `obs` or `act`


- `env._module_init(self)`
  - Initialize crop, soil, and site modules for PCSE engine
  - This function is about module initialization of PCSE engine, so you don't need to worry about it.

- `env._engine_init(self)`
  - Initialize PCSE engine
  - Call `_module_init`
  - This function is about module initialization of PCSE engine, so you don't need to worry about it.

- `env.reset(self)`
  - Reset environment
  - Call `_engine_init`
  - returns
    - `obs (ndarray)`

- `env.step(self, action)`
  - Apply actions
      - args
          - `action (ndarray)` : 13 different 1-d continuous actions. Check [const.py](https://github.com/TeamSPWK/spwk-agtech-task/blob/master/spwk_agtech/const.py)
      - returns
          - `next_obs (ndarray)` : It is observed after given actions.
          - `reward` : reward from current actions and next observations.
          - `self.done` : done state
          - `info` : additional information (nothing in now).

- `env.render(self, mode='human')`
  - Print current profit.
  - Render current state. (11 different 1-d observations from beginning to the present)

<h2 align="center">Submission Guidelines</h2>

<h3>Submission Deadline</h3>

- Applicants are free to set the deadline for submission. However, you must notify us of the deadline in advance.
    - Deadline notification due : Until the day after the test notification mail is sent
    - Please notify us by e-mail (<hychoi@spacewalk.tech>).

You don't have to rush to submit, so give yourself plenty of time.
Once you fix the deadline, please stick to it.
If you think you will not be able to do it within the deadline, please notify us by e-mail in advance.

<h3>Submission List</h3>

- Answer document
    - Answer document should describe in detail how and why you solved the problem like that.
    - There is no restrictions on the format of the document, but the document should be sufficiently expressive of how you solved the problem.
    - Please add some images and performance properties of your final results on the document (the graph from `env.render()`).
    - Training graph have to be attached to the document.
    - We will find not only your theoretical approach, but also your skills to produce proper training information, and to drive the learning with proper tuning.

- code
    - Agent(Problem solver) code have to be submitted.
    - Reproducibility is crucial. If you have additional resources required to reproduce your result, you should include them in your submission (checkpoints, etc.)

(Optional)
- Wrapper environment code
    - If you created wrapper environment code, you should include it in your submission.

Reproducibility with minimum effort is the most important for your submission.
You don't have to divide your code into agent part and environment part.
Please attach all things that required for us to reproduce your result.
If we fail to reproduce your result described on answer document, and if there are no valid cause for the non-reproducibility, your score may be deducted.

<h3>How to Submit</h3>

- Please submit your submission by e-mail(<hychoi@spacewalk.tech>).

<h2 align="center">Scoring Criteria</h2>

- Problem solving process : 20%
    - Problem solving process have to be scientific.
- Scientific hypothesis : 15%
    - It includes the expected characteristic of the problem, and why you choose the method.
    - The hypothesis have to be reasonable, and scientific.
- Phenomenon analysis : 15%
    - Once you validate your hypothesis with experiment, you can get the result.
    - From result, observe phenomenon and analyze it.
- Document & code quality : 15%
    - The document should follow your problem solving process.
    - Your hypothesis should be described on the document.
    - The phenomenon analysis have to be described on the document.
    - Document and code should be written in a communicative manner.
- Result performance : 35%
    - Spacewalk is a performance-oriented company, and we will score result performance as well as problem solving procedure.
    - We will retrain your model, and check if the result is same as you described on the document.
    - The criteria of result performance are "final performance", "sample efficiency" and "training stability".
      - The final performance is scored by re-evaluation using your checkpoint. We will use your highest score.
      - The sample efficiency is scored based on the average of episodes taken to produce results up to baseline scores (`1402`) during `10 tries`. The maximum episode length tested is `100k`. If the final performance does not reach the baseline score, You will get 0 for this criteria. If there are cases that do not reach the baseline, they are excluded from the average.
      - During scoring the sample efficiency, We will also score the training stability of your algorithms. If the final performance does not reach the baseline score, You will get 0 for this criteria. The default score is 10 points, and one point will be deducted for each case of not reaching the baseline score.


      | Score | Final performance (50 %) | Sample efficiency (30 %) |
      | -----------: | -----------: | -----------: |
      | 10     | > 3000    | <= 250    |
      | 9      | >= 2800   | <= 500    |
      | 8      | >= 2600   | <= 1K     |
      | 7      | >= 2300   | <= 2K     |
      | 6      | >= 2000   | <= 3K     |
      | 5      | >= 1700   | <= 6K     |
      | 4      | >= 1402   | <= 12K    |
      | 3      | >= 700    | <= 25K    |
      | 2      | >= 0      | <= 50K    |
      | 1      | >= -20000 | <= 100K   |
      | 0      | < -20000  | > 100K    |

<h2 align="center">Appendix</h2>

Examples of `env.render()`

`profit: -36628.68041134567 USD/ha`
![download](https://user-images.githubusercontent.com/87963916/128451737-0848aa67-8e61-4209-886f-f29860af3b5e.png)


