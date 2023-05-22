"""
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
# from gymnasium import wrappers
import tensorflow as tf
from importlib import import_module
from collections import deque
import Analyzer

from enum import Enum
class Mode(Enum):
    train = 'train'
    test = 'test'
    continued_train = 'continued_train'
class EnvName(Enum):   # NOTE: gym env name used '-' instead of '_'
    Pendulum_v1 = 'Pendulum-v1'
    CartPole_v1 = 'CartPole-v1'
class AgentName(Enum):
    DQN = 'DQN'
    DDPG = 'DDPG'
    SAC = 'SAC'
    SAC_discrete = 'SAC_discrete'
    SAC_softmax = 'SAC_softmax'
    SAC_softmax_max = 'SAC_softmax_max'
    SAC_entropy = 'SAC_entropy'


class Game:
    def __init__(self, config):
        self.config = config
        self.nEpisodes_toTrain = config["NumOfEpisodes_toTrain"]
        self.nEpisodes_toTest = config["NumOfEpisodes_toTest"] 
        self.period_toSaveModels = config["Period_toSaveModels"]
        self.sumReward_toStopTrain = config["SumReward_toStopTrain"]

    def run(self, mode, env, agent, envUtil, analyzer):
        nEpisodes = self.nEpisodes_toTrain if mode == Mode.train else self.nEpisodes_toTest
        for episodeCnt in range(1, nEpisodes+1):  # for episodeCnt in tqdm(range(1, nEpisodes+1)):
            analyzer.beforeEpisode()
            observFrEnv, info = env.reset()  # observFrEnv shape=(observDim)
            while True:
                    #   print(f"observFrEnv={observFrEnv}")
                observ = envUtil.observCoder.encode(observFrEnv)    # shape=(observDim)
                    #   print(f"observ={observ}")
            
                action = agent.act(observ, envUtil.actionCoder)     # actionCoder to get random action to explore; action shape=(actionDim)

                actionToEnv = envUtil.actionCoder.decode(action)    # actionToEnv: scalar for Discrete, shape=(actionToEnv.nParameters) for Box 

                next_observFrEnv, reward, terminated, truncated, info = env.step(actionToEnv)

                done = (terminated or truncated)  # bool
                experience = envUtil.experienceFrom(observFrEnv, actionToEnv, reward, next_observFrEnv, done, agent.npDtype)
                agent.replayBuffer.remember(experience)
 
                if agent.isReadyToTrain():
                    batch, indices, importance_weights = agent.replayBuffer.sample(agent.batchSz)
                    #   print(f"batch=\n{batch}")
                    loss, td_error = agent.train(batch, importance_weights)

                    if agent.isPER == True:
                        agent.replayBuffer.update_priorities(indices, td_error)
                    analyzer.afterTrain(loss, agent)

                analyzer.afterTimestep(reward)

                observFrEnv = next_observFrEnv
                if done:
                    break

            analyzer.afterEpisode(episodeCnt, agent)
            # Save model
            if mode == Mode.train: 
                if analyzer.isTrainedEnough():
                    agent.save(msg="networks saved and training stopped...")
                    break
                elif (episodeCnt % self.period_toSaveModels == 0): 
                    agent.save(msg="networks saved...")


def getLogger(filepath="./log.log"):
    logger = logging.getLogger("game")
    logger.setLevel(logging.DEBUG) #   INFO, DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    path = Path(filepath)
    path.parent.mkdir(exist_ok=True, parents=True)
    fileHandler = logging.FileHandler(filename=filepath, mode="w")
    fileHandler.setLevel(logging.DEBUG) # INFO, DEBUG
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


if __name__ == "__main__":
    before = time.time()
    np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=160, suppress=True)
    if (not tf.test.is_built_with_cuda()) or len(tf.config.list_physical_devices('GPU')) == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-m', '--mode', type=str, required=True, choices=[i.value for i in Mode], help=f"running mode")
    parser.add_argument('-e', '--environment', type=str, required=True, choices=[i.value for i in EnvName])
    parser.add_argument('-n', '--agent', type=str, required=True, choices=[i.value for i in AgentName])
    args = parser.parse_args()
    mode = Mode(args.mode)  # Mode[config["Mode"]]
    envName = EnvName(args.environment)  # EnvName[config["Environment"]]
    agentName = AgentName(args.agent)  # AgentName[config["Agent"]]
    envModule = import_module(f"EnvUtil_{envName.name}")                  
    envUtil = getattr(envModule, f"EnvUtil_{envName.name}")                 # class
        #   actionToEnv = getattr(envModule, f"ActionCoder_{envName.name}")()    # object
        #   observCoder = getattr(envModule, f"ObservCoder_{envName.name}")()    # object
        #   experienceFrom = getattr(envModule, "experienceFrom")                # function
    agentModule = import_module(f"{agentName.name}")
    Agent = getattr(agentModule, f"{agentName.name}")

    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)
    logger = getLogger(filepath= f"{config['LogPath']}/{envName.value}_{agentName.value}_{mode.value}.log")   
    #   sys.stdout = StdToLog(logger,logging.INFO)
    #   sys.stderr = StdToLog(logger,logging.DEBUG)

    logger.info(f"mode = {mode}")
    logger.info(f"environment = {envName}")
    logger.info(f"agent = {agentName}")
    logger.info(f"config={config}")

    render_mode = "human" if mode == Mode.test else None  
    env = gym.make(envName.value, render_mode=render_mode)
    logger.info(f"env name: {env.unwrapped.spec.id}")  # like "CartPole-v1"
    logger.info(f"env action space: {env.action_space}")
    logger.info(f"env observation space: {env.observation_space}")

    observDim = envUtil.observCoder.nParameters
    actionDim = envUtil.actionCoder.n if envUtil.actionCoder.spaceType == gym.spaces.Discrete else envUtil.actionCoder.nParameters
    #   if envUtil.actionCoder.spaceType == gym.spaces.Box: 
    #       actionDim = envUtil.actionCoder.nParameters 
    #   elif envUtil.actionCoder.type == gym.spaces.Discrete:
    #       actionDim = envUtil.actionCoder.n
    #   else:
    #       actionDim = envUtil.actionCoder.nParameters 
    agent = Agent(mode.value, config, logger, observDim=observDim, actionDim=actionDim)
    analyzer = getattr(Analyzer, f"Analyzer_{envName.name}")(config, logger)
    """
        if env.unwrapped.spec.id == "CartPole-v1": 
            analyzer = Analyzer_CartPole_v1(self.config) 
        elif env.unwrapped.spec.id == "Pendulum-v1": 
            analyzer = Analyzer_Pendulum_v1(self.config)
    """

    game = Game(config)
    game.run(mode, env, agent, envUtil, analyzer)

    logger.info(f"total time={time.time() - before}")

