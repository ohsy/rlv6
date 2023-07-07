"""
"""
import os
import sys
import json
import time
from datetime import datetime
import logging
import argparse
import pathlib 
import numpy as np
import gymnasium as gym
# from gymnasium import wrappers
import tensorflow as tf
from importlib import import_module
from collections import deque

from env_daiso.environment import DaisoSokcho
from coder_daiso import Coder_daiso as Coder
from analyzer_daiso import Analyzer_daiso as Analyzer

from enum import Enum
class Mode(Enum):
    train = 'train'
    test = 'test'
    continued_train = 'continued_train'
class EnvName(Enum):   # NOTE: gym env name used '-' instead of '_'
    Pendulum_v1 = 'Pendulum-v1'
    CartPole_v1 = 'CartPole-v1'
    LunarLander_v2 = 'LunarLander-v2'
    DaisoSokcho = 'DaisoSokcho'
    DaisoSokcho_discrete = 'DaisoSokcho_discrete'
class AgentName(Enum):
    DQN = 'DQN'
    DDPG = 'DDPG'
    SAC = 'SAC'
    SAC_discrete = 'SAC_discrete'
    SAC_entropy = 'SAC_entropy'
    SAC_ec = 'SAC_ec'  # entropy_continuous


class Game:
    def __init__(self, config):
        self.config = config
        self.period_toSaveModels = config["Period_toSaveModels"]

    def run(self, nEpisodes, mode, env, agent, coder, analyzer):
        analyzer.beforeMainLoop()
        for episodeCnt in range(1, nEpisodes+1):  # for episodeCnt in tqdm(range(1, nEpisodes+1)):
            analyzer.beforeEpisode()
            observFrEnv, info = env.reset()  # observFrEnv (observDim)
            while True:
                observ = coder.observCoder.encode(observFrEnv)    # (observDim)
            
                action = agent.act(observ, coder.actionCoder)     # actionCoder to get random action to explore; (actionDim)

                actionToEnv = coder.actionCoder.decode(action)    # scalar for Discrete, (actionToEnv.nParameters) for Box 

                next_observFrEnv, reward, terminated, truncated, info = env.step(actionToEnv)

                done = (terminated or truncated)  # bool
                experience = coder.experienceFrom(observFrEnv, actionToEnv, reward, next_observFrEnv, done, agent.npDtype)
                agent.replayBuffer.remember(experience)
 
                if agent.isReadyToTrain():
                    batch, indices, importance_weights = agent.replayBuffer.sample(agent.batchSz)
                    #   print(f"batch=\n{batch}")
                    loss, td_error = agent.train(batch, importance_weights)

                    if agent.isPER == True:
                        agent.replayBuffer.update_priorities(indices, td_error)
                    analyzer.afterTrain(loss, agent)

                analyzer.afterTimestep(reward, info)

                observFrEnv = next_observFrEnv
                if done:
                    break

            analyzer.afterEpisode(episodeCnt, agent)
            # Save model
            if mode == Mode.train: 
                if analyzer.isTrainedEnough():
                    agent.save()
                    analyzer.afterSave("networks saved and training stopped...")
                    break
                elif (episodeCnt % self.period_toSaveModels == 0): 
                    agent.save()
                    analyzer.afterSave("networks saved...")
        analyzer.afterMainLoop()


def getLogger(filepath="./log.log"):
    logger = logging.getLogger("game")
    logger.setLevel(logging.DEBUG) #   INFO, DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileHandler = logging.FileHandler(filename=filepath, mode="w")
    fileHandler.setLevel(logging.DEBUG) # INFO, DEBUG
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


if __name__ == "__main__":
    np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=160, suppress=True)
    if (not tf.test.is_built_with_cuda()) or len(tf.config.list_physical_devices('GPU')) == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(os.getcwd()+'/config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-e', '--environment', type=str, required=True, choices=[i.value for i in EnvName])
    parser.add_argument('-n', '--agent', type=str, required=True, choices=[i.value for i in AgentName])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=[i.value for i in Mode], help=f"running mode")
    args = parser.parse_args()
    envName = EnvName(args.environment)  # EnvName[config["Environment"]]
    agentName = AgentName(args.agent)  # AgentName[config["Agent"]]
    mode = Mode(args.mode)  # Mode[config["Mode"]]

    dt = datetime.now().strftime('%m%d_%H%M')
    logdirpath = f"{config['LogPath']}/{envName.value}_{agentName.value}_{mode.value}"
    logfilepath = f"{logdirpath}/{dt}.log"
    pathlib.Path(logfilepath).parent.mkdir(exist_ok=True, parents=True)
    logger = getLogger(filepath = logfilepath)
    summaryPath = f"{logdirpath}/{dt}_summary"  # directory 
    summaryWriter = tf.summary.create_file_writer(summaryPath)

    coder = Coder(envName.name, config, logger)  
    analyzer = Analyzer(envName.name, config, logger, summaryWriter)
    if envName == envName.DaisoSokcho or envName == envName.DaisoSokcho_discrete:
        env = DaisoSokcho(phase = mode.value)
    else:
        env = gym.make(envName.value, render_mode=("human" if mode == Mode.test else None))  

    Agent = getattr(import_module(f"{agentName.name}"), f"{agentName.name}")
    agent = Agent(envName.name, mode.value, config, logger, coder.observCoder.encodedDim, coder.actionCoder.encodedDim)

    nEpisodes = config["NumOfEpisodes_toTrain"] if mode in [Mode.train, Mode.continued_train] else config["NumOfEpisodes_toTest"] 

    logger.info(f"environment = {envName.value}")  # logger.info(f"env name: {env.unwrapped.spec.id}")  # like "CartPole-v1"
    logger.info(f"agent = {agentName.value}")
    logger.info(f"mode = {mode.value}")
    logger.info(f"config={config}")
    logger.info(f"env action space: {env.action_space}")
    logger.info(f"env observation space: {env.observation_space}")
    logger.info(f"coder = {coder.__class__.__name__}")
    logger.info(f"analyzer = {analyzer.__class__.__name__}")
    logger.info(f"explorer = {agent.explorer.__class__.__name__}")
    logger.info(f"memoryCapacity = {agent.memoryCapacity}")
    logger.info(f"memoryCnt_toStartTrain = {agent.memoryCnt_toStartTrain}")
    logger.info(f"nEpisodes = {nEpisodes}")

    game = Game(config)
    game.run(nEpisodes, mode, env, agent, coder, analyzer)

