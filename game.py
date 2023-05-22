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

    def run(self, mode, env, agent, envUtil):
        #   analyzer = Analyzer_cartpoleV1_dqn(self.config) if env.unwrapped.spec.id == "CartPole-v1" and agent.__class__.__name__ == "DQN" \
        #         else Analyzer_pendulumV1_actorcritic(self.config)
        if env.unwrapped.spec.id == "CartPole-v1": 
            analyzer = Analyzer_CartPole_v1(self.config) 
        elif env.unwrapped.spec.id == "Pendulum-v1": 
            analyzer = Analyzer_Pendulum_v1(self.config)

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


class Analyzer:
    status_train = "train"
    status_preTrain = "preTr"  # preparing train

    def __init__(self, config):
        self.sumReward_toStopTrain = config["SumReward_toStopTrain"]
        self.avgReward_toStopTrain = config["AvgReward_toStopTrain"]
        self.status = self.status_preTrain  
        self.preStatus = None
        self.rewards = []
        self.sumReward = 0
        self.sumReward_max = 0
        self.sumReward_recent_maxlen = 4
        self.sumReward_recent = deque(maxlen=self.sumReward_recent_maxlen)    
        self.avgReward = 0
        self.losss0 = []
        self.losss1 = []
        self.trainCnt = 0
        self.timeBeforeEpisode = None
        self.envName: str
        
    def beforeEpisode(self):
        self.rewards = []
        self.losss0 = []
        self.losss1 = []
        self.timeBeforeEpisode = time.time()

    def afterTrain(self, loss, agent):
        if type(loss) is tuple:
            self.losss0.append(loss[0])
            self.losss1.append(loss[1])
            agent.summaryWrite("loss0", loss[0], step=self.trainCnt)
            agent.summaryWrite("loss1", loss[1], step=self.trainCnt)
        else:
            self.losss0.append(loss)
            agent.summaryWrite("loss", loss, step=self.trainCnt)
        #   tf.summary.scalar("alpha_loss", alpha_loss, step=trainCnt)
        #   tf.summary.scalar("alpha", agent.alpha, step=trainCnt)
        self.trainCnt += 1

    def afterTimestep(self, reward):
        self.rewards.append(reward)

    def afterEpisode(self, episodeCnt, agent):
        self.preStatus = self.status
        self.status = self.status_train if agent.isReadyToTrain() or self.preStatus == self.status_train else self.status_preTrain
        tm = time.time() - self.timeBeforeEpisode
        if self.preStatus != self.status_train and self.status == self.status_train:  # only once
            agent.summary()

        avg_loss0 = sum(self.losss0) / len(self.losss0) if len(self.losss0) != 0 else 0
        avg_loss1 = sum(self.losss1) / len(self.losss1) if len(self.losss1) != 0 else 0
        self.sumReward = sum(self.rewards)  # NOTE: sumReward != return as gamma is not applied
        self.sumReward_recent.append(self.sumReward)
        self.sumReward_max = self.sumReward if self.sumReward > self.sumReward_max else self.sumReward_max
        self.avgReward = self.sumReward / len(self.rewards)  # NOTE: sumReward != return as gamma is not applied
        msg = f"({self.status}) episode {episodeCnt}: {tm:.3f}sec" 
        msg += f", avg_loss0: {avg_loss0:.3f}" 
        msg += f", avg_loss1: {avg_loss1:.3f}" if len(self.losss1) > 0 else ""
        if self.envName == EnvName.CartPole_v1.name:
            msg += f", sumReward: {self.sumReward:.3f}, sumReward_max: {self.sumReward_max:.3f}" 
        elif self.envName == EnvName.Pendulum_v1.name:
            msg += f", avgReward: {self.avgReward:.3f}" 
        msg += f", epsilon: {agent.explorer.epsilon:.3f}" if hasattr(agent.explorer,"epsilon") else ""
        logger.info(msg)
        agent.summaryWrite("sumReward", self.sumReward, step=episodeCnt)
        agent.summaryWrite("avgReward", self.avgReward, step=episodeCnt)

    def isTrainedEnough(self):
        pass

class Analyzer_CartPole_v1(Analyzer):
    def __init__(self, config):
        super().__init__(config)
        self.envName = "CartPole_v1"
    def isTrainedEnough(self):
        self.sumReward_recent_mean = sum(self.sumReward_recent) / self.sumReward_recent_maxlen
        return self.sumReward_recent_mean > self.sumReward_toStopTrain

class Analyzer_Pendulum_v1(Analyzer):
    def __init__(self, config):
        super().__init__(config)
        self.envName = "Pendulum_v1"
    def isTrainedEnough(self):
        return self.avgReward > self.avgReward_toStopTrain

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
    eModule = import_module(f"EnvUtil_{envName.name}")                  
    envUtil = getattr(eModule, f"EnvUtil_{envName.name}")                 # class
        #   actionToEnv = getattr(eModule, f"ActionCoder_{envName.name}")()    # object
        #   observCoder = getattr(eModule, f"ObservCoder_{envName.name}")()    # object
        #   experienceFrom = getattr(eModule, "experienceFrom")                # function
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

    game = Game(config)
    game.run(mode, env, agent, envUtil)

    logger.info(f"total time={time.time() - before}")

