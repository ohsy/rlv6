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
from coder import Coder
from analyzer import Analyzer
from daiso_env.stateobservation_DIS import ObservationObj
from daiso_env.actioncontrol_DIS import ControlObj

from daiso_env.environment import DaisoSokcho

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
class AgentName(Enum):
    DQN = 'DQN'
    DDPG = 'DDPG'
    SAC = 'SAC'
    SAC_discrete = 'SAC_discrete'
    SAC_entropy = 'SAC_entropy'
    SAC_ec = 'SAC_ec'  # entropy_continuous


class Game:
    def __init__(self, config, logger, summaryWriter):
        self.config = config
        self.period_toSaveModels = config["Period_toSaveModels"]
        with open(os.getcwd() + "/daiso_env/config.json") as f:
            daiso_config = json.load(f)
        self.obsObj = ObservationObj(daiso_config)
        self.crtlObj = ControlObj(daiso_config)
        self.logger = logger
        self.summaryWriter = summaryWriter

    def run(self, nEpisodes, mode, env, agent, coder, analyzer):
        analyzer.beforeMainLoop()
        stepCnt = 0
        for episodeCnt in range(1, nEpisodes+1):  # for episodeCnt in tqdm(range(1, nEpisodes+1)):
            analyzer.beforeEpisode()
            cost_terms, temp_terms, cons_terms = [], [], []
            observFrEnv, info = env.reset(mode.value)  # observFrEnv (observDim)
            while True:
                observ = np.array(list(self.obsObj.observation_fromState(observFrEnv).get_values()))    # (observDim)
            
                action = agent.act(observ, coder.actionCoder)     # actionCoder to get random action to explore; (actionDim)

                actionToEnv = coder.actionCoder.decode(action)    # scalar for Discrete, (actionToEnv.nParameters) for Box 

                next_observFrEnv, reward, terminated, truncated, timestepInfo = env.step(actionToEnv)
                stepCnt += 1
                done = (terminated or truncated)  # bool
                cost_terms.append(timestepInfo['reward_terms']["cost_term"])
                temp_terms.append(timestepInfo['reward_terms']["temperature_term"])
                cons_terms.append(timestepInfo['reward_terms']["consecutive_term"])

                # experience = coder.experienceFrom(observFrEnv, actionToEnv, reward, next_observFrEnv, done, agent.npDtype)
                next_observ = np.array(list(self.obsObj.observation_fromState(next_observFrEnv).get_values()))    # (observDim)
                experience = (
                        np.array(observ, dtype=agent.npDtype),        # (observDim)
                        np.array(action, dtype=agent.npDtype),        # (actionDim)
                        np.array([reward], dtype=agent.npDtype),      # scalar to ndarray of dtype
                        np.array(next_observ, dtype=agent.npDtype),   # (observDim)
                        np.array([done], dtype=agent.npDtype)         # bool to ndarray of dtype
                )  
                agent.replayBuffer.remember(experience)

                self.logger.info(f"=================")
                self.logger.info(f"observFrEnv={observFrEnv}")
                self.logger.info(f"actionToEnv={actionToEnv}")
                self.logger.info(f"next_observFrEnv={next_observFrEnv}")
                self.logger.info(f"observ={observ}")
                self.logger.info(f"action={action}")
                self.logger.info(f"reward={reward}")
                self.logger.info(f"cost_term={timestepInfo['reward_terms']['cost_term']}")
                self.logger.info(f"temp_term={timestepInfo['reward_terms']['temperature_term']}")
                self.logger.info(f"cons_term={timestepInfo['reward_terms']['consecutive_term']}")
                self.logger.info(f"next_observ={next_observ}")
 
                if agent.isReadyToTrain():
                    batch, indices, importance_weights = agent.replayBuffer.sample(agent.batchSz)
                    #   print(f"batch=\n{batch}")
                    loss, td_error = agent.train(batch, importance_weights)

                    if agent.isPER == True:
                        agent.replayBuffer.update_priorities(indices, td_error)
                    analyzer.afterTrain(loss, agent)

                analyzer.afterTimestep(reward)
                with self.summaryWriter.as_default():
                    tf.summary.scalar("reward", reward, step=stepCnt)
                    tf.summary.scalar("cost_term", timestepInfo['reward_terms']["cost_term"], step=stepCnt)
                    tf.summary.scalar("temp_term", timestepInfo['reward_terms']["temperature_term"], step=stepCnt)
                    tf.summary.scalar("cons_term", timestepInfo['reward_terms']["consecutive_term"], step=stepCnt)

                observFrEnv = next_observFrEnv
                if done:
                    break

            analyzer.afterEpisode(episodeCnt, agent)
            avg_cost = sum(cost_terms) / len(cost_terms)
            avg_temp = sum(temp_terms) / len(temp_terms)
            avg_cons = sum(cons_terms) / len(cons_terms)
            with self.summaryWriter.as_default():
                tf.summary.scalar("avg_cost_term", avg_cost, step=episodeCnt)
                tf.summary.scalar("avg_temp_term", avg_temp, step=episodeCnt)
                tf.summary.scalar("avg_cons_term", avg_cons, step=episodeCnt)

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
    logger.setLevel(logging.INFO) #   INFO, DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileHandler = logging.FileHandler(filename=filepath, mode="w")
    fileHandler.setLevel(logging.INFO) # INFO, DEBUG
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
    if envName == EnvName.DaisoSokcho:
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

    game = Game(config, logger, summaryWriter)
    game.run(nEpisodes, mode, env, agent, coder, analyzer)

