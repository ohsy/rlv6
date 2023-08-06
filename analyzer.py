
import time
from statistics import mean
from collections import deque
import tensorflow as tf

from enum import Enum
class TargetToMonitor(Enum):
    sumReward = 'sumReward'
    avgReward = 'avgReward'

class Analyzer:
    status_train = "train"
    status_preTrain = "preTr"       # preparing train
    targetToMonitor: TargetToMonitor 

    def __init__(self, envName, config, logger, summaryWriter):
        self.envName = envName
        self.logger = logger
        self.summaryWriter = summaryWriter
        self.movingAvgWindowSz = config["MovingAverageWindowSize"]
        self.status = self.status_preTrain  
        self.preStatus = None
        self.rewards = []
        self.sumReward = 0
        self.sumReward_recent = deque(maxlen=self.movingAvgWindowSz)    
        self.avgReward = 0
        self.avgReward_recent = deque(maxlen=self.movingAvgWindowSz)    
        self.losss0 = []
        self.losss1 = []
        self.stepCnt = 0
        self.trainCnt = 0
        self.timeBeforeEpisode = None

        env_config = config[self.envName]
        self.targetToMonitor = TargetToMonitor(env_config["TargetToMonitor"])
        self.sumReward_toStopTrain = env_config["sumReward_toStopTrain"]
        self.avgReward_toStopTrain = env_config["avgReward_toStopTrain"]
        self.timeBeforeMainLoop = 0  # used for time.time()

    def beforeMainLoop(self):
        self.timeBeforeMainLoop = time.time()

    def timeForMainLoop(self):
        return time.time() - self.timeBeforeMainLoop

    def afterMainLoop(self):
        self.logger.info(f"time for main loop = {self.timeForMainLoop():.3f}sec")

    def beforeEpisode(self):
        self.rewards = []
        self.losss0 = []
        self.losss1 = []
        self.timeBeforeEpisode = time.time()

    def afterTrain(self, loss, agent):
        if type(loss) is tuple:
            self.losss0.append(loss[0])
            self.losss1.append(loss[1])
            with self.summaryWriter.as_default():
                tf.summary.scalar("loss0", loss[0], step=self.trainCnt)
                tf.summary.scalar("loss1", loss[1], step=self.trainCnt)
        else:
            self.losss0.append(loss)
            with self.summaryWriter.as_default():
                tf.summary.scalar("loss", loss, step=self.trainCnt)
        #   tf.summary.scalar("alpha_loss", alpha_loss, step=trainCnt)
        #   tf.summary.scalar("alpha", agent.alpha, step=trainCnt)
        self.trainCnt += 1

    def afterTimestep(self, reward, info):
        self.rewards.append(reward)
        if self.envName in ["DaisoSokcho", "DaisoSokcho_discrete"]:
            with self.summaryWriter.as_default():
                tf.summary.scalar("n_AC_1F", info.action[0], step=self.trainCnt)
                tf.summary.scalar("n_AC_2F", info.action[1], step=self.trainCnt)
                tf.summary.scalar("n_AC_3F", info.action[2], step=self.trainCnt)
                tf.summary.scalar("n_AC_4F", info.action[3], step=self.trainCnt)
                tf.summary.scalar("n_AC_5F", info.action[4], step=self.trainCnt)
        self.stepCnt += 1

    def afterEpisode(self, episodeCnt, agent):
        self.preStatus = self.status
        self.status = self.status_train if agent.isReadyToTrain() or self.preStatus == self.status_train else self.status_preTrain
        tm = time.time() - self.timeBeforeEpisode
        if self.preStatus != self.status_train and self.status == self.status_train:  # only once
            agent.summary()

        avg_loss0 = sum(self.losss0) / len(self.losss0) if len(self.losss0) > 0 else 0
        avg_loss1 = sum(self.losss1) / len(self.losss1) if len(self.losss1) > 0 else 0
        self.sumReward = sum(self.rewards)  # NOTE: sumReward != return due to gamma 
        self.sumReward_recent.append(self.sumReward)
        self.sumReward_movingAvg = mean(self.sumReward_recent)
        self.avgReward = self.sumReward / len(self.rewards)  # NOTE: sumReward != return due to gamma 
        self.avgReward_recent.append(self.avgReward)
        msg = f"({self.status}) episode {episodeCnt}: {tm:.3f}sec" 
        msg += f", avg_loss0={avg_loss0:.3f}"  # critic if actor-critic
        msg += f", avg_loss1={avg_loss1:.3f}"  # actor if actor-critic
        if self.targetToMonitor == TargetToMonitor.sumReward:
            msg += f", sumReward={self.sumReward:.3f}, sumReward_movingAvg={self.sumReward_movingAvg:.3f}"
            with self.summaryWriter.as_default():
                tf.summary.scalar("sumReward", self.sumReward, step=episodeCnt)
        elif self.targetToMonitor == TargetToMonitor.avgReward:
            msg += f", avgReward={self.avgReward:.3f}, avgReward_movingAvg={self.avgReward_movingAvg:.3f}"
            with self.summaryWriter.as_default():
                tf.summary.scalar("avgReward", self.avgReward, step=episodeCnt)
        msg += f", epsilon={agent.explorer.epsilon:.3f}" if hasattr(agent.explorer,"epsilon") else ""
        self.logger.info(msg)

    def afterSave(self, msg):
        self.logger.info(msg + f", time for main loop = {self.timeForMainLoop():.3f}sec")

    def isTrainedEnough(self):
        if self.targetToMonitor == TargetToMonitor.sumReward:
            return self.sumReward_movingAvg > self.sumReward_toStopTrain
        elif self.targetToMonitor == TargetToMonitor.avgReward:
            return self.avgReward_movingAvg > self.avgReward_toStopTrain
        else:
            return False

