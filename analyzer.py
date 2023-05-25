
import time
from collections import deque

class Analyzer:
    status_train = "train"
    status_preTrain = "preTr"  # preparing train

    def __init__(self, config, logger):
        self.logger = logger
        self.sumReward_toStopTrain = config["SumReward_toStopTrain"]
        self.avgReward_toStopTrain = config["AvgReward_toStopTrain"]
        self.status = self.status_preTrain  
        self.preStatus = None
        self.rewards = []
        self.sumReward = 0
        self.sumReward_max = 0
        self.sumReward_recent_maxlen = 3
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
        if self.envName == "CartPole_v1":
            msg += f", sumReward: {self.sumReward:.3f}, sumReward_max: {self.sumReward_max:.3f}" 
        elif self.envName == "Pendulum_v1":
            msg += f", avgReward: {self.avgReward:.3f}" 
        msg += f", epsilon: {agent.explorer.epsilon:.3f}" if hasattr(agent.explorer,"epsilon") else ""
        self.logger.info(msg)
        agent.summaryWrite("sumReward", self.sumReward, step=episodeCnt)
        agent.summaryWrite("avgReward", self.avgReward, step=episodeCnt)

    def isTrainedEnough(self):
        pass

class Analyzer_CartPole_v1(Analyzer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.envName = "CartPole_v1"
    def isTrainedEnough(self):
        self.sumReward_recent_mean = sum(self.sumReward_recent) / self.sumReward_recent_maxlen
        return self.sumReward_recent_mean > self.sumReward_toStopTrain

class Analyzer_Pendulum_v1(Analyzer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.envName = "Pendulum_v1"
    def isTrainedEnough(self):
        return self.avgReward > self.avgReward_toStopTrain

