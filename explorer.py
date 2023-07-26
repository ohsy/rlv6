
import math
import random

class Explorer_epsilonDecay:
    def __init__(self, mode, config, savePath, replayMemory=None):  # replayMemory dummy
        self.mode = mode
        self.config = config
        self.savePath = savePath
        self.replayMemory = replayMemory  # only to get capacity

        self.filePath = f"{self.savePath}/epsilonDecay.txt"
        self.MemoryRatio_toStartTrain = 0.001
        self.epsilonInit = config["EpsilonInit"]
        self.epsilonMin = config["EpsilonMin"]
        self.epsilonLambda = config["EpsilonLambda"]
        self.epsilon = self.epsilonInit
        self.epsilonDecayCnt = 0 
            #   self.epsilonDecay = (self.epsilonInit - self.epsilonMin) / self.epsilonMaxActCount
        self.epsilonDecay = config["EpsilonDecay"]  
            # so that epsilonDecay**(repalyMemory.capacity/2) ~ epsilonMin; ex. 0.999**4600 ~ 0.01

        if mode == "continued_train":
            self.load()
        
    def isReadyToExplore(self):
        self.decay_epsilon()
        b = (self.mode in ["train", "continued_train"]) and random.random() < self.epsilon
        return b

    def decay_epsilon(self):
            #   self.epsilon = max(self.epsilon * self.epsilonDecay, self.epsilonMin) 
        self.epsilon = self.epsilonMin \
                + (self.epsilonInit - self.epsilonMin) * math.exp(-self.epsilonLambda * self.epsilonDecayCnt)
        self.epsilonDecayCnt += 1

    def save(self):
        with open(self.filePath, "wt") as f:
            f.write(f"epsilon={self.epsilon}\n")
            f.write(f"epsilonDecayCnt={self.epsilonDecayCnt}\n")

    def load(self):
        with open(self.filePath, "rt") as f:
            lst = f.readline().split('=')
            self.epsilon = float(lst[1])
            lst = f.readline().split('=')
            self.epsilonDecayCnt = int(lst[1])

    def get_memoryCnt_toStartTrain(self):
        memoryRatio_toStartTrain = 0.001
        memoryCnt_toStartTrain = int(memoryRatio_toStartTrain * self.replayMemory.capacity)
        return memoryCnt_toStartTrain


class Explorer_replayMemoryFiller:
    def __init__(self, mode, config, savePath, replayMemory):
        self.mode = mode
        self.savePath = savePath

        self.replayMemory = replayMemory  # only to get memoryCnt
        self.memoryCnt_toFillWithRandomAction = int(config["MemoryRatio_toFillWithRandomAction"] * replayMemory.capacity)

    def isReadyToExplore(self):
        b1 = self.mode in ["train", "continued_train"]
        b2 = self.replayMemory.memoryCnt < self.memoryCnt_toFillWithRandomAction
        return b1 and b2

    def save(self):
        pass
    def load(self):
        pass

    def get_memoryCnt_toStartTrain(self):
       return self.memoryCnt_toFillWithRandomAction 

