
import tensorflow as tf
from analyzer import Analyzer

class Analyzer_daiso(Analyzer):
    def afterTimestep(self, reward, info):
        super().afterTimestep(reward, info)
        tf.summary.scalar("reward", reward, step=self.stepCnt)
        with self.summaryWriter.as_default():
            tf.summary.scalar("cost_term", info['reward_terms']["cost_term"], step=self.stepCnt)
            tf.summary.scalar("temperature_term", info['reward_terms']["temperature_term"], step=self.stepCnt)
            tf.summary.scalar("consecutive_term", info['reward_terms']["consecutive_term"], step=self.stepCnt)

