
import tensorflow as tf
from analyzer import Analyzer

class Analyzer_daiso(Analyzer):
    def afterTimestep(self, reward, info):
        super().afterTimestep(reward, info)
        tf.summary.scalar("reward", reward, step=self.stepCnt)
        if self.envName in ["DaisoSokcho", "DaisoSokcho_discrete"]:
        with self.summaryWriter.as_default():
            tf.summary.scalar("cost_term", info['reward_terms']["cost_term"], step=self.stepCnt)
            tf.summary.scalar("temperature_term", info['reward_terms']["temperature_term"], step=self.stepCnt)
            tf.summary.scalar("consecutive_term", info['reward_terms']["consecutive_term"], step=self.stepCnt)
            tf.summary.scalar("n_AC_1F", info["action"][0], step=self.stepCnt)
            tf.summary.scalar("n_AC_2F", info["action"][1], step=self.stepCnt)
            tf.summary.scalar("n_AC_3F", info["action"][2], step=self.stepCnt)
            tf.summary.scalar("n_AC_4F", info["action"][3], step=self.stepCnt)
            tf.summary.scalar("n_AC_5F", info["action"][4], step=self.stepCnt)

