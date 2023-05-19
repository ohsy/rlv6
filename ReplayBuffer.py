
import json
import random
import numpy as np
from collections import deque, namedtuple

import tensorflow as tf
# import tensorflow_probability as tfp
# from ..utils.bcolor import *

class NpEncoder(json.JSONEncoder):  
    """ https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class NpDecoder(json.JSONDecoder):   # TEMP
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ReplayBuffer:
    def __init__(self, config, npDtype, tfDtype, npIntDtype):
        """
        experience is namedtuple ("experience", field_names=["observ", "action", "reward", "next_observ", "done"])
        where each field is an ndarray of numbers like float(observ, action, reward) or int(action or done).
        """
        self.config = config
        self.npDtype = npDtype
        self.npIntDtype = npIntDtype
        self.tfDtype = tfDtype
        self.mode = self.config["Mode"]
        self.capacity = self.config["MemoryCapacity"]
        #   self.filePath = f"{self.config['SavePath']}/replayBuffer.json"

        self.eps = 1e-6
        if 'continued_train' in self.mode:
            self.buffer = self.load()
        else:
            self.buffer = deque(maxlen=self.capacity)
        self.memoryCnt = len(self.buffer)
        self.experienceTuple = namedtuple("experience", field_names=["observ", "action", "reward", "next_observ", "done"])

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        #   total_size = self.__len__()  # why?
        indices = np.random.choice(self.memoryCnt, size=batch_size)
        observs, actions, rewards, next_observs, done = zip(*[self.buffer[ix] for ix in indices]) # each returned: tuple of ndarrays
        #   print("in sample:")
        #   printtype(observs)
        #   printtype(actions)
        #   printtype(rewards)
        #   printtype(next_observs)
        #   printtype(done)
        experiences = self.experienceTuple(
                np.array(observs), np.array(actions), np.array(rewards), np.array(next_observs), np.array(done)  
        ) # each-argument.reshape(batch_size, -1),  # reshape not needed
        # print(f"type(experiences)={type(experiences)}\nexperiences={experiences}")
        return experiences, indices, None  # TEMP: None for importance_weights

    def remember(self, experience):
        self.buffer.append(experience)
        self.memoryCnt = len(self.buffer)

    def save(self, filePath):
        """ field as a list. experience as a list of lists.  buffer as a list of lists of lists.  """
        self.filePath = filePath
        with open(f"{filePath}", 'wt') as fp:
            json.dump(list(self.buffer), fp, cls=NpEncoder) 

    def load(self):
        with open(f"{self.filePath}", 'rt') as fp:
            experiences = [ 
                    (   np.array(ex[0], dtype=self.npDtype), 
                        #   np.array(ex[1], dtype=self.npIntDtype),
                        np.array(ex[1], dtype=self.npDtype),
                        np.array(ex[2], dtype=self.npDtype), 
                        np.array(ex[3], dtype=self.npDtype), 
                        np.array(ex[4], dtype=self.npDtype)
                    ) for ex in json.load(fp)
            ]  # json.load() returns a list of lists of lists
            buffer = deque(experiences, maxlen=self.capacity)
        return buffer

class PERBuffer(ReplayBuffer):
    def __init__(self, config, npDtype, tfDtype):
        super().__init__(config, npDtype, tfDtype)
        self.priorities = deque(maxlen=config["MemoryCapacity"])
        self.alpha = 0.6  # NOTE if alpha = 0, then uniform case else if alpha = 1, then greedy prioritized case
        self.beta = 0.4
        self.beta_annealing_step = 0.001

    # @tf.function
    def sample(self, batch_size):
        total_size = self.__len__()
        # Prioirity probability
        probabilities = np.array(self.priorities, self.npDtype) ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        # Sample experiences
        indices = np.random.choice(total_size, size=batch_size, p=probabilities)
        observs, actions, rewards, next_observs, done = zip(*[self.buffer[ix] for ix in indices])
        # Reward normalization
        if self.config["RewardNormalization"]:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + self.eps)
        experiences = self.experienceTuple(
            np.array(observs, dtype=self.npDtype).reshape(batch_size, -1),
            np.array(actions, dtype=self.npDtype).reshape(batch_size, -1),
            np.array(rewards, dtype=self.npDtype).reshape(batch_size, -1),
            np.array(next_observs, dtype=self.npDtype).reshape(batch_size, -1),
            np.array(done, dtype=self.npDtype).reshape(batch_size, -1),
        )
        # Importance weights
        weights = (total_size * probabilities) ** (-self.beta)
        # weights /= max(weights)
        weights = weights.astype(self.npDtype)
        # beta annealing
        self.beta = min(1.0, self.beta + self.beta_annealing_step)
        return experiences, indices, weights

    def remember(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1.0))
        self.memoryCnt = len(self.buffer)

    # @tf.function
    def update_priorities(self, indices, td_errors):
        updated_priorities = np.abs(td_errors.numpy().reshape(-1)) + self.eps
        priorities = np.array(self.priorities, self.npDtype)
        priorities[indices] = updated_priorities
        self.priorities = deque(priorities, maxlen=self.config["MemoryCapacity"])
        # for ix, priority in zip(indices, updated_priorities):
        #     self.priorities[ix] = priority

