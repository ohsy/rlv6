
import json
import random
import numpy as np
from collections import deque, namedtuple

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

class ReplayMemory:
    experienceTuple = namedtuple("experience", field_names=["observ","action","reward","next_observ","done"])

    def __init__(self, capacity=0, isRewardNorm=True, npDtype=np.float32):
        """
        experience is namedtuple ("experience", field_names=["observ","action","reward","next_observ","done"])
        where each field is an ndarray of numbers like float(observ, action, reward) or int(action or done).
        """
        self.capacity = capacity
        self.isRewardNorm = isRewardNorm
        self.npDtype = npDtype

        self.tiny = 1e-6
        self.buffer = deque(maxlen=self.capacity)
        self.memoryCnt = len(self.buffer)

    @classmethod
    def load(cls, path, capacity=0, isRewardNorm=True, npDtype=np.float32):
        mem = ReplayMemory(capacity, isRewardNorm, npDtype)
        with open(path, 'rt') as fp:
            experiences = [ 
                    (   np.array(ex[0], dtype=mem.npDtype), 
                        np.array(ex[1], dtype=mem.npDtype),
                        np.array(ex[2], dtype=mem.npDtype), 
                        np.array(ex[3], dtype=mem.npDtype), 
                        np.array(ex[4], dtype=mem.npDtype)
                    ) for ex in json.load(fp)
            ]  # json.load() returns a list of lists of lists
        mem.buffer = deque(experiences, maxlen=capacity)
        return mem

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        #   total_size = self.__len__()  # why?
        indices = np.random.choice(self.memoryCnt, size=batch_size)
        observs, actions, rewards, next_observs, done = zip(*[self.buffer[ix] for ix in indices]) # tuple of ndarrays
        experiences = self.experienceTuple(
                np.array(observs, dtype=self.npDtype), 
                np.array(actions, dtype=self.npDtype), 
                np.array(rewards, dtype=self.npDtype), 
                np.array(next_observs, dtype=self.npDtype), 
                np.array(done, dtype=self.npDtype)) 
        return experiences, indices, None  # TEMP: None for importance_weights

    def remember(self, experience):
        self.buffer.append(experience)
        self.memoryCnt = len(self.buffer)

    def save(self, path):
        """ dump self.buffer as json file.
            field as a list. experience as a list of lists.  buffer as a list of lists of lists.  
        """
        with open(path, 'wt') as fp:
            json.dump(list(self.buffer), fp, cls=NpEncoder) 

class PERMemory(ReplayMemory):
    def __init__(self, mode, capacity, isRewardNorm, npDtype):
        super().__init__(mode, capacity, isRewardNorm, npDtype)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # NOTE if alpha = 0, then uniform case else if alpha = 1, then greedy prioritized case
        self.beta = 0.4
        self.beta_annealing_step = 0.001

    def sample(self, batch_size):
        total_size = self.__len__()
        # Prioirity probability
        probabilities = np.array(self.priorities, self.npDtype) ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        # Sample experiences
        indices = np.random.choice(total_size, size=batch_size, p=probabilities)
        observs, actions, rewards, next_observs, done = zip(*[self.buffer[ix] for ix in indices])
        # Reward normalization
        if self.isRewardNorm:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + self.tiny)
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

    def update_priorities(self, indices, td_errors):
        updated_priorities = np.abs(td_errors.numpy().reshape(-1)) + self.tiny
        priorities = np.array(self.priorities, self.npDtype)
        priorities[indices] = updated_priorities
        self.priorities = deque(priorities, maxlen=self.capacity)
        # for ix, priority in zip(indices, updated_priorities):
        #     self.priorities[ix] = priority

