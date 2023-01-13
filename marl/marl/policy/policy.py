import torch
import torch.nn as nn
import numpy as np
from marl.tools import gymSpace2dim,get_combinatorial_actions

from marl.tools import ClassSpec, _std_repr

class Policy(object):
    policy = {}
    
    def __init__(self, action_space,all_actions):
        self.action_space = action_space
        self.all_actions = all_actions

    def __call__(self, state,num_actions=1):
        raise NotImplementedError
    
    def __repr__(self):
        return _std_repr(self)
    
    def random_action(self, observation=None,num_actions=1):
        #if len(observation.shape) < 3:
        if num_actions == 1:
            return self.all_actions[self.action_space.sample()]
        else:
            actions = []
            for i in range(num_actions):    
                a = self.action_space.sample()
                while a in actions:
                    a = self.action_space.sample()
                actions.append(self.all_actions[a])
            return actions
        # else:
        #     actions_list = []
        #     for j in range(observation.shape[0]):
        #         actions = []
        #         for i in range(num_actions):    
        #             a = self.action_space.sample()
        #             while a in actions:
        #                 a = self.action_space.sample()
        #             actions.append(self.all_actions[a])
        #         actions_list.append(actions)
        #     return actions_list

    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Policy.policy[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Policy.policy.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Policy.policy[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Policy.policy.keys()
    
class ModelBasedPolicy(Policy):
    
    def __init__(self, model):
        self.model = model
    
    def load(self, filename):
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load(filename=filename)

    def save(self, filename):
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), filename)
        else:
            self.model.save(filename=filename)

def register(id, entry_point, **kwargs):
    Policy.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Policy.make(id, **kwargs)
    
def available():
    return Policy.available()