import numpy as np
import copy

# class for the n-player game
class Game:
    def __init__(self, no_players = 15, no_counters = 30):
        self.no_players = no_players
        self.no_counters = no_counters

        # choices = actions, counters = state
        self.choices = []
        self.counters = []

    # initialize and return initial state
    def initialize(self):
        for i in range(0, self.no_counters):
            self.counters.append(0)
        
        return copy.deepcopy(self.counters)

    # make an action in the environment
    def choose_counters(self, choices):
        for i in range(0, self.no_players):
            self.counters[choices[i] - 1] += 1

        return copy.deepcopy(self.counters)
    
    # return the reward (number of unique counter choices present in input state)
    def get_reward(self, state):
        reward = 0
        for i in state:
            if i == 1:
                reward += 1
        
        return reward
    
    def reset(self):
        self.counters = []
        for i in range(0, self.no_counters):
            self.counters.append(0)

    #def get_reward(self):

#if __name__ == "__main__":
#    game = Game()
#    initial_state = game.initialize()
#    next_state = game.choose_counters([1, 3, 23, 24, 6, 8, 27, 5, 30, 3, 11, 12, 1, 25, 3])
#
#    print(initial_state)
#    print(next_state) 
#
#    initial_reward = game.get_reward(initial_state)
#    subsequent_reward = game.get_reward(next_state)
#   
#    print(initial_reward)
#    print(subsequent_reward)
