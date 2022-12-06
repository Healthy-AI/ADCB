import os
import pickle as pkl


class Environment(object):
    """
    Abstract class specifying which methods environments have and what they return
    """

    def __init__(self, **kwargs):
        """
        Initializes the environment
        """
        pass

    def reset(self):
        """
        Resets the environment and returns an observation

        Returns:
            observation (object)
        """
        raise Exception('Reset not implemented')

    def step(self, action):
        """
        Plays an action, returns a reward and updates or terminates the environment

        Args:
            action: Played action

        Returns:
            observation (object): The observation following the action (None if terminal state reached).
            reward (float): The reward of the submitted action
            done (boolean): True if the environment has reached a terminal state
            info (dict): Returns e.g., the reward distribution of all actions so that regret can be computed
        """
        raise Exception('Stepping/acting not implemented')

    def save(self, path, overwrite=False, make_dirs=False):
        """
        Stores the environment to a binary
        """

        if not overwrite and os.path.isfile(path):
            raise Exception(
                'File at location %s already exists and overwrite is set to False' % path)

        if make_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        pkl.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        """
        Loads the environment from a binary
        """

        return pkl.load(open(path, 'rb'))
