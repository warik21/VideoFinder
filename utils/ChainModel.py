# This file defines a class that represents a chain model. The chain model is essentially
# a langchain which we can "play with" in orderr to find the best fitting video for our
# prompt.

import os


class ChainModel:
    def __init__(self, api_key, channel):
        self.api_key = api_key
        self.channel = channel

    def prompt(self, prompt):
        pass

    def model(self):
        pass

    def parser(self):
        pass

