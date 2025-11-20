from enum import Enum

class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

class HeadDirection(Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

