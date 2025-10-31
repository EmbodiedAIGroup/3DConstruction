from enum import Enum

class RobotAction(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

    def __str__(self):
        return self.name