import numpy as np
import pystk
import math

forward = 1

def to_numpy(location):
    return np.float32([location[0], location[2]])


def quaternion_to_euler(x, y, z, w):
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    return Y


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        pass
        
    def act(self, image, player_info):
        world = pystk.WorldState()
        world.update()
        kart2 = world.karts[3]
        pos_ball = to_numpy(world.soccer.ball.location)
        pos_me2 = to_numpy(kart2.location)
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        # pos_me[1] is up and down
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        angle_kart2 = quaternion_to_euler(kart2.rotation[0],kart2.rotation[1],kart2.rotation[2],kart2.rotation[3])
        angle_puck = math.degrees(math.atan((pos_ball[0]-pos_me2[0])/(pos_ball[1]-pos_me2[1])))
        # print(pos_me2[1])
        # print(pos_ball[0]-pos_me[0])
        # print(pos_ball[1]-pos_me[1])
        # print(pos_ball[1], pos_me[1], pos_ball[0], pos_me[0])
        # print(angle_puck)
        # print(angle_kart)
        # print(pos_ball[1])
        # print(pos_me[0])
        global forward
        if pos_ball[1] - pos_me2[1] > 10 and abs(angle_kart2 - angle_puck) < 20:
            forward = 1
        if pos_ball[1] - pos_me2[1] > 15 and abs(angle_kart2 - angle_puck) < 40:
            forward = 1
        if pos_ball[1] - pos_me2[1] > 25 or abs(angle_kart2 - angle_puck) < 15:
            forward = 1
        if pos_ball[1] - pos_me2[1] > 0 and angle_kart2 < angle_puck and forward == 1:
            action['steer'] = 1
            action['acceleration'] = 1
        if pos_ball[1] - pos_me2[1] > 0 and angle_kart2 > angle_puck and forward == 1:
            action['steer'] = -1
            action['acceleration'] = 1
        if pos_ball[1] - pos_me2[1] < 0 and angle_kart2 > 0:
            forward = 0
            action['steer'] = 1
            action['brake'] = True
            action['acceleration'] = 0
        if pos_ball[1] - pos_me2[1] < 0 and angle_kart2 < 0:
            forward = 0
            action['steer'] = -1
            action['brake'] = True
            action['acceleration'] = 0
        if forward == 0 and angle_kart2 > 0:
            action['steer'] = 1
            action['brake'] = True
            action['acceleration'] = 0
        if forward == 0 and angle_kart2 < 0:
            action['steer'] = -1
            action['brake'] = True
            action['acceleration'] = 0
        # if world.karts[2].velocity == 0 and angle_kart2 > 0:
        #     action['steer'] = -1
        #     action['brake'] = True
        #     action['acceleration'] = 0
        # if world.karts[2].velocity == 0 and angle_kart2 < 0:
        #     action['steer'] = 1
        #     action['brake'] = True
        #     action['acceleration'] = 0
        # if angle_puck > 80:
        #     action['steer'] = 1
        #     action['brake'] = True
        #     action['acceleration'] = 0
        # if angle_puck < -80:
        #     action['steer'] = -1
        #     action['brake'] = True
        #     action['acceleration'] = 0
        """
        Your code here.
        """

        return action

