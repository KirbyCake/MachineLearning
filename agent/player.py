import numpy as np
import torch
import torch.nn.functional as F
import pystk
import math


class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size=3, stride=2):
        super().__init__()
        self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
        self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
        self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
        self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

    def forward(self, x):
        return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))) + self.skip(x)


class Planner(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16], n_output_channels=6, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)


    def forward(self, img):
        z = (img - self.input_mean[None, :, None, None].to(img.device)) / self.input_std[None, :, None, None].to(img.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        img = self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


def to_numpy(location):
    return np.float32([location[0], location[2]])


def quaternion_to_euler(x, y, z, w):
    t = +2.0 * (w * y - z * x)
    t = +1.0 if t > +1.0 else
    t = -1.0 if t < -1.0 else t
    Y = math.degrees(math.asin(t))

    return Y

forward = 1

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
        kart0 = world.karts[2]
        pos_ball = to_numpy(world.soccer.ball.location)
        pos_me1 = to_numpy(kart0.location)
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        # pos_me[1] is up and down
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        angle_kart1 = quaternion_to_euler(kart0.rotation[0],kart0.rotation[1],kart0.rotation[2],kart0.rotation[3])
        angle_puck = math.degrees(math.atan((pos_ball[0]-pos_me1[0])/(pos_ball[1]-pos_me1[1])))
        # print(pos_ball[0]-pos_me[0])
        # print(pos_ball[1]-pos_me[1])
        # print(pos_ball[1], pos_me[1], pos_ball[0], pos_me[0])
        # print(angle_puck)
        # print(angle_kart)
        # print(pos_me1[1])
        global forward
        if pos_ball[1] - pos_me1[1] > 10 and abs(angle_kart1 - angle_puck) < 20:
            forward = 1
        if pos_ball[1] - pos_me1[1] > 15 and abs(angle_kart1 - angle_puck) < 40:
            forward = 1
        if pos_ball[1] - pos_me1[1] > 25 or abs(angle_kart1 - angle_puck) < 15:
            forward = 1
        if pos_ball[1] - pos_me1[1] > 0 and angle_kart1 < angle_puck and forward == 1:
            action['steer'] = 1
            action['acceleration'] = 1
        if pos_ball[1] - pos_me1[1] > 0 and angle_kart1 > angle_puck and forward == 1:
            action['steer'] = -1
            action['acceleration'] = 1
        if pos_ball[1] - pos_me1[1] < 0 and angle_kart1 > 0:
            forward = 0
            action['steer'] = 1
            action['brake'] = True
            action['acceleration'] = 0
        if pos_ball[1] - pos_me1[1] < 0 and angle_kart1 < 0:
            forward = 0
            action['steer'] = -1
            action['brake'] = True
            action['acceleration'] = 0
        if forward == 0 and angle_kart1 > 0:
            action['steer'] = 1
            action['brake'] = True
            action['acceleration'] = 0
        if forward == 0 and angle_kart1 < 0:
            action['steer'] = -1
            action['brake'] = True
            action['acceleration'] = 0
        # if angle_kart1 > 80:
        #     action['acceleration'] = .7
        # if angle_kart1 < -80:
        #     action['acceleration'] = .7
        # if world.karts[0].velocity == 0 and angle_kart1 < 0:
        #     action['steer'] = -1
        #     action['brake'] = True
        #     action['acceleration'] = 0
        # if world.karts[0].velocity == 0 and angle_kart1 > 0:
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

