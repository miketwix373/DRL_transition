
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 1
        #n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 1):
        super().__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=1, padding=1)

        self.fc1 = nn.Linear(48*32, 320)
        self.fc2 = nn.Linear(320, 160)
        self.out = nn.Linear(160, 1)


    def forward(self, x): 
        self.f_act   = open('~/sharedscratch/img_CaNS_DRL/run/shapes.dat','w')
        self.f_act.write(str(x.shape))
        self.f_act.close()
        x = self.conv1(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)        
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) 
        x = self.conv3(x)
        self.f_act   = open('~/sharedscratch/img_CaNS_DRL/run/x_shapes.dat','w')
        self.f_act.write(str(x.shape))
        self.f_act.write(x)
        self.f_act.close()
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = x.reshape(-1, 48*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return F.softmax(x, dim=1)
'''


