import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.2,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Relu layers, Dropout layers and a tanh layer.
        self.fc1 = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(512, 509),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.th = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh()
        )
        # ***********************************************************************

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.cat((x, input), 1)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.th(x)

        # ***********************************************************************


        return x
