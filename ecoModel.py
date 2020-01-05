import torch
import torch.nn as nn
import torch.optim as optim
import math
import ecoUtils
import ecoEnv
import random
import numpy as np


class Trainer():
    def __init__(self):

        # Nomenclature
        # Global - multiple complete code runs
        #   Session - a complete run of the code
        #       Cycle
        #           Training Frame
        #           Episode
        #               Validation

        # Hyper Parameters
        self.gamma = 0.9
        self.minibatch_size = 512
        self.initial_weights_setting = 0.1
        self.learning_rate = 0.0001
        self.grad_clipping = False
        self.starting_epsilon = 0.1
        self.epsilon_decay = 0.00001

        # Training Process Parameters
        self.frames_per_cycle = 2000 # the number of frames for training in a session
        self.episodes_per_validation = 500 # the number of game sessions to complete during validation testing

        # Other Shared variables
        self.cycle_frame = 1 # the current frame of the cycle
        self.observation_width = 1000
        self.action_count = 19682
        self.model = Net(observation_width=self.observation_width, action_count=self.action_count)
        self.replay_memory_size = 1000
        self.replay_memory = ecoUtils.readMemory('Memory')

        # Not Used by Trainer (required for Orchestrator)
        # self.global_frame = global_frame_start  # the number of frames used in training so far
        # self.global_cycle = global_cycle_start  # the total number of training cycles across all sessions
        # self.cycles_per_session = cycles_per_session  # the number of training cycles to perform in a session
        # self.session_frame = 0  # the training frame in the current session
        # self.session_cycle = 0  # the number of training cycles in the current session

    def epsilon(self):
        # TODO Each training run needs to be aware of the starting global frame and additional frames\
        #  so it can calculate epsilon
        return self.starting_epsilon / (math.exp(self.global_frame * self.epsilon_decay))

    def init_weights(self):
        if type(self.model) == nn.Conv2d or type(self.model) == nn.Linear:
            torch.nn.init.uniform_(self.model.weight, -1 * self.initial_weights_setting, self.initial_weights_setting)
            self.model.bias.data.fill_(self.initial_weights_setting)

    def train(self):
        self.model.train()

        losses = []

        # define Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize mean squared error loss
        criterion = nn.MSELoss()

        # instantiate game
        market = ecoEnv.market(None)
        entityCount = market.entityCount()

        # Get state for each entity's perspective
        state_0 = market.getStateExternal(self.observation_width)
        state_0_tensor = self.convertState(state_0)

        while self.cycle_frame <= self.frames_per_cycle:

            # Forward pass on model
            rewards_pred_0 = self.model(state_0_tensor)

            # Loop through each entity's predicted reward and build an action vector.  Add it to a list.
            # action_0 expected to be list of tuples
            #   First value is the EntityID
            #   Second is integer between 0 and 19,682

            action_0 = []
            entID = 0

            for reward_pred_item in rewards_pred_0:

                # epsilon greedy exploration
                random_action = random.random() <= self.epsilon()

                if random_action:
                    action = (entID,random.randint(0,self.model.number_of_actions))
                else:
                    action = (entID,torch.argmax(reward_pred_item)[0])

                action_0.append(action)

                entID += 1

            # Get next reward and termination
            rewards_1, terminal_1 = market.takeActionsExternal(action_0)

            # Get next state for each entity's perspective
            state_1 = market.getStateExternal(self.observation_width)

            # save transition to replay memory
            self.replay_memory.append((state_0, action_0, rewards_1, state_1, terminal_1))

            # if replay memory is full, remove the oldest transition
            if len(self.replay_memory) > self.replay_memory_size:
                self.replay_memory.pop(0)

            # sample random minibatch
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

            # unpack minibatch
            state_0_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_0_batch = torch.cat(tuple(self.makeOneHotAction(d[1]) for d in minibatch))
            reward_1_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            state_1_batch_tensor = self.convertState(state_1_batch)

            # get output for the next state
            reward_pred_1_batch = self.model(state_1_batch_tensor)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_1_batch[i] if minibatch[i][4]
                                      else reward_1_batch[i] + self.gamma * torch.max(reward_pred_1_batch[i])
                                      for i in range(len(minibatch))))

            # extract Q-value
            q_value = torch.sum(self.model(state_0_batch) * action_0_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            # From blog I read
            loss = criterion(q_value, y_batch)

            # do backward pass
            loss.backward()

            # gradient clipping
            if self.grad_clipping:
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)

            optimizer.step()

            # If last state was terminal then reset environment and get state
            # else set state to previous state
            if terminal_1:
                market = ecoEnv.market(None)
                state_0 = market.getStateExternal(self.observation_width)
            else:
                state_0 = state_1

            self.cycle_frame = self.cycle_frame + 1

        losses = np.array(losses)
        avgLoss = losses.mean()

        return avgLoss

    def convertState(self,state_list):
        output = torch.tensor(state_list).unsqueeze(0)
        output = output.float()
        return output

    def makeOneHotAction(self,action):
        output = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
        output[action] = 1
        return output

class Net(nn.Module):

    # This model takes an observation of the environment and predicts a reward for each possible action

    def __init__(self,observation_width,action_count):
        super(Net, self).__init__()

        span = action_count - observation_width

        connect_1_2 = int((span*0.25+observation_width))
        connect_2_3 = int((span*0.50+observation_width))
        connect_3_4 = int((span*0.75+observation_width))

        self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(in_features=connect_3_4, out_features=action_count)

    def forward(self, x):

        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)

        return out