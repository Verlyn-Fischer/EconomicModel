import ecoModel

class Orchestrator():
    def __init__(self,global_frame_start, global_cycle_start, cycles_per_session):

        self.global_frame = global_frame_start  # the number of frames used in training so far
        self.global_cycle = global_cycle_start  # the total number of training cycles across all sessions
        self.cycles_per_session = cycles_per_session  # the number of training cycles to perform in a session
        self.session_frame = 0  # the training frame in the current session
        self.session_cycle = 0  # the number of training cycles in the current session

    def learn(self):
        env = ecoModel.Trainer()
        env.init_weights()
        env.train()


def main():

    myOrch = Orchestrator(global_frame_start=0,global_cycle_start=0,cycles_per_session=1)
    myOrch.learn()

main()