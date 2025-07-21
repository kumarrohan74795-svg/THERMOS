class RLVariables:

  def __init__(self) :
    self.agent = []
    self.invalid_action_cnt = 0
    self.global_steps = 0
    self.episode_cnt = 0
    self.train_w = []
    ## PPO Arguments
    self.obs = [] #observation
    self.actions = []
    self.logprobs = []
    self.rewards = []
    self.dones = [] #done
    self.values = []
    self.preferences = []
    self.action_masks = []

    self.next_obs = [] #next observation
    self.next_done = [] #next done
    self.batch_values = []
    self.batch_step = 0
