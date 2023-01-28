from copy import deepcopy
import torch
from torch import Tensor
from config import Config
from net import QNet  # DQN


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau

        self.last_state = None  # save the last state of the trajectory for training. `last_state.shape == (state_dim)`
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentDQN(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, 1), dtype=torch.int32).to(self.device)
        rewards = torch.ones(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state
        #print("ary_state: ", ary_state)
        get_action = self.act.get_action
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            if if_random:
                action = torch.randint(self.action_dim, size=(1,))[0]
            else:
                action = get_action(state.unsqueeze(0))[0, 0]

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]:
        obj_critics = 0.0
        q_values = 0.0

        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for i in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            obj_critics += obj_critic.item()
            q_values += q_value.item()
        return obj_critics / update_times, q_values / update_times

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)
            next_q = self.cri_target(next_state).max(dim=1, keepdim=True)[0]
            q_label = reward + undone * self.gamma * next_q
        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value.mean()


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, undones = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]
