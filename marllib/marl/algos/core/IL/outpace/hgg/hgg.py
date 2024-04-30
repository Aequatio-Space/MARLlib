import copy
import time
import os
import numpy as np
import torch

from warp_drive.utils.common import get_project_root
from .gcc_utils import gcc_load_lib, c_double


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, ord=2)


def goal_concat(obs, goal):
    return np.concatenate([obs, goal], axis=0)


class TrajectoryPool:
    def __init__(self, pool_length):
        self.length = pool_length
        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler:
    def __init__(self, env_name, config, achieved_trajectory_pool, num_episodes,
                 aim_discriminator=None, max_episode_timesteps=None, split_ratio_for_meta_nml=0.1,
                 split_type_for_meta_nml='last',
                 normalize_aim_output=False,
                 add_noise_to_goal=False, cost_type='meta_nml_aim_f', gamma=0.99, hgg_c=3.0, hgg_L=5.0, device='cuda',
                 hgg_gcc_path=None

                 ):
        # Assume goal env
        self.env_name = env_name

        self.add_noise_to_goal = add_noise_to_goal
        self.cost_type = cost_type

        self.vf = None
        self.critic = None
        self.policy = None

        self.max_episode_timesteps = max_episode_timesteps
        self.split_ratio_for_meta_nml = split_ratio_for_meta_nml
        self.split_type_for_meta_nml = split_type_for_meta_nml
        self.normalize_aim_output = normalize_aim_output
        self.gamma = gamma
        self.hgg_c = hgg_c
        self.hgg_L = hgg_L
        self.device = device

        self.loss_function = torch.nn.BCELoss(reduction='none')

        # TODO: removed logic for goal shape
        self.dim = 2
        # TODO: Deleted logic for self.delta
        self.delta = 1.0
        self.goal_distance = goal_distance
        self.aim_discriminator = aim_discriminator

        self.length = num_episodes  # args.episodes

        # randomly generate (x,y) between [0,1]
        init_goal = np.random.rand(self.dim)

        self.pool = np.tile(init_goal[np.newaxis, :], [self.length, 1]) + np.random.normal(0, self.delta,
                                                                                           size=(self.length, self.dim))

        self.match_lib = gcc_load_lib(
            os.path.join(get_project_root(), "MARLlib/marllib/marl/algos/core/IL/outpace/hgg/cost_flow.c"))

        self.achieved_trajectory_pool = achieved_trajectory_pool

        self.final_goal_states = None

        meta_nml_kwargs = config['meta_nml_kwargs']
        self.equal_pos_neg_test = meta_nml_kwargs['equal_pos_neg_test']
        self.meta_nml_negatives_only = meta_nml_kwargs['meta_nml_negatives_only']
        self.meta_nml_train_every_k = meta_nml_kwargs['meta_nml_train_every_k']
        self.meta_nml_train_on_positives = meta_nml_kwargs['meta_nml_train_on_positives']
        self.meta_nml_use_preprocessor = meta_nml_kwargs['meta_nml_use_preprocessor']
        self.meta_nml_custom_embedding_key = meta_nml_kwargs['meta_nml_custom_embedding_key']
        self.meta_task_batch_size = meta_nml_kwargs['meta_task_batch_size']
        self.meta_nml_shuffle_states = meta_nml_kwargs['meta_nml_shuffle_states']
        self.num_initial_meta_epochs = meta_nml_kwargs['num_initial_meta_epochs']
        self.num_meta_epochs = meta_nml_kwargs['num_meta_epochs']
        self.nml_grad_steps = meta_nml_kwargs['nml_grad_steps']
        self.test_strategy = meta_nml_kwargs['test_strategy']
        self.accumulation_steps = meta_nml_kwargs['accumulation_steps']
        # TODO: Deleted MAX_DIS Estimate
        self.max_dis = 10000

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        if noise_std is None: noise_std = self.delta

        if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
            goal += np.random.normal(0, noise_std, size=2)
            goal = np.clip(goal, (-2, -2), (10, 10))
        elif self.env_name in ['sawyer_peg_pick_and_place']:
            noise = np.random.normal(0, noise_std, size=goal.shape[-1])
            goal += noise
        elif self.env_name in ['sawyer_peg_push']:
            noise = np.random.normal(0, noise_std, size=goal.shape[-1])
            noise[2] = 0
            goal += noise
            goal[..., -3:] = np.clip(goal[..., -3:], (-0.6, 0.2, 0.0147), (0.6, 1.0, 0.0148))
        elif self.env_name == "PointSpiralMaze-v0":
            goal += np.random.normal(0, noise_std, size=2)
            goal = np.clip(goal, (-10, -10), (10, 10))
        elif self.env_name in ["PointNMaze-v0"]:
            goal += np.random.normal(0, noise_std, size=2)
            goal = np.clip(goal, (-2, -2), (10, 18))
        elif self.env_name in ['crowdsim']:
            goal = pre_goal
        else:
            raise NotImplementedError

        return goal.copy()

    def sample(self, idx):
        if self.add_noise_to_goal:
            if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0', "PointSpiralMaze-v0", "PointNMaze-v0"]:
                noise_std = 0.5
            elif self.env_name in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
                noise_std = 0.05
            else:
                raise NotImplementedError('Should consider noise scale env by env')
            return self.add_noise(self.pool[idx], noise_std=noise_std)
        else:
            return self.pool[idx].copy()

    def sample_negatives(self, replay_buffer, size):  # from replay buffer
        obs, _, _, _, _, _ = replay_buffer.sample_without_relabeling(size, 0.99, sample_only_state=False)
        # TODO: negatives to fill
        negatives = None
        labels = np.zeros(len(negatives))

        return negatives.astype(np.float32), labels

    def sample_positives(self, size):  # from final goal

        final_goal = self.final_goal_states.copy()

        rand_positive_ind = np.random.randint(0, final_goal.shape[0], size=size)

        batch = final_goal[rand_positive_ind]

        positives = batch

        return positives.astype(np.float32), np.ones(len(positives))

    def sample_meta_test_batch(self, size, replay_buffer=None):
        if self.meta_nml_negatives_only:
            return self.sample_negatives(replay_buffer, size)
        else:
            negatives = self.sample_negatives(replay_buffer, size // 2)
            positives = self.sample_positives(size // 2)
            return tuple(np.concatenate([a, b], axis=0) for a, b in zip(negatives, positives))

    def get_prob_by_meta_nml(self, observations, epoch, replay_buffer=None):

        if epoch == 0:
            finetuning_sample = None
        else:
            finetuning_sample = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer)

        classifier_inputs = observations

        eval_inputs = classifier_inputs
        prob = self.meta_nml.evaluate(eval_inputs,
                                      num_grad_steps=self.nml_grad_steps, train_data=finetuning_sample)[:, 1]

        return prob

    def update(self, initial_goals, desired_goals, replay_buffer=None, meta_nml_epoch=0):
        if self.achieved_trajectory_pool.counter == 0:
            self.pool = copy.deepcopy(desired_goals)
            return

        achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
        # for meta nml computational efficiency, jump 5% of max timesteps
        if 'meta_nml' in self.cost_type:
            if self.split_type_for_meta_nml == 'uniform':
                # uniform split
                achieved_pool = [traj[::int(self.max_episode_timesteps * self.split_ratio_for_meta_nml)] for traj in
                                 achieved_pool]  # list of reduced ts
                raise NotImplementedError
            elif self.split_type_for_meta_nml == 'last':
                # uniform split on last N steps
                if self.env_name in ['AntMazeSmall-v0']:
                    interval = 6
                elif self.env_name in ['PointUMaze-v0', 'sawyer_peg_push', 'sawyer_peg_pick_and_place', "PointNMaze-v0",
                                       "PointSpiralMaze-v0"]:
                    interval = 4
                else:
                    raise NotImplementedError
                achieved_pool = [np.concatenate(
                    [traj[int(-self.split_ratio_for_meta_nml * self.max_episode_timesteps)::interval], traj[-1:]],
                    axis=0) for traj in achieved_pool]  # list of reduced ts

        assert len(
            achieved_pool) >= self.length, 'If not, errors at assert match_count==self.length, e.g. len(achieved_pool)=5, self.length=25, match_count=5'
        if 'aim_f' in self.cost_type:
            assert self.aim_discriminator is not None
        candidate_goals = []
        candidate_edges = []
        candidate_id = []

        achieved_value = []
        for i in range(len(achieved_pool)):
            # maybe for all timesteps in an episode
            obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                   range(achieved_pool[i].shape[0])]  # list of [dim] (len = ts)

            with torch.no_grad():
                obs_t = torch.from_numpy(np.stack(obs, axis=0)).float().to(self.device)  # [ts, dim]
                if self.vf is not None:
                    value = self.vf(obs_t).detach().cpu().numpy()[:, 0]
                    value = np.clip(value, -1.0 / (1.0 - self.gamma), 0)
                elif self.critic is not None and self.policy is not None:
                    n_sample = 10
                    tiled_obs_t = torch.tile(obs_t, (n_sample, 1, 1)).view(
                        (-1, obs_t.shape[-1]))  # [ts, dim] -> [n_sample*ts, dim]
                    dist = self.policy(obs_t)  # obs : [ts, dim]
                    action = dist.rsample((n_sample,))  # [n_sample, ts, dim]
                    action = action.view((-1, action.shape[-1]))  # [n_sample*ts, dim]
                    actor_Q1, actor_Q2 = self.critic(tiled_obs_t, action)
                    actor_Q = torch.min(actor_Q1, actor_Q2).view(n_sample, -1, actor_Q1.shape[
                        -1])  # [n_sample*ts, dim(1)] -> [n_sample, ts, dim(1)]
                    value = torch.mean(actor_Q, dim=0).detach().cpu().numpy()[:, 0]  # [ts, dim(1)] -> [ts,]
                    value = np.clip(value, -1.0 / (1.0 - self.gamma), 0)
                elif (self.aim_discriminator is not None) and (
                        'aim_f' in self.cost_type):  # or value function is proxy for aim outputs
                    value = -self.aim_discriminator(obs_t).detach().cpu().numpy()[:, 0]
                # value = np.clip(value, -1.0/(1.0-self.gamma), 0)
            if 'aim_f' in self.cost_type:
                achieved_value.append(value.copy())
            elif 'meta_nml' in self.cost_type:
                pass
            else:
                raise NotImplementedError

        n = 0
        graph_id = {'achieved': [], 'desired': []}
        for i in range(len(achieved_pool)):
            n += 1
            graph_id['achieved'].append(n)
        for i in range(len(desired_goals)):
            n += 1
            graph_id['desired'].append(n)
        n += 1
        self.match_lib.clear(n)

        if 'aim_f' in self.cost_type:
            # print('normalize aim output in hgg update!')
            # For considering different traj length
            aim_outputs_max = -np.inf
            aim_outputs_min = np.inf
            for i in range(len(achieved_value)):  # list of aim_output [ts,]
                if achieved_value[i].max() > aim_outputs_max:
                    aim_outputs_max = achieved_value[i].max()
                if achieved_value[i].min() < aim_outputs_min:
                    aim_outputs_min = achieved_value[i].min()
            for i in range(len(achieved_value)):
                achieved_value[i] = ((achieved_value[i] - aim_outputs_min) / (
                        aim_outputs_max - aim_outputs_min + 0.00001) - 0.5) * 2  # [0, 1] -> [-1,1]

        if 'meta_nml' in self.cost_type:

            # achieved_pool : list of [ts, dim] (ts could be different)
            achieved_pool_traj_lengths = [traj.shape[0] for traj in achieved_pool]  # list of ts
            reshaped_achieved_pool = np.concatenate([traj for traj in achieved_pool], axis=0)  # [ts_1+ts_2+ ... , dim]
            start = time.time()
            reshaped_classification_probs = self.get_prob_by_meta_nml(reshaped_achieved_pool, meta_nml_epoch,
                                                                      replay_buffer=replay_buffer,
                                                                      goal_env=self.eval_env)
            # print('meta evaluation time in hgg update : ', time.time() - start)
            classification_probs = []
            for idx, length in enumerate(achieved_pool_traj_lengths):
                if idx == 0:
                    start_idx = 0
                    end_idx = length
                else:
                    start_idx = end_idx
                    end_idx = start_idx + length
                classification_probs.append(
                    torch.from_numpy(reshaped_classification_probs[start_idx:end_idx]).squeeze().float().to(
                        self.device))  # list of [ts, dim(1)] or [ts]

        for i in range(len(achieved_pool)):
            self.match_lib.add(0, graph_id['achieved'][i], 1, 0)

        for i in range(len(achieved_pool)):
            # meta_nml uncertainty distance metric, aim_f bias
            # cross entropy (when the probability becomes 1 (goal example), the loss is minimized)
            if (self.aim_discriminator is not None) and ('aim_f' in self.cost_type) and (
                    'meta_nml' in self.cost_type):
                labels = torch.ones_like(classification_probs[i]).to(self.device)
                cross_entropy_loss = self.loss_function(classification_probs[i], labels).detach().cpu().numpy()
                res = cross_entropy_loss - achieved_value[i] / (self.hgg_L / self.max_dis / (1 - self.gamma))
            elif (self.aim_discriminator is not None) and ('aim_f' in self.cost_type):
                res = - achieved_value[i] / (self.hgg_L / self.max_dis / (1 - self.gamma))
            elif ('meta_nml' in self.cost_type):
                labels = torch.ones_like(classification_probs[i]).to(self.device)
                cross_entropy_loss = self.loss_function(classification_probs[i], labels).detach().cpu().numpy()
                res = cross_entropy_loss
            match_dis = np.min(res)

            for j in range(len(desired_goals)):
                if ('aim_f' in self.cost_type) or ('meta_nml' in self.cost_type):
                    pass
                else:
                    raise NotImplementedError

                match_idx = np.argmin(res)

                edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
                candidate_goals.append(achieved_pool[i][match_idx])
                candidate_edges.append(edge)
                candidate_id.append(j)
        for i in range(len(desired_goals)):
            self.match_lib.add(graph_id['desired'][i], n, 1, 0)

        match_count = self.match_lib.cost_flow(0, n)
        assert match_count == self.length

        explore_goals = [0] * self.length
        for i in range(len(candidate_goals)):
            if self.match_lib.check_match(candidate_edges[i]) == 1:
                explore_goals[candidate_id[i]] = candidate_goals[i].copy()
        assert len(explore_goals) == self.length
        self.pool = np.array(explore_goals)
