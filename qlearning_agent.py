from typing import Any, Iterable, Sequence, Tuple
from state_representation import StateRepresentation
from utils import rotate_clockwise, rotate_counterclockwise
import numpy as np
np.random.seed(123)


class QlearningAgent:

    def __init__(self, env):
        self.env = env
        self.qtable = np.zeros([StateRepresentation.get_state_space_size(), env.action_space.n])

    def train(self,
              episodes: int,
              alpha=0.1, 
              alpha_min=0.01,
              gamma=0.6, 
              epsilon=0.1, 
              decay=0.0,
              epsilon_min=0.01,
              verbose: bool=False
              ) -> Iterable[float]:
        
        rewards = []

        for episode in range(episodes):

            obs = self.env.reset()
            state = StateRepresentation(
                agent_x_coord=0,
                agent_y_coord=0,
                agent_orientation=(0, 1),
                has_agent_grabbed_gold=False,
                has_agent_climbed_out=False
            )
            state_id = state.get_index()

            episode_reward = 0
            actions_cnt = 0
            random_actions_count = 0
            done = False

            # while the agent is alive or 
            # has not reached the target
            while not done:

                actions_cnt += 1
                
                if epsilon > np.random.random():
                    # exploration: choose random acton
                    # in case if paramenter epsilon will be larger 
                    # then random number from 0 to 1
                    random_actions_count += 1
                    action = np.random.choice(self.env.actions)
                else:
                    # choose the action with the highest value in the current state
                    # action_id = np.argmax(self.qtable[state_id])
                    max_q = np.max(self.qtable[state_id])
                    max_q_ids = np.where(self.qtable[state_id] == max_q)[0]
                    action_id = np.random.choice(max_q_ids) 
                    action = self.env.actions[action_id]
                
                # perform action and collect info
                new_obs, r, done, _ = self.env.step(action)

                percept = self.env.space_to_percept(obs)
                new_percept = self.env.space_to_percept(new_obs)

                # Reward engineering
                # ==================

                # MOVE 

                if action.value == 0:
                    if new_percept.bump:
                        r = -10
                    else:
                        state.agent_x_coord = state.agent_x_coord + state.agent_orientation[0]
                        state.agent_y_coord = state.agent_y_coord + state.agent_orientation[1]

                        if episode/episodes < 0.3:
                            # create artifitial reward
                            # to make action without punishment 
                            # for the 1st part of rounds
                            r = 0
                        else:
                            # real reward
                            r = -1

                # RIGHT
                
                if action.value == 1:
                    state.agent_orientation = rotate_clockwise(state.agent_orientation)

                    if episode/episodes < 0.3:
                        # create artifitial reward
                        # to make action without punishment 
                        # for the 1st part of rounds
                        r = 0
                    else:
                        # real reward
                        r = -1

                # LEFT

                if action.value == 2:
                    state.agent_orientation = rotate_counterclockwise(state.agent_orientation)

                    if episode/episodes < 0.3:
                        # create artifitial reward
                        # to make action without punishment 
                        # for the 1st part of rounds
                        r = 0
                    else:
                        # real reward
                        r = -1

                # SHOOT

                if action.value == 3:
                    state.has_agent_shot_arrow = True

                    if new_percept.scream and episode/episodes < 0.3:
                        # create artifitial reward
                        # to explore additional environmennt
                        # despite the fact that in reality we lose
                        # points becasuse of shooting
                        r = 10
                    else:
                        # real reward
                        r = -10

                # GRAB

                if action.value == 4: 

                    if percept.glitter:
                        state.has_agent_grabbed_gold = True
                        r = 1000
                
                # CLIMB
                
                if action.value == 5:
                    
                    if state.agent_x_coord == 0 and state.agent_y_coord == 0:
                        state.has_agent_climbed_out = True
                        if state.has_agent_grabbed_gold:
                            r = 1000
                        else:
                            # create artifitial reward
                            # not to punush CLIMB action
                            # helps map02 to climb out instead of dying
                            r = 0
                    else:
                        r = -1
                
                # update expected reward (Q function)
                new_state_id = state.get_index()

                self.qtable[state_id, action.value] = \
                    (1 - alpha) * self.qtable[state_id, action.value] + alpha * (r + gamma * np.max(self.qtable[new_state_id]))

                obs = new_obs
                state_id = new_state_id
                episode_reward += r

                # if agent dies or reaches a goal
                if done:
                    if verbose:
                        if episode % 1_000 == 0:
                            print(f"Episode {episode}/{episodes}, " \
                                + f"reward {episode_reward}, " \
                                + f"random actions {random_actions_count}, " \
                                + f"actions {actions_cnt}, " \
                                + f"epsilon {epsilon:.2f}, " \
                                + f"alpha {alpha:.2f}, " \
                            )
                    
                    # decrease probability of choosing random action
                    epsilon = epsilon_min + (epsilon - epsilon_min) * (1 - episode/episodes)

                    # decay learning rate
                    alpha = alpha_min + (alpha - alpha_min) * np.exp(-decay * episode)

                    rewards.append(episode_reward)

        self.env.close()
        return self.qtable, rewards
    

    def evaluate(self) -> Tuple[int, Sequence[str]]:
        """
        Use learned qtable to check result
        (we dont modify qtable now).
        """
        eval_reward = 0
        frames = []
        actions = []

        obs = self.env.reset()
        
        state = StateRepresentation(
            agent_x_coord=0,
            agent_y_coord=0,
            agent_orientation=(0, 1),
            has_agent_grabbed_gold=False,
            has_agent_climbed_out=False
        )
        state_id = state.get_index()
        
        done = False

        while not done:
            # save the frame corresponding to the current state
            frames.append(self.env.render('ansi'))
            
            # select best action
            action_id = np.argmax(self.qtable[state_id])
            action = self.env.actions[action_id]
            actions.append(action)
            
            # perform the action
            new_obs, reward, done, info = self.env.step(action)
                
            percept = self.env.space_to_percept(obs)
            new_percept = self.env.space_to_percept(new_obs)
                
            # MOVE 

            if action.value == 0 and not new_percept.bump:
                state.agent_x_coord = state.agent_x_coord + state.agent_orientation[0]
                state.agent_y_coord = state.agent_y_coord + state.agent_orientation[1]

            # RIGHT
            
            if action.value == 1:
                state.agent_orientation = rotate_clockwise(state.agent_orientation)

            # LEFT

            if action.value == 2:
                state.agent_orientation = rotate_counterclockwise(state.agent_orientation)

            # SHOOT

            if action.value == 3:
                state.has_agent_shot_arrow = True

            # GRAB

            if action.value == 4 and percept.glitter:
                state.has_agent_grabbed_gold = True
            
            # CLIMB
            
            if action.value == 5 and state.agent_x_coord == 0 and state.agent_y_coord:
                state.has_agent_climbed_out = True
            
            state_id = state.get_index()
            obs = new_obs
            eval_reward += reward

        self.env.close()

        return eval_reward, frames, actions
