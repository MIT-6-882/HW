from collections import namedtuple, defaultdict
from itertools import count
import numpy as np
import time
import heapq as hq


class AStar:
    """Planning with A* search. Used by SingleStateDeterminization.
    """
    
    Node = namedtuple("Node", ["state", "parent", "action", "g"])

    def __init__(self, successor_fn, check_goal_fn, get_cost=None,
                 actions=None, heuristic=None, timeout=100):
        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._get_cost = get_cost
        self._heuristic = heuristic or (lambda s : 0)
        self._timeout = timeout
        self._actions = actions
        
    def __call__(self, state, verbose=False):
        return self._get_plan(state, verbose=verbose)

    def set_actions(self, actions):
        self._actions = actions

    def _get_plan(self, state, verbose=True):
        start_time = time.time()
        queue = []
        state_to_best_g = defaultdict(lambda : float("inf"))
        tiebreak = count()

        root_node = self.Node(state=state, parent=None, action=None, g=0)
        hq.heappush(queue, (self._get_priority(root_node), next(tiebreak), root_node))
        num_expansions = 0

        while len(queue) > 0 and (time.time() - start_time < self._timeout):
            _, _, node = hq.heappop(queue)
            if state_to_best_g[node.state] < node.g:
                continue
            # If the goal holds, return
            if self._check_goal(node.state):
                if verbose:
                    print("\nPlan found!")
                return self._finish_plan(node), {'node_expansions' : num_expansions}
            num_expansions += 1
            if verbose:
                print(f"Expanding node {num_expansions}", end='\r', flush=True)
            # Generate successors
            for action, child_state, cost in self._get_successors(node.state):
                # If we already found a better path to child, don't bother
                if state_to_best_g[child_state] <= node.g+cost:
                    continue
                # Add new node
                child_node = self.Node(state=child_state, parent=node, action=action,
                                       g=node.g+cost)
                priority = self._get_priority(child_node)
                hq.heappush(queue, (priority, next(tiebreak), child_node))
                state_to_best_g[child_state] = child_node.g

        if verbose:
            print("Warning: planning failed.")
        return [], {'node_expansions' : num_expansions}
    
    def _get_successors(self, state):
        for action in self._actions:
            next_state = self._get_successor_state(state, action)
            if self._get_cost is None:
                cost = 1
            else:
                cost = self._get_cost(state, action)
            yield action, next_state, cost

    def _finish_plan(self, node):
        plan = []
        nodes = []
        while node.parent is not None:
            plan.append(node.action)
            nodes.append(node)
            node = node.parent
        plan.reverse()
        nodes.reverse()
        # import ipdb; ipdb.set_trace()
        return plan

    def _get_priority(self, node):
        h = self._heuristic(node)
        if isinstance(h, tuple):
            return (tuple(node.g + hi for hi in h), h)
        return (node.g + h, h)


class UCT:
    """Implementation of UCT based on Leslie's lecture notes. Used by POUCT.
    """
    def __init__(self, actions, reward_fn, transition_fn, done_fn=None, 
                 num_search_iters=100, gamma=0.9, rng=None):
        self._actions = actions
        self._reward_fn = reward_fn
        self._transition_fn = transition_fn
        self._done_fn = done_fn or (lambda s,a : False)
        self._num_search_iters = num_search_iters
        self._gamma = gamma
        self._rng = rng
        self._Q = None
        self._N = None
        self._node_expansions = 0

    def run(self, state, horizon=100):
        # Initialize Q[s][a][d] -> float
        self._Q = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        # Initialize N[s][a][d] -> int
        self._N = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        # Loop search
        for it in range(self._num_search_iters):
            # Update Q
            self._search(state, 0, horizon=horizon)
        info = {"node_expansions" : self._node_expansions}
        self._node_expansions = 0
        return info

    def get_action(self, state, t=0):
        # Return best action, break ties randomly
        return max(self._actions, key=lambda a : (self._Q[state][a][t], self._rng.uniform()))

    def _search(self, s, depth, horizon=100):
        # Base case
        if depth == horizon:
            return 0.
        # Select an action, balancing explore/exploit
        a = self._select_action(s, depth, horizon=horizon)
        # Create a child state
        next_state = self._transition_fn(s, a)
        self._node_expansions += 1
        # Get value estimate
        if self._done_fn(s, a):
            # Some environments terminate problems before the horizon 
            q = self._reward_fn(s, a)
        else:
            q = self._reward_fn(s, a) + self._gamma * self._search(next_state, depth+1, horizon=horizon)
        # Update values and counts
        num_visits = self._N[s][a][depth] # before now
        # First visit to (s, a, depth)
        if num_visits == 0:
            self._Q[s][a][depth] = q
        # We've been here before
        else:
            # Running average
            self._Q[s][a][depth] = (num_visits / (num_visits + 1.)) * self._Q[s][a][depth] + \
                                   (1 / (num_visits + 1.)) * q
        # Update num visits
        self._N[s][a][depth] += 1
        return self._Q[s][a][depth]

    def _select_action(self, s, depth, horizon):
        # If there is any action where N(s, a, depth) == 0, try it first
        untried_actions = [a for a in self._actions if self._N[s][a][depth] == 0]
        if len(untried_actions) > 0:
            return self._rng.choice(untried_actions)
        # Otherwise, take an action to trade off exploration and exploitation
        N_s_d = sum(self._N[s][a][depth] for a in self._actions)
        best_action_score = -np.inf
        best_actions = []
        for a in self._actions:
            explore_bonus = (np.log(N_s_d) / self._N[s][a][depth])**((horizon + depth) / (2*horizon + depth))
            score = self._Q[s][a][depth] + explore_bonus
            if score > best_action_score:
                best_action_score = score
                best_actions = [a]
            elif score == best_action_score:
                best_actions.append(a)
        return self._rng.choice(best_actions)

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)


def value_iteration(mdp, max_num_iterations=1000, change_threshold=1e-4,
                    gamma=0.99, print_every=None, include_ns_in_r=False):
    """Run value iteration for a certain number of iterations or until
    the max change between iterations is below a threshold.

    Gamma is the temporal discount factor.

    Returns
    -------
    Q : { hashable : { hashable : float } }
        Q[state][action] = action-value.
    """
    # Get states, actions, T, and R
    states = mdp.get_all_states()
    actions = mdp.get_all_actions()
    T = mdp.get_transition_probabilities
    if not include_ns_in_r:
        R = lambda s,a,ns : mdp.get_reward(s, a)
    else:
        R = mdp.get_reward

    # Initialize Q to all zeros
    Q = { s : { a : 0. for a in actions } for s in states }

    for it in range(max_num_iterations):
        next_Q = {}
        max_change = 0.
        for s in states:
            next_Q[s] = {}
            for a in actions:
                # Handle terminal states
                if hasattr(mdp, 'state_is_terminal') and mdp.state_is_terminal(s):
                    next_Q[s][a] = 0. # terminal states always 0
                else:
                    # Main equation!
                    next_Q[s][a] = sum([p * (R(s, a, ns) + \
                                             gamma * max(Q[ns].values())) \
                                        for ns, p in T(s, a).items()])
                max_change = max(abs(Q[s][a] - next_Q[s][a]), max_change)
        if print_every is not None and it % print_every == 0:
            print(f"VI max change after iteration {it} : {max_change}")
        Q = next_Q
        # Check if we can terminate early
        if max_change < change_threshold:
            break

    return Q

def create_policy_from_Q(Q, rng=np.random):
    """Create a policy from action-values Q

    If we are sure that there are no ties, then this function could be
    one line: lambda s : max(Q[s], key=Q[s].get)

    But we want to randomly sample to break ties.

    Parameters
    ----------
    Q : { hashable : { hashable : float } }
        Q[state][action] = action-value.
    
    Returns
    -------
    policy : fn: state -> action
    """
    # Create exploit policy from Q
    def policy(s):
        best_actions = set()
        best_action_value = -np.inf
        for a, val in Q[s].items():
            if val > best_action_value:
                best_action_value = val
                best_actions = { a }
            elif val == best_action_value:
                best_actions.add(a)
        if len(best_actions) == 1:
            return next(iter(best_actions))
        # Break ties randomly
        best_actions = sorted(best_actions)
        rng.shuffle(best_actions)
        return best_actions[0]
    return policy

