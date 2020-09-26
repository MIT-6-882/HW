from collections import namedtuple, defaultdict
from itertools import count
import time
import heapq as hq


class AStar:
    """Planning with A* search. Used by SingleStateDeterminization.
    """
    
    Node = namedtuple("Node", ["state", "parent", "action", "g"])

    def __init__(self, successor_fn, check_goal_fn, heuristic=None, timeout=100):
        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._heuristic = heuristic or (lambda s : 0)
        self._timeout = timeout
        self._actions = None
        
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
            # If we already found a better path here, don't bother
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
            for action, child_state in self._get_successors(node.state):
                # If we already found a better path to child, don't bother
                if state_to_best_g[child_state] <= node.g+1:
                    continue
                # Add new node
                child_node = self.Node(state=child_state, parent=node, action=action, g=node.g+1)
                priority = self._get_priority(child_node)
                hq.heappush(queue, (priority, next(tiebreak), child_node))
                state_to_best_g[child_state] = child_node.g

        if verbose:
            print("Warning: planning failed.")
        return [], {'node_expansions' : num_expansions}
    
    def _get_successors(self, state):
        for action in self._actions:
            next_state = self._get_successor_state(state, action)
            yield action, next_state

    def _finish_plan(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        plan.reverse()
        return plan

    def _get_priority(self, node):
        h = self._heuristic(node)
        if isinstance(h, tuple):
            return (tuple(node.g + hi for hi in h), h)
        return (node.g + h, h)