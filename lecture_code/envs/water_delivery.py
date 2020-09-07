from utils import render_from_layout, get_asset_path
import numpy as np
import matplotlib.pyplot as plt


class WaterDeliveryEnv:
    """A grid world where a robot must pick up a water bottle and then
    deliver water to all people in the grid. Rewards are given when each
    person is quenched.

    Parameters
    ----------
    layout : np.ndarray, layout.shape = (height, width, num_objects)
        The initial state.
    """
    # Types of objects
    OBJECTS = ROBOT, ROBOT_WITH_WATER, WATER, PERSON, QUENCHED_PERSON = range(5)

    # Create a default layout
    DEFAULT_LAYOUT = np.zeros((15, 15, len(OBJECTS)), dtype=bool)
    DEFAULT_LAYOUT[14, 7, ROBOT] = 1
    DEFAULT_LAYOUT[0, 14, WATER] = 1
    DEFAULT_LAYOUT[3, 1, PERSON] = 1
    DEFAULT_LAYOUT[7, 1, PERSON] = 1
    DEFAULT_LAYOUT[11, 1, PERSON] = 1

    # Actions
    ACTIONS = UP, DOWN, LEFT, RIGHT = range(4)

    # Reward for quenching
    QUENCH_REWARD = 100

    # For rendering
    TOKEN_IMAGES = {
        ROBOT : plt.imread(get_asset_path('robot.png')),
        ROBOT_WITH_WATER : plt.imread(get_asset_path('robot_with_water.png')),
        WATER : plt.imread(get_asset_path('water.png')),
        PERSON : plt.imread(get_asset_path('person.png')),
        QUENCHED_PERSON : plt.imread(get_asset_path('quenched_person.png')),
    }

    def __init__(self, layout=None):
        if layout is None:
            layout = self.DEFAULT_LAYOUT
        self._initial_layout = layout
        self._layout = layout.copy()

    def reset(self):
        self._layout = self._initial_layout.copy()
        return self.get_state(), {}

    def step(self, action):
        # Start out reward at 0
        reward = 0.

        # Move the robot
        rob_r, rob_c = None, None
        for robot_type in [self.ROBOT, self.ROBOT_WITH_WATER]:
            where_robot = np.argwhere(self._layout[..., robot_type])
            if len(where_robot) == 0:
                continue
            assert len(where_robot) == 1 or rob_r is not None, "Multiple robots in grid"
            rob_r, rob_c = where_robot[0]
            dr, dc = {self.UP : (-1, 0), self.DOWN : (1, 0), 
                      self.LEFT : (0, -1), self.RIGHT : (0, 1)}[action]
            new_r, new_c = rob_r + dr, rob_c + dc
            if 0 <= new_r < self._layout.shape[0] and 0 <= new_c < self._layout.shape[1]:
                # Remove old robot
                self._layout[rob_r, rob_c, robot_type] = 0
                # Add new robot
                self._layout[new_r, new_c, robot_type] = 1
                # Update local vars
                rob_r, rob_c = new_r, new_c
        assert rob_r is not None, "Missing robot in grid"

        # Handle water pickup
        if self._layout[rob_r, rob_c, self.ROBOT] and self._layout[rob_r, rob_c, self.WATER]:
            # Make robot have water
            self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] = 1
            self._layout[rob_r, rob_c, self.ROBOT] = 0
            # Remove water from grid
            self._layout[rob_r, rob_c, self.WATER] = 0

        # Handle people quenching
        if self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] and self._layout[rob_r, rob_c, self.PERSON]:
            # Quench person
            self._layout[rob_r, rob_c, self.QUENCHED_PERSON] = 1
            self._layout[rob_r, rob_c, self.PERSON] = 0
            # Reward for quenching
            reward += self.QUENCH_REWARD

        # Check done: all people quenched
        done = (len(np.argwhere(self._layout[..., self.PERSON])) == 0)

        return self.get_state(), reward, done, {}

    def render(self):
        return render_from_layout(self._layout, self._get_token_images, dpi=50)

    def _get_token_images(self, obs_cell):
        images = []
        for token in [self.ROBOT, self.ROBOT_WITH_WATER,self. WATER, 
                      self.PERSON, self.QUENCHED_PERSON]:
            if obs_cell[token]:
                images.append(self.TOKEN_IMAGES[token])
        return images

    def get_state(self):
        return tuple(sorted(map(tuple, np.argwhere(self._layout))))

    def set_state(self, state):
        self._layout = np.zeros_like(self._initial_layout)
        for i, j, k in state:
            self._layout[i, j, k] = 1


if __name__ == "__main__":
    import imageio

    max_num_steps = 100
    use_default_layout = True

    if use_default_layout:
        layout = None
    else:
        layout = np.zeros((5, 5, len(WaterDeliveryEnv.OBJECTS)), dtype=bool)
        layout[4, 2, WaterDeliveryEnv.ROBOT] = 1
        layout[0, 4, WaterDeliveryEnv.WATER] = 1
        layout[1, 0, WaterDeliveryEnv.PERSON] = 1
        layout[2, 0, WaterDeliveryEnv.PERSON] = 1
        layout[3, 0, WaterDeliveryEnv.PERSON] = 1

    images = []
    env = WaterDeliveryEnv(layout)
    state, _ = env.reset()
    images.append(env.render())
    print("Initial state:", state)
    for _ in range(max_num_steps):
        action = np.random.choice(env.ACTIONS)
        print("Taking action", action)
        state, reward, done, _ = env.step(action)
        print("State:", state)
        print("Reward, Done:", reward, done)
        images.append(env.render())
        if done:
            break
    outfile = "/tmp/water_delivery_random_actions.mp4"
    imageio.mimsave(outfile, images)
    print("Wrote out to", outfile)


