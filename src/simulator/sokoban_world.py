import numpy as np
import time
import sys
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    MPL_INSTALLED = True
except:
    MPL_INSTALLED = False

if sys.version_info[0] == 2:
    # Running python 2.7
    range = xrange

class SokobanWorld(object):
    def __init__(self, config={}):
        self._config = config

        # Matplotlib
        self._MPL_env = {}
        self._MPL_viewer = None

        # Internal state info
        self._walls = None
        self._robot_loc = None
        self._obj_locs = {}
        self._goal_locs = set()

        # Execution statistics
        self._clear_execution_stats()

    def reset_world(self, walls, robot_loc, obj_locs, goal_locs):
        self._walls = np.array(walls, dtype="bool")
        self._robot_loc = np.array(robot_loc)
        self._obj_locs = {}
        for o_id, obj_loc in enumerate(obj_locs):
            self._obj_locs[tuple(obj_loc)] = o_id
        self._goal_locs = set()
        for goal_loc in goal_locs:
            self._goal_locs.add(tuple(goal_loc))
        if self._MPL_viewer is not None and plt.fignum_exists(self._MPL_viewer.number):
            self._reset_MPL_world()
            self._init_MPL_world()
            self._update_MPL_canvas()
        self._clear_execution_stats()

    def _update_execution_history(self, action):
        self._history["robot_loc"].append(self._robot_loc)
        for o_loc, o_id in self._obj_locs.items():
            self._history["obj_loc"][o_id].append(o_loc)
        self._history["observation"].append(self.get_observation())
        self._history["action"].append(action)
        self._history["length"] += 1

    def _clear_execution_stats(self):
        num_objects = len(self._obj_locs)
        num_goals = len(self._goal_locs)
        self._history = {
            "robot_loc": [],
            "obj_loc": [[] for _ in range(num_objects)],
            "goal_loc": list(self._goal_locs),
            "observation": [],
            "action": [],
            "length": 0,
        }

        self.execution_stats = {
            "target_state_reached": False,
            "NoOp_executed": False,
            "stuck_at": -1,
            "oscillating": -1,
            # "deadlocked": False,
        }

        if self._MPL_viewer is not None and plt.fignum_exists(self._MPL_viewer.number):
            # Show more cycles before stopping animation, makes it more apparent.
            self._num_cyles = 6 
        else:
            self._num_cyles = 2

    def _compute_execution_stats(self, stochastic):
        # NOTE: the following line assumes objects will never be in a "stacked" state
        if self._obj_locs:
            assert len(self._obj_locs) == len(self._goal_locs), \
                "number of objects and number of goals must be equal"
            target_state_reached = np.all([o_loc in self._goal_locs for o_loc in self._obj_locs.keys()])
        else:
            target_state_reached = tuple(self._robot_loc) in self._goal_locs
        self.execution_stats["target_state_reached"] |= target_state_reached

        if not stochastic:
            state_repeats = np.all(np.isclose(self._history["observation"][-1],
                                              self._history["observation"]).reshape([self._history["length"], -1]), axis=1)
            if state_repeats[-3:].sum() == 3: # Stuck if repeats 3 times...
                self.execution_stats["stuck_at"] = self._history["action"][-1]
            elif state_repeats[:-1].sum() == self._num_cyles:
                state_period = np.diff(np.where(state_repeats)[0][-2:])[0]
                self.execution_stats["oscillating"] = state_period

    def step(self, action_distr, stochastic=False, compute_stats=True):
        assert self._walls is not None, "the world hasn't been loaded..."
        if stochastic:
            action = np.random.choice(action_distr.size, p=action_distr)
        else:
            action = action_distr.argmax()
        if compute_stats:
            self._update_execution_history(action)

        finished = False
        if action != 4:
            self._move_robot(action)
        else:
            self.execution_stats["NoOp_executed"] = True
            if self.execution_stats["target_state_reached"]:
                finished = True
        if compute_stats:
            self._compute_execution_stats(stochastic)
        self._update_MPL_world()
        return finished


    def _move_robot(self, action):
        next_loc = self._get_next_loc(self._robot_loc, action)
        if tuple(next_loc) in self._obj_locs:
            blocked = self._move_obj(next_loc, action)
        elif self._walls[next_loc[0], next_loc[1]] == True:
            blocked = True
        else:
            blocked = False

        if not blocked:
            self._robot_loc = next_loc

    def _move_obj(self, curr_loc, action):
        next_loc = self._get_next_loc(curr_loc, action)
        if tuple(next_loc) in self._obj_locs or self._walls[next_loc[0], next_loc[1]] == True:
            return True
        else:
            o_id = self._obj_locs.pop(tuple(curr_loc))
            self._obj_locs[tuple(next_loc)] = o_id
            return False

    def _get_next_loc(self, curr_loc, action):
        if action == 0:
            next_loc = curr_loc + [0,1] # MoveUp
        elif action == 1:
            next_loc = curr_loc + [-1,0] # MoveLeft
        elif action == 2:
            next_loc = curr_loc + [0,-1] # MoveDown
        elif action == 3:
            next_loc = curr_loc + [1,0] # MoveRight
        return next_loc

    def get_observation(self):
        obs = np.zeros(self._walls.shape+(4,), dtype="bool")
        obs[:,:,0] = self._walls
        for x,y in self._goal_locs:
            obs[x,y,1] = 1
        for x,y in self._obj_locs.keys():
            obs[x,y,2] = 1
        x,y = self._robot_loc
        obs[x,y,3] = 1

        # NOTE: walls uses the convention walls[x,y]
        # observation images use the convention img[height-y,x]
        # rot90 converts to the img convention
        return np.rot90(obs)

    def screenshot(self, filename):
        if self._MPL_viewer is not None and plt.fignum_exists(self._MPL_viewer.number):
            self._MPL_viewer.savefig(filename, bbox_inches='tight')

    def render_MPL(self):
        assert MPL_INSTALLED, "install Matplotlib to use this feature"
        if self._MPL_viewer is None or not plt.fignum_exists(self._MPL_viewer.number):
            self._MPL_viewer = self._init_MPL_viewer()
            self._init_MPL_world()
            self._MPL_viewer.show()
            self._update_MPL_canvas()

    def _init_MPL_viewer(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')

        # Remove the numbers
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Remove the ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        return fig

    def _update_MPL_canvas(self):
        self._MPL_viewer.canvas.draw()
        self._MPL_viewer.canvas.flush_events()

    def _init_MPL_world(self):
        ax = self._MPL_viewer.get_axes()[0]
        self.draw_env(ax)

    def draw_env(self, ax):
        if self._walls is not None:
            self._MPL_env["walls"] = self._init_MPL_walls(ax, self._walls)
        if self._goal_locs:
            self._MPL_env["goals"] = self._init_MPL_goals(ax, np.array([g for g in self._goal_locs]))
        self._MPL_env["objs"] = self._init_MPL_objs(ax, self._obj_locs)
        if self._robot_loc is not None:
            self._MPL_env["robot"] = self._init_MPL_robot(ax, self._robot_loc)

    def _init_MPL_walls(self, ax, walls):
        color = "#377eb8" # Green
        centers = np.transpose(np.where(walls==1))
        corners = np.expand_dims([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]],axis=1)
        verts = centers + np.tile(corners,[1,centers.shape[0],1])
        squares = np.swapaxes(verts, 0, 1)
        collection = matplotlib.collections.PolyCollection(squares, facecolors=color)
        collection.set_zorder(-1)
        ax.add_collection(collection)

        X, Y = walls.shape
        # Set axis limits
        ax.set_xlim([-0.5, X-0.5])
        ax.set_ylim([-0.5, Y-0.5])
        # Setup the grid
        ax.set_xticks(np.arange(-0.5, X, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Y, 1), minor=True)
        # Show the grid
        ax.xaxis.grid(which='minor', linestyle="-", color='black', zorder=1)
        ax.yaxis.grid(which='minor', linestyle="-", color='black', zorder=1)
        return collection

    def _init_MPL_goals(self, ax, centers):
        color = "#4daf4a" # Green
        corners = np.expand_dims([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]],axis=1)
        verts = centers + np.tile(corners,[1,centers.shape[0],1])
        squares = np.swapaxes(verts, 0, 1)
        collection = matplotlib.collections.PolyCollection(squares, facecolors=color)
        collection.set_zorder(-1)
        ax.add_collection(collection)
        return collection

    def _init_MPL_objs(self, ax, obj_locs):
        obj_patches = []
        for obj_loc in obj_locs:
            color = "#984ea3" # Purple
            mpl_obj = Circle(xy=obj_loc, radius=0.25, color=color, zorder=0)
            ax.add_patch(mpl_obj)
            obj_patches.append(mpl_obj)
        return obj_patches

    def _init_MPL_robot(self, ax, robot_loc):
        color = "#e41a1c" # Red
        mpl_obj = Circle(xy=robot_loc, radius=0.45, color=color, zorder=0)
        ax.add_patch(mpl_obj)
        return mpl_obj

    def _update_MPL_world(self):
        if self._MPL_viewer is not None and plt.fignum_exists(self._MPL_viewer.number):
            self._MPL_env["robot"].center = self._robot_loc
            for mpl_obj, obj_loc in zip(self._MPL_env["objs"], self._obj_locs):
                mpl_obj.center = obj_loc
            self._update_MPL_canvas()

    def _reset_MPL_world(self):
        if self._MPL_viewer is not None and plt.fignum_exists(self._MPL_viewer.number):
            if "walls" in self._MPL_env:
                self._MPL_env["walls"].remove()
                del self._MPL_env["walls"]
            if "goals" in self._MPL_env:
                self._MPL_env["goals"].remove()
                del self._MPL_env["goals"]
            if "robot" in self._MPL_env:
                self._MPL_env["robot"].remove()
                del self._MPL_env["robot"]
            if "objs" in self._MPL_env:
                for obj in self._MPL_env["objs"]:
                    obj.remove()
                    del obj
                del self._MPL_env["objs"]

        self._MPL_env = {}

    def simulate(self, policy, walls, r_loc, o_locs, g_locs, timesteps, stochastic=False, pause_time=0):
        traj = {
            "action": [],
            "img_state": [],
        }
        stats = {
            "stuck": False,
            "cycle": False,
            "goal_reached": False,
            "other_failure": False,
        }

        self.reset_world(walls, r_loc, o_locs, g_locs)
        if pause_time > 0:
            time.sleep(pause_time)
        init_img_state = self.get_observation()
        goal_img_state = init_img_state.copy()
        goal_img_state[:,:,3] *= False # Robot layer is zeroed out
        goal_img_state[:,:,2] = goal_img_state[:,:,1] # Objects should be at goal
        policy.initialize(init_img_state, goal_img_state)
        for t in range(timesteps):
            obs = self.get_observation()
            traj["img_state"].append(obs)

            action = policy.act(obs)
            traj["action"].append(action)
            finished = self.step(action, stochastic=stochastic)
            if pause_time > 0:
                time.sleep(pause_time)
            if finished:
                break
            if self.execution_stats["stuck_at"] != -1:
                traj["img_state"] = traj["img_state"][:-2]
                traj["action"] = traj["action"][:-2]
                break
            if self.execution_stats["oscillating"] != -1:
                n = (self._num_cyles-1)*self.execution_stats["oscillating"]+1
                traj["img_state"] = traj["img_state"][:-n]
                traj["action"] = traj["action"][:-n]
                break

        if self.execution_stats["target_state_reached"]:
            stats["goal_reached"] = True
        elif self.execution_stats["stuck_at"] != -1:
            stats["stuck"] = True
        elif self.execution_stats["oscillating"] != -1:
            stats["cycle"] = True
        else:
            stats["other_failure"] = True

        return stats



if __name__ == "__main__":
    sw = SokobanWorld()
    sw.render_MPL()

    walls = np.ones([9,12])
    walls[1:-1,1:-1] = 0

    sw.reset_world(walls, [5,10], set([(6,4),(6,3)]), set([(3,5)]))

