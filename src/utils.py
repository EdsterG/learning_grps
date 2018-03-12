import numpy as np
import settings
import tarfile
import h5py
import time
import os


def load_dataset(dataset_name, tmp_dir="/tmp"):
    print ("Loading dataset...")
    dataset_filename = "{}.hdf5".format(dataset_name)
    dataset_filepath = "{}/data/dataset/{}".format(settings.PROJECT_PATH, dataset_filename)
    filepath_gz = "{}.tar.gz".format(dataset_filepath)
    if os.path.exists(filepath_gz):
        dataset_filepath = "{}/{}".format(tmp_dir, dataset_filename)
        if not os.path.exists(dataset_filepath):
            start = time.time()
            with tarfile.open(filepath_gz, mode='r') as f:
                f.extractall(tmp_dir)
            print ("Dataset decompression took: {} seconds".format(time.time()-start))
        return h5py.File(dataset_filepath, 'r')
    else:
        raise Exception("Can't find file {}".format(filepath_gz))


def convert_to_execution_traces(dataset_file, num_samples):
    data = {}
    data["X_dim"], data["Y_dim"] = dataset_file.attrs["world_shape"]
    data["r_loc"] = dataset_file["robot_loc"][:num_samples]
    data["o_locs"] = dataset_file["obj_loc"][:num_samples]
    data["g_locs"] = dataset_file["goal_loc"][:num_samples]
    data["s_len"] = dataset_file["sequence_length"][:num_samples]
    data["s_mask"] = dataset_file["sequence_mask"][:num_samples]

    data["wall"] = dataset_file["wall"][:num_samples]

    NUM_ACTIONS = dataset_file.attrs["num_actions"]
    Tmax = data["r_loc"].shape[1]
    N = num_samples*Tmax

    # Create observation
    obs = np.zeros([num_samples, Tmax, data["Y_dim"], data["X_dim"], 4], dtype="bool")
    obs[:,:,:,:,0] = np.expand_dims(data["wall"], axis=1)
    for i in range(data["g_locs"].shape[1]):
        x, y = data["g_locs"][:,i,:].T
        idx0 = np.repeat(range(num_samples),Tmax)
        idx1 = np.tile(range(Tmax), num_samples)
        idx2 = np.repeat(x, Tmax)
        idx3 = np.repeat(y, Tmax)
        obs[idx0,idx1,idx2,idx3,1] = 1

    obs = obs.reshape([-1, data["Y_dim"], data["X_dim"], 4])
    idx0 = range(N)
    for i in range(data["o_locs"].shape[1]):
        idx1, idx2 = data["o_locs"][:,i,:].reshape([-1,2]).T
        obs[idx0,idx1,idx2,2] = 1
    idx1, idx2 = data["r_loc"].reshape([-1,2]).T
    obs[idx0,idx1,idx2,3] = 1
    obs = obs.reshape([num_samples, Tmax, data["Y_dim"], data["X_dim"], 4])

    # NOTE: walls uses the convention walls[x,y]
    # observation images use the convention img[height-y,x]
    # rot90 converts to the img convention
    obs = np.transpose(obs, axes=[2,3,0,1,4])
    obs = np.rot90(obs)
    obs = np.transpose(obs, axes=[2,3,0,1,4])

    # Create one-hot action
    idx0, idx1 = range(N), dataset_file["action"][:num_samples].flatten()
    action_one_hot = np.zeros([N, NUM_ACTIONS], dtype='bool')
    action_one_hot[idx0, idx1] = 1
    action_one_hot = action_one_hot.reshape([num_samples, Tmax, NUM_ACTIONS])

    # Create steps-to-goal
    steps_to_goal = np.zeros_like(dataset_file["action"][:num_samples])
    for i in range(num_samples):
        T = data["s_len"][i]
        steps_to_goal[i,:T-1] = range(T-1,0,-1)
    steps_to_goal = np.expand_dims(steps_to_goal, axis=-1)

    data["obs"] = obs
    data["action_one_hot"] = action_one_hot
    data["steps_to_goal"] = steps_to_goal

    return data
