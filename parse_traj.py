# 3. combining controls and states

import numpy as np
import matplotlib.pyplot as plt
import os


X_pre = []
U_mid = []
X_post = []

def find_npz_files(directory):
    npz_files = []
    npz_dirs = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            npz_dirs.append(dir)
        for file in files:
            if (file.endswith(".npz")):
                full_path = os.path.join(root, file)
                npz_files.append(full_path)
    return npz_files, npz_dirs

# combining controls and states
def combine(file_path, u_buffer=5, x_buffer=1):
    data = np.load(file_path)
    l = 0; r = data['X'].shape[0]

    X = data['X']
    U = data['U']

    results = []
    u_idx = 0
    # for i, x in enumerate(X[:-u_buffer]):
    for i in range(0, X.shape[0] - x_buffer, x_buffer):
        # find all U.time > X.time
        while u_idx < len(U) and U[u_idx][0] <= X[i, 0]:
            u_idx += 1
        
        u = U[u_idx : min(u_idx+u_buffer, U.shape[0])]
        u = [j for sub in u for j in sub[1:]]
        u = np.array(u)

        x1 = np.array(X[i])
        x2 = np.array(X[i+x_buffer])
        xux = np.concatenate((x1, u, x2))
        # x1_u = np.concatenate((x1, u))
        # u_x2 = np.concatenate((u, x2))
        X_pre.append(x1)
        U_mid.append(u)
        X_post.append(x2)
        results.append(xux)
        # get x+ and x- from here !!

    # np.savez(f"np_data_final/UX_0.npz", X=data['X'][l:r, :], U=data['U'])
    return np.array(results) # shape[1] = len(x) + len(u) * ctrl_buffer_max

if __name__ == "__main__":
        wd = os.getcwd() + '/np_data_new/'
        npz_file_paths, npz_dirs = find_npz_files(wd)
        combined_results = None
        x_buffer, u_buffer = 3, 2
        for i, path in enumerate(npz_file_paths):
            results = combine(npz_file_paths[i], u_buffer=u_buffer, x_buffer=x_buffer)
            if combined_results is None:
                combined_results = results
            else:
                combined_results = np.concatenate((combined_results, results), axis=0)
        combined_results = np.array(combined_results)
        print(combined_results.shape)
        np.savez(f"np_data_final/traj__x{x_buffer}_u{u_buffer}", X0=X_pre, U=U_mid, X1=X_post)

        # print(npz_file_paths)
        print()