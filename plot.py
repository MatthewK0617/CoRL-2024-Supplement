import matplotlib.pyplot as plt
import numpy as np
# sliced_array = np.load("np_data_final/traj_div__x3_u2.npz")['X0'][:, :, 1:4]
# print(sliced_array.shape)

# # for i in range(6, sliced_array.shape[0], 1):
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111, projection='3d')
# #     ax.plot(sliced_array[i, :, 0], sliced_array[i, :, 1], sliced_array[i, :, 2])
# #     ax.set_title(f"Trajectory {i}")
# #     ax.set_xlabel("X")
# #     ax.set_ylabel("Y")
# #     ax.set_zlabel("Z")
# #     plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # for i in range(6, sliced_array.shape[0]):
# i = 0
# ax.plot(sliced_array[i, :, 0], sliced_array[i, :, 1], sliced_array[i, :, 2], label=f"Trajectory {i}")

# # Adding labels and title
# ax.set_title("Trajectories 6 and 7")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Adding a legend to distinguish between the trajectories
# ax.legend()

# # Show the plot
# plt.show()

data = np.load("models/nb_3_cb_2_obs_RBF_100_reg_EDMDc.npy", allow_pickle=True)
print(data.item().observables)

# Check if data is a dictionary and print keys
if isinstance(data, dict):
    print("Data keys:", data.keys())

# Check for observables attribute in data
if isinstance(data, dict) and 'observables' in data:
    observables = data['observables']
    print("Observables:", observables)
elif hasattr(data, 'observables'):
    observables = data.observables
    print("Observables:", observables)
else:
    print("The observables attribute is missing from the data object.")