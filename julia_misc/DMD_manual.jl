using LinearAlgebra, NPZ, Statistics

# load data
function load_crazyflie_dataset(file_path)
    data = npzread(file_path)
    return data
end

file_path = "np_data_final/traj_div2__x3_u2.npz"
crazyflie_data = load_crazyflie_dataset(file_path)

X0 = permutedims(crazyflie_data["X0"][:, :, 2:end], (3, 2, 1))
X1 = permutedims(crazyflie_data["X1"][:, :, 2:end], (3, 2, 1))

println("Dimensions of X0: ", size(X0))
println("Dimensions of X1: ", size(X1))

num_states, num_time_points, num_trajectories = size(X0)

# data matrices
X = reshape(X0[:, 1:end-1, :], num_states, :)
Y = reshape(X1[:, 2:end, :], num_states, :)

#  train/test split size
train_size = Int(floor(num_trajectories / 2))
test_size = num_trajectories - train_size

# train data
X_train = reshape(X0[:, 1:end-1, 1:train_size], num_states, :)
Y_train = reshape(X1[:, 2:end, 1:train_size], num_states, :)

# test data
X_test = reshape(X0[:, 1:end-1, train_size+1:end], num_states, :)
Y_test = reshape(X1[:, 2:end, train_size+1:end], num_states, :)

# K
K_train = rand(num_states, num_states)

# grad. descent hyper-params
learning_rate = 0.01
num_iterations = 10000
lambda = 0.1 # regularization term

# calculate infinity norm of residuals
function inf_norm_residuals(K, X, Y)
    residuals = Y - (K * X)
    return maximum(abs.(residuals))
end

# proximal gradient descent (min(infinity_norm))
for iter in 1:num_iterations
    residuals = Y_train - K_train * X_train 
    min_residual_idx = argmin(abs.(residuals)) # min infinity norm

    # gradient computation of the infinity norm w.r.t. K
    gradient = zeros(size(K_train))
    for i in 1:num_states # 7
        for j in 1:num_states # 7
            gradient[i, j] = sign(residuals[min_residual_idx[1], min_residual_idx[2]]) * X_train[j, min_residual_idx[2]]
        end
    end

    gradient += lambda * sign.(K_train) # add reg. term to gradient
    K_train -= learning_rate * gradient # update K using gradient

    # Apply proximal operator (soft thresholding) for L1 regularization
    # K_train = sign.(K_train) .* max.(abs.(K_train) .- learning_rate * lambda, 0.0)

    if iter % 100 == 0
        println("Iteration $iter: Infinity Norm=", inf_norm_residuals(K_train, X_train, Y_train))
    end
end

# predict on test set
Y_pred = K_train * X_test

# MSE(test_set)
mse = mean((Y_pred - Y_test).^2)
println("Mean Squared Error on the test set: $mse")

# infinity_norm(test_set)
infinity_norm = inf_norm_residuals(K_train, X_test, Y_test)
println("Infinity Norm on the test set: $infinity_norm")

(mse, infinity_norm)
