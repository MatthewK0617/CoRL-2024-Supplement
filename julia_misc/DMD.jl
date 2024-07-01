using LinearAlgebra, NPZ, Statistics, ProximalAlgorithms, ProximalOperators, Plots

# Load data
function load_crazyflie_dataset(file_path)
    data = npzread(file_path)
    return data
end

file_path = "np_data_final/traj_div2__x3_u2.npz"
crazyflie_data = load_crazyflie_dataset(file_path)

X0 = permutedims(crazyflie_data["X0"][:, :, 2:end], (3, 2, 1))
U = permutedims(crazyflie_data["U"][:, :, 2:end], (3, 2, 1))
X1 = permutedims(crazyflie_data["X1"][:, :, 2:end], (3, 2, 1))

println("Dimensions of X0: ", size(X0))
println("Dimensions of X1: ", size(X1))

num_states, num_time_points, num_trajectories = size(X0)

# Data matrices
X = reshape(X0[:, 1:end, :], num_states, :)
Y = reshape(X1[:, 1:end, :], num_states, :)
U_reshaped = reshape(U[:, 1:end, :], size(U, 1), :)

# Train/test split size
train_size = Int(floor(num_trajectories / 2)) + 2
test_size = num_trajectories - train_size

# Train data
X_train = reshape(X0[:, 1:end, 1:train_size], num_states, :)
Y_train = reshape(X1[:, 1:end, 1:train_size], num_states, :)
U_train = reshape(U[:, 1:end, 1:train_size], size(U, 1), :)

# Test data
X_test = reshape(X0[:, 1:end, train_size+1:end], num_states, :)
Y_test = reshape(X1[:, 1:end, train_size+1:end], num_states, :)
U_test = reshape(U[:, 1:end, train_size+1:end], size(U, 1), :)

# K and L
K_train = cov(X_train')
L_train = cov(U_train')

println("Dimensions of K_train: ", size(K_train))
println("Dimensions of L_train: ", size(K_train))


lambda = 0.00 # Regularization term

function predict(K, L, X, U)
    return K * X + L * U
end

function inf_norm_residuals(K, L, X, U, Y)
    residuals = Y - predict(K, L, X, U)
    return maximum(abs.(residuals))
end

function mse_residuals(K, L, X, U, Y)
    residuals = Y - predict(K, L, X, U)
    println(size(residuals))
    println(typeof(residuals))

    return mean(residuals .^ 2)
end

function loss(params_vec)
    K = reshape(params_vec[1:num_states*num_states], num_states, num_states)
    L = reshape(params_vec[num_states*num_states+1:end], num_states, size(U_train, 1))
    return mse_residuals(K, L, X_train, U_train, Y_train)
end

# Initial Mean Squared Error on the test set
mse = mse_residuals(K_train, L_train, X_test, U_test, Y_test)
println("Initial Mean Squared Error on the test set: $mse")

inr = inf_norm_residuals(K_train, L_train, X_test, U_test, Y_test)
println("Initial Infinity Norm on the test set: $inr")

# Regularization
reg = ProximalOperators.NormL2(lambda)

# Initial vectorized parameters (K and L)
params_vec_initial = vcat(vec(K_train), vec(L_train))

# Proximal gradient descent using FastForwardBackward
ffb = ProximalAlgorithms.FastForwardBackward(
    maxit=10000, 
    verbose=true,
    freq=1000,
    Lf=3000,
)

solution_vec, iterations = ffb(x0=params_vec_initial, f=loss, g=reg)

# Reshape the solution back to matrix form
K_train_optimized = reshape(solution_vec[1:num_states*num_states], num_states, num_states)
L_train_optimized = reshape(solution_vec[num_states*num_states+1:end], num_states, size(U_train, 1))

# Predict on test set
Y_pred = predict(K_train_optimized, L_train_optimized, X_test, U_test)

# Mean Squared Error on the test set
mse = mean((Y_pred - Y_test).^2)
println("Mean Squared Error on the test set: $mse")

# Infinity Norm on the test set
infinity_norm = inf_norm_residuals(K_train_optimized, L_train_optimized, X_test, U_test, Y_test)
println("Infinity Norm on the test set: $infinity_norm")
