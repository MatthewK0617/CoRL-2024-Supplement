using LinearAlgebra, NPZ, Statistics, ProximalAlgorithms, ProximalOperators, Plots, Combinatorics

# Load data
function load_crazyflie_dataset(file_path)
    data = npzread(file_path)
    return data
end

function Poly(x, d)
    state_dim, time = size(x)
    
    # gen all polynomial combinations up to degree d
    function polynomial_combinations(vec, degree)
        terms = []
        for i in 1:degree
            for comb in with_replacement_combinations(vec, i)
                push!(terms, prod(comb))
            end
        end
        return terms
    end
    
    expanded = [polynomial_combinations(x[:, t], d) for t in 1:time]
    expanded_matrix = hcat(expanded...)

    return expanded_matrix
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

d = 2 # Polynomial degree

# Lifted training data
X_train_lifted = Poly(X_train, d)
Y_train_lifted = Poly(Y_train, d)
U_train_lifted = Poly(U_train, d) 

# Lifted testing data
X_test_lifted = Poly(X_test, d)
Y_test_lifted = Poly(Y_test, d)  
U_test_lifted = Poly(U_test, d)  

# Initialize K and L for lifted dimensions
num_lifted_features_x = size(X_train_lifted, 1)
num_lifted_features_u = size(U_train_lifted, 1)
num_states_lifted = size(Y_train_lifted, 1)

# Initial random matrices K and L
K_train = rand(num_states_lifted, num_lifted_features_x)
L_train = rand(num_states_lifted, num_lifted_features_u)

println("Dimensions of K_train: ", size(K_train))
println("Dimensions of L_train: ", size(L_train))

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
    return mean(residuals .^ 2)
end

function loss(params_vec)
    K = reshape(params_vec[1:num_states_lifted*num_lifted_features_x], num_states_lifted, num_lifted_features_x)
    L = reshape(params_vec[num_states_lifted*num_lifted_features_x+1:end], num_states_lifted, num_lifted_features_u)
    return mse_residuals(K, L, X_train_lifted, U_train_lifted, Y_train_lifted)
end

# Initial vectorized parameters (K and L)
params_vec_initial = vcat(vec(K_train), vec(L_train))

# Proximal gradient descent using FastForwardBackward
ffb = ProximalAlgorithms.FastForwardBackward(
    maxit=20000, 
    verbose=true,
    freq=1000,
    Lf=3000,
)

solution_vec, iterations = ffb(x0=params_vec_initial, f=loss, g=reg)

K_train_optimized = reshape(solution_vec[1:num_states_lifted*num_lifted_features_x], num_states_lifted, num_lifted_features_x)
L_train_optimized = reshape(solution_vec[num_states_lifted*num_lifted_features_x+1:end], num_states_lifted, num_lifted_features_u)

# Predict on test set
Y_pred = predict(K_train_optimized, L_train_optimized, X_test_lifted, U_test_lifted)

# Mean Squared Error on the test set
mse = mean((Y_pred - Y_test_lifted).^2)
println("Mean Squared Error on the test set: $mse")

# Infinity Norm on the test set
infinity_norm = inf_norm_residuals(K_train_optimized, L_train_optimized, X_test_lifted, U_test_lifted, Y_test_lifted)
println("Infinity Norm on the test set: $infinity_norm")