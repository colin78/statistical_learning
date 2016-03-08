using JuMP, Gurobi, GLM, DataFrames
include("FirstOrderHeuristic.jl")
include("BestSubset_helper_functions.jl")
tic()
####################################
# CHANGE THE FOLLOWING FOR YOUR PARTICULAR DATA:

# BIG M CONSTANT
M = 50

# PATH TO YOUR DATASETS HERE
trainfile = "Data/Train lpga2008_opt.csv" 
validationfile = "Data/Validation lpga2008_opt.csv" 
testfile = "Data/Test lpga2008_opt.csv"

####################################

train = readcsv(trainfile)
validation = readcsv(validationfile)
test = readcsv(testfile)
print("\nDone reading in data\n")

y_train = train[:, 1]
X_train = train[:,2:end]
y_validation = validation[:, 1]
X_validation = validation[:,2:end]
y_test = test[:,1]
X_test = test[:, 2:end]
N, D = size(X_train)

# Add nonlinear transformations
num_opt = 4 # number of distinct transformations for feature X
if num_opt > 1
	X_train = [X_train X_train.^2 sqrt(X_train) log(X_train)]
	X_validation = [X_validation X_validation.^2 sqrt(X_validation) log(X_validation)]
	X_test = [X_test X_test.^2 sqrt(X_test) log(X_test)]
end
D_orig = D
D = num_opt*D

# determine centering and scaling factors
mean_X_train = mean(X_train,1);
mean_y_train = mean(y_train);
X_train = X_train .- mean_X_train;
denom_X_train = zeros(D);
for i in 1:D
	denom_X_train[i] = norm(X_train[:,i]);
end

# center and scale the datasets
X_train = X_train ./ denom_X_train';
X_validation = (X_validation .- mean_X_train)./denom_X_train'
X_test = (X_test .- mean_X_train)./denom_X_train'
y_train = y_train .- mean_y_train;
y_validation = y_validation .- mean_y_train
y_test = y_test .- mean_y_train

SST_test = sum((mean(y_train) - y_test).^2)
K_options = [1:D_orig]
rho_options = [0; 10.0.^[-6:2]] # robustness parameter range

# Find the correlation matrix of the independent variables
# of the training data
cor_matrix = cor(X_train)
pair_list, magnitude = sorted_correlations(cor_matrix)
num_pairs = length(magnitude)

#######
# Build MIO optimization model

print("Building optimization model\n")

m = Model(solver = GurobiSolver(OutputFlag=0))

@defVar(m, Beta[1:D]);
@defVar(m, z[1:D], Bin);


# Big M constraints
@addConstraint(m, m_gt[d=1:D], Beta[d] <= M*z[d]);
@addConstraint(m, m_lt[d=1:D], -M*z[d] <= Beta[d]);

# Sparsity constraint
@addConstraint(m, sparsity, sum{z[d], d=1:D} <= K_options[1])

# Pairwise multicolinearity constraint
threshold_multicol = 0.8

for i=1:num_pairs
	if magnitude[i] > threshold_multicol
		@addConstraint(m, z[pair_list[i,1]] + z[pair_list[i,2]] <= 1)
	else
		break
	end
end

# Group sparsity constraint
group_sparsity = false
groups = ([1 2 3 4], [5 6 7], [8 9 10 11])

if group_sparsity
	for i=1:length(groups)
		num_grp = length(groups[i])
		@addConstraint(m, sum{z[groups[i][j]], j=1:num_grp} <= 1)
	end
end

# Single choice of nonlinear transformation constraint
if num_opt > 1
	@addConstraint(m, non_linear[j=1:D_orig], sum{z[j + t*D_orig], t=0:(num_opt-1)} <= 1)
end

# Objective function
a = 0
for i=1:N
	a += 0.5(y_train[i] - dot(Beta, vec(X_train[i,:])))^2
end
setObjective(m, :Min, a)

print("\nDone building optimization model\n")

Betavals = zeros(D,D)
MIO_num_real_RSS = 0
bestR2 = 0
R2_test = 0
bestBetavals = zeros(D)
bestZ = zeros(D)

# Robustness
robustness = true
best_rho = 0.01
if robustness
	@defVar(m, Beta_abs[1:D] >= 0)
	@addConstraint(m, beta_pos[j=1:D], Beta_abs[j] >= Beta[j])
	@addConstraint(m, beta_neg[j=1:D], Beta_abs[j] >= -Beta[j])
end

# Statistical significance variables
verbose = true
checkSig = false
bestR2_sig = 0
bestBetavals_sig = zeros(D)

for rho in rho_options
	# Add robustness by adding L-1 regularizer term
	if robustness
		@setObjective(m, :Min, a + rho*sum(Beta_abs))
	end
	println()

	significant = false

	K = 1
	max_runs = 10
	while K <= D
		significant = false

		if verbose
			println("starting to solve rho = $rho, k = $K")
		end
		chgConstrRHS(sparsity, K) #Sparsity constraint
		try
			# println("getting warm start solution")
			betaWarm = WarmStart(X_train, y_train, K)
			zWarm = 1*(betaWarm .!= 0)
			for d=1:D
				setValue(Beta[d], betaWarm[d])
				setValue(z[d], zWarm[d])
			end
			# println("set warm start solution")
			status = solve(m)
		catch
			println("*******STATUS ISSUE*****", status)
		finally	
			for j=1:D
				Betavals[K, j] = getValue(Beta[j])
			end

			y_hat_validation = X_validation*Betavals[K,:]'
			RSS_current = sum((y_hat_validation - y_validation).^2)
			SST = sum((y_validation - mean(y_train)).^2)

			newR2 = 1-RSS_current/SST

			if(newR2 > bestR2)
				bestR2 = newR2
				bestBetavals = Betavals[K,:]'
				bestZ = getValue(z)[:]
				best_rho = rho
			end
		end

		if !checkSig
			bestR2_sig = bestR2
			bestBetavals_sig = bestBetavals
			K += 1
		else
			soln = 1*(abs(bestBetavals) .> 0.00001)
			significant = stat_sig(X_train, y_train, soln)

			if significant
				bestR2_sig = bestR2
				bestBetavals_sig = bestBetavals
				best_rho = rho
				K += 1
			else
				# Add a constraint to exclude this solution 
				# during the next MIP solve
				@addConstraint(m, sum{z[j]*soln[j] + (1-z[j])*(1-soln[j]), j=1:D} <= D - 1)
				# Reset best R2
				bestR2 = bestR2_sig

				# For each K, only perform a maximum number of runs
				max_runs -= 1
				if max_runs < 0
					K += 1
				end
			end
		end
	end

	if !robustness
		break
	end
end

# Out of sample testing
soln = abs(bestBetavals_sig) .> 0.00001
y_hat_test = X_test*bestBetavals_sig
RSS_test = sum((y_hat_test - y_test).^2)
R2_test = 1- RSS_test/SST_test
best_K = sum(soln)
max_corr = maximum(abs(cor_matrix[soln,soln]) - eye(best_K))


println("\n***RESULTS***")
println("N:\t", N)
println("D:\t", D)
println("best K:\t", best_K)
println("best rho:\t", best_rho)
println("max corr:\t", max_corr)
println("MIO R2 test\t", R2_test)
toc()
stat_sig(X_train, y_train, soln)