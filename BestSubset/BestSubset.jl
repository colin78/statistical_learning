using JuMP, Gurobi
include("FirstOrderHeuristic.jl")
include("BestSubset_helper_functions.jl")
tic()
####################################
# CHANGE THE FOLLOWING FOR YOUR PARTICULAR DATA:

# BIG M CONSTANT
M = 50

# PATH TO YOUR DATASETS HERE
trainfile = "Train lpga2009_opt.csv" 
validationfile = "Validation lpga2009_opt.csv" 
testfile = "Test lpga2009_opt.csv"

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
K_options = reverse([1:D])

# find the correlation matrix of the independent variables of the training data
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
threshold_multicol = 0.5

for i=1:num_pairs
	if magnitude[i] > threshold_multicol
		@addConstraint(m, z[pair_list[i,1]] + z[pair_list[i,2]] <= 1)
	else
		break
	end
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

for K in K_options
	println("\nstarting to solve k = ", K)
	chgConstrRHS(sparsity, K) #Sparsity constraint
	try
		println("getting warm start solution")
		betaWarm = WarmStart(X_train, y_train, K)
		zWarm = 1*(betaWarm .!= 0)
		for d=1:D
			setValue(Beta[d], betaWarm[d])
			setValue(z[d], zWarm[d])
		end
		println("set warm start solution")
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
						
		if (newR2 > bestR2)					
			bestR2 = newR2
			bestBetavals = Betavals[K,:]
		end		
	end	
end


# Out of sample testing
y_hat_test = X_test*bestBetavals'
RSS_test = sum((y_hat_test - y_test).^2)
R2_test = 1- RSS_test/SST_test
MIO_nonzeros = find(abs(bestBetavals) .> 0.00001)
best_K = length(MIO_nonzeros)
max_corr = maximum(abs(cor_matrix[MIO_nonzeros,MIO_nonzeros]) - eye(best_K))


println("\n***RESULTS***")
println("N:\t", N)
println("D:\t", D)
println("best K:\t", best_K)
println("max corr:\t", max_corr)
println("MIO R2 test\t", R2_test)
toc()