# If you do not have Ipopt, run the following command:
# Pkg.add("Ipopt")
# make sure you have JuMP as well!
# Pkg.add("JuMP")
using JuMP, Ipopt

# PATH TO YOUR DATASETS HERE
trainfile = "monks-problems_train.csv" 
testfile = "monks-problem_test.csv"

train = readcsv(trainfile)
test = readcsv(testfile)

X_train = train[:, 1:(end-1)]
Y_train = train[:, end]
X_test = test[:, 1:(end-1)]
Y_test = test[:, end]


#########################################################
# solve logistic regression (with robustification) ######
function solve_jump(X, Y; 
	lambda::Float64 	= 0.00, 
	gamma_pct::Float64 	= 0.00, 
	reg                 = false,
	norm 				= "l1", 
	t_limit 			= 1.0e10,
	beta_warm			= [],
	)


	n, p = size(X)
	gamma = gamma_pct * n
	Y = [yi == 1.0 ? 1.0 : -1.0 for yi in Y]

	robX = (lambda == 0 || reg == true  ? false:true)
	reg  = (lambda == 0 || reg == false ? false:true)
	robY = (gamma_pct == 0 	? false:true)

	m = Model(solver=IpoptSolver(print_level 	= 1, 
									max_iter	= 8000,
									max_cpu_time= t_limit
									))

	@defVar(m, fx[1:n])

	# warm start
	if beta_warm == [] 
		beta_warm = zeros(p)
	end
	@defVar(m, beta[i=1:p],   start = beta_warm[i])

	@addConstraint(m, fxConstr[i=1:n], fx[i] == sum{X[i, j]*beta[j], j= 1:p})

	if reg == true || robX == true # add norm constraints
		@defVar(m, hs_norm >= 0, start = 0.0)

		if norm == "l2"
			@addNLConstraint(m, hs_norm^2 >= sum{(beta[j])^2, j = 2:p})

		elseif norm == "l1"
			@defVar(m, z[1:p] >= 0, start = 0.0)
			@addConstraint(m, hs_norm == sum{z[j], j=2:p})

			@addConstraint(m, pos_abs[i = 2:p], z[i] >=  beta[i])
			@addConstraint(m, neg_abs[i = 2:p], z[i] >= -beta[i])
		end

	end


	# define the objective functions for different robust scenarios
	if     (robX == false && robY == false && reg == false)   ###### classical logistic

		@setNLObjective(m, Max, 
			-sum{log(1+exp(-Y[i]*fx[i])), i = 1:n})
		
	elseif (reg == true && robX == false) ###### regularized logistic
		# println("Norm is $norm")
		@setNLObjective(m, Max, 
			-sum{log(1+exp(-Y[i]*fx[i])), i = 1:n} 
				- 0.5*n*lambda*hs_norm)

	elseif (robX == true && robY == false) ###### robX

		@setNLObjective(m, Max, 
				-sum{log(1+exp(-Y[i]*fx[i] 
				+ lambda*hs_norm)), i = 1:n})

	elseif (robX == false && robY == true) ###### robY

		@defVar(m, mu <= 0, 	 start = 0.0)
		@defVar(m, nu[1:n] <= 0, start = 0.0)

		@setNLObjective(m, Max, 
			-sum{log(1+exp(-Y[i]*fx[i])), i = 1:n}
			+ gamma*mu + sum{nu[i], i = 1:n})

		@addNLConstraint(m, sum_mu_nu[i = 1:n], 
			mu + nu[i] <= log(1+exp(-Y[i]*fx[i])) 
						- log(1+exp( Y[i]*fx[i])))
 
	elseif (robX == true && robY == true) ##### global rob

		@defVar(m, mu <= 0, 	 start = 0.0)
		@defVar(m, nu[1:n] <= 0, start = 0.0)

		@setNLObjective(m, Max, 
			-sum{log(1+exp(-Y[i]*fx[i]
			+ lambda*hs_norm)), i = 1:n}
			+ gamma*mu + sum{nu[i], i = 1:n})

		@addNLConstraint(m, sum_mu_nu[i = 1:n], 
			mu + nu[i] <= log(1+exp(-Y[i]*fx[i] + lambda*hs_norm)) 
						- log(1+exp( Y[i]*fx[i] + lambda*hs_norm)))

	end

	status = solve(m)
	obj = getObjectiveValue(m)

	param = getValue(beta)[:]
	return(param)
end


# helper sigmoid function to calculate probability
function sigmoid(X)
    den = 1.0 + e ^ (-1.0 * X)
    d = 1.0 / den
    return d
end

# predict the probability of y = 1
function predict_logit(X, beta_est; binary = false)
	local y_pred = zeros(size(X, 1))
	for i = 1:size(X, 1)
	 	y_pred[i] = sigmoid(dot(vec(X[i, :]),beta_est))
	end
	y_pred = (binary == false? y_pred : round(y_pred))
	return(y_pred)
end

# calculate accuracy percentage
function calc_accu(beta, Xtest, Ytest)
	return countnz(Ytest .== predict_logit(Xtest, beta; binary = true))/length(Ytest)
end

##### classical 
tic()
beta = solve_jump(X_train, Y_train);
accu = calc_accu(beta, X_test, Y_test)
println("Nominal LR:  Time = ", toq(), "       OS-Accuracy = ", accu)
##### regularized
tic()
beta_reg = solve_jump(X_train, Y_train; lambda = 0.08, reg = true);
accu_reg = calc_accu(beta_reg, X_test, Y_test)
println("Regularized LR: Time = ", toq(), "    OS-Accuracy = ", accu_reg)
#####  robust X
tic()
beta_X = solve_jump(X_train, Y_train; lambda = 0.08);
accu_X = calc_accu(beta_X, X_test, Y_test)
println("Robust-X LR: Time = ", toq(), "       OS-Accuracy = ", accu_X)
#####  robust Y
tic()
beta_Y = solve_jump(X_train, Y_train; gamma_pct = 0.05);
accu_Y = calc_accu(beta_Y, X_test, Y_test)
println("Robust-Y LR: Time = ", toq(), "       OS-Accuracy = ", accu_Y)

