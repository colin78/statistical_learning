using JuMP, Gurobi

# PATH TO YOUR DATASETS HERE
trainfile = "monks-problems_train.csv" 
testfile = "monks-problem_test.csv"

train = readcsv(trainfile)
test = readcsv(testfile)

X_train = train[:, 1:(end-1)]
Y_train = train[:, end]
X_test = test[:, 1:(end-1)]
Y_test = test[:, end]

# Convert Y's from 0/1 to -1/+1
Y_train = 2*Y_train-1
Y_test = 2*Y_test-1

function svm(x, y; model="hinge_loss", rho=0, gamma_perc=0,
			M=10000, OutputFlag=1, TimeLimit=600,
			warm_w=0, warm_b=0, LogFile="", Threads=0)
	m = Model(solver=GurobiSolver(TimeLimit=TimeLimit, OutputFlag=OutputFlag, LogFile=LogFile,Threads=Threads))
	n,p = size(x)
	@defVar(m, eps[1:n] <= 0)
	@defVar(m, w[1:p])
	@defVar(m, b)
	if(warm_w!=0)
		setValue(w, warm_w)
		setValue(b, warm_b)
	end

	@defExpr(d[i=1:n], y[i]*(sum{x[i,j]*w[j], j=1:p} - b))

	if model == "hinge_loss"
		@addConstraint(m, eps_lt[i=1:n], eps[i] <= d[i] - 1)
		@setObjective(m, Max, sum(eps))

	# Robust Features: L-1 norm uncertainty set
	elseif model == "robX"
		@defVar(m, winf >= 0)
		@addConstraint(m, pos_abs[j=1:p], winf >= w[j])
		@addConstraint(m, neg_abs[j=1:p], winf >= -w[j])

		@addConstraint(m, eps_lt[i=1:n], eps[i] <= d[i] - 1 - rho*winf)
		@setObjective(m, Max, sum(eps))

	# Robust Labels: L-1 norm uncertainty set
	elseif model == "robY"
		@defVar(m, phi[1:n] <= 0)
		@defVar(m, s[1:n], Bin)
		@defVar(m, t[1:n], Bin)
		@defVar(m, q <= 0)
		@defVar(m, r[1:n] <= 0)

		@addConstraint(m, qr_lt[i=1:n], q + r[i] <= phi[i] - eps[i])

		@addConstraint(m, eps_lt[i=1:n], eps[i] <= d[i] - 1)
		@addConstraint(m, phi_lt[i=1:n], phi[i] <= -d[i] - 1)
		@addConstraint(m, esp_gt1[i=1:n], eps[i] >= -M*s[i])
		@addConstraint(m, esp_gt2[i=1:n], eps[i] >= d[i] - 1 - M*(1-s[i]))
		@addConstraint(m, phi_gt1[i=1:n], phi[i] >= -M*t[i])
		@addConstraint(m, phi_gt2[i=1:n], phi[i] >= -d[i] - 1 - M*(1-t[i]))

		@setObjective(m, Max, sum(eps) + gamma_perc*n*q + sum(r))

	# Robust in both Features and Labels: L-1 norm uncertainty set
	elseif model == "robXY"
		@defVar(m, phi[1:n] <= 0)
		@defVar(m, s[1:n], Bin)
		@defVar(m, t[1:n], Bin)
		@defVar(m, q <= 0)
		@defVar(m, r[1:n] <= 0)

		@defVar(m, winf >= 0)
		@addConstraint(m, pos_abs[j=1:p], winf >= w[j])
		@addConstraint(m, neg_abs[j=1:p], winf >= -w[j])

		@addConstraint(m, qr_lt[i=1:n], q + r[i] <= phi[i] - eps[i])

		@addConstraint(m, eps_lt[i=1:n], eps[i] <= d[i] - 1 - rho*winf)
		@addConstraint(m, phi_lt[i=1:n], phi[i] <= -d[i] - 1 - rho*winf)
		@addConstraint(m, esp_gt1[i=1:n], eps[i] >= -M*s[i])
		@addConstraint(m, esp_gt2[i=1:n], eps[i] >= d[i] - 1 - M*(1-s[i]) - rho*winf)
		@addConstraint(m, phi_gt1[i=1:n], phi[i] >= -M*t[i])
		@addConstraint(m, phi_gt2[i=1:n], phi[i] >= -d[i] - 1 - M*(1-t[i]) - rho*winf)

		@setObjective(m, Max, sum(eps) + gamma_perc*n*q + sum(r))
	end

	status = solve(m)
	isOptimal = (status == :Optimal)

	return(getValue(w)[:], getValue(b), isOptimal)
end

w_opt, b_opt, is_opt = svm(X_train, Y_train, model="robXY", rho=0.1, gamma_perc=0.1)