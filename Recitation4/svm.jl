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

function svm(x, y; model="hinge_loss", C=0, rho=0, gamma_perc=0,
			M=10000, OutputFlag=0, TimeLimit=600,
			warm_w=0, warm_b=0, LogFile="", Threads=0)
	m = Model(solver=GurobiSolver(TimeLimit=TimeLimit, OutputFlag=OutputFlag, LogFile=LogFile,Threads=Threads))
	n,p = size(x)
	@defVar(m, eps[1:n] >= 0)
	@defVar(m, w[1:p])
	@defVar(m, b)
	if(warm_w!=0)
		setValue(w, warm_w)
		setValue(b, warm_b)
	end

	@defExpr(d[i=1:n], y[i]*(sum{x[i,j]*w[j], j=1:p} - b))

	if model == "hinge_loss"
		@addConstraint(m, eps_gt[i=1:n], eps[i] >= 1 - d[i])
		@setObjective(m, Min, sum(eps))

	# Classical SVM: L-2 regularizer term
	elseif model == "classical"
		@addConstraint(m, eps_gt[i=1:n], eps[i] >= 1 - d[i])
		@setObjective(m, Min, 0.5*sum{w[j]*w[j], j=1:p} + C*sum(eps))

	# Robust Features: L-1 norm uncertainty set
	elseif model == "robX"
		@defVar(m, winf >= 0)
		@addConstraint(m, pos_abs[j=1:p], winf >= w[j])
		@addConstraint(m, neg_abs[j=1:p], winf >= -w[j])

		@addConstraint(m, eps_gt[i=1:n], eps[i] >= 1 - d[i] + rho*winf)
		@setObjective(m, Min, sum(eps))

	# Mixed-Integer Optimization Formulations
	else
		@defVar(m, r[1:n] >= 0)
		@defVar(m, phi[1:n] >= 0)
		@defVar(m, q >= 0)
		@defVar(m, s[1:n], Bin)
		@defVar(m, t[1:n], Bin)
		
		@addConstraint(m, qr_gt[i=1:n], q + r[i] >= phi[i] - eps[i])
		@addConstraint(m, esp_lt1[i=1:n], eps[i] <= M*s[i])
		@addConstraint(m, phi_lt1[i=1:n], phi[i] <= M*t[i])
		@setObjective(m, Min, sum(eps) + gamma_perc*n*q + sum(r))

		# Robust Labels: L-1 norm uncertainty set
		if model == "robY"
			@addConstraint(m, eps_gt[i=1:n], eps[i] >= 1 - d[i])
			@addConstraint(m, phi_gt[i=1:n], phi[i] >= 1 + d[i])
			@addConstraint(m, esp_lt2[i=1:n], eps[i] <= 1 - d[i] + M*(1-s[i]))
			@addConstraint(m, phi_lt2[i=1:n], phi[i] <= 1 + d[i] + M*(1-t[i]))

		# Robust in both Features and Labels: L-1 norm uncertainty set
		elseif model == "robXY"
			@defVar(m, winf >= 0)
			@addConstraint(m, pos_abs[j=1:p], winf >= w[j])
			@addConstraint(m, neg_abs[j=1:p], winf >= -w[j])

			@addConstraint(m, eps_gt[i=1:n], eps[i] >= 1 - d[i] + rho*winf)
			@addConstraint(m, phi_gt[i=1:n], phi[i] >= 1 + d[i] + rho*winf)
			@addConstraint(m, esp_lt2[i=1:n], eps[i] <= 1 - d[i] + M*(1-s[i]) + rho*winf)
			@addConstraint(m, phi_lt2[i=1:n], phi[i] <= 1 + d[i] + M*(1-t[i]) + rho*winf)
		end
	end

	solve(m)

	return(getValue(w)[:], getValue(b))
end

function dsvm(x, y; model="hinge_loss", rho=0,
			M=10000, OutputFlag=0, TimeLimit=600,
			warm_w=0, warm_b=0, LogFile="", Threads=0)
	m = Model(solver=GurobiSolver(TimeLimit=TimeLimit, OutputFlag=OutputFlag, LogFile=LogFile,Threads=Threads))
	n,p = size(x)
	@defVar(m, z[1:n], Bin)
	@defVar(m, w[1:p])
	@defVar(m, b)
	if(warm_w!=0)
		setValue(w, warm_w)
		setValue(b, warm_b)
	end

	@defExpr(d[i=1:n], y[i]*(sum{x[i,j]*w[j], j=1:p} - b))

	if model == "hinge_loss"
		@addConstraint(m, z_gt[i=1:n], d[i] >= 1 - M*z[i])
	
	# Robust Features: L-1 norm uncertainty set
	elseif model == "robX"
		@defVar(m, winf >= 0)
		@addConstraint(m, pos_abs[j=1:p], winf >= w[j])
		@addConstraint(m, neg_abs[j=1:p], winf >= -w[j])
		@addConstraint(m, z_gt[i=1:n], d[i] - rho*winf >= 1 - M*z[i])
	end

	@setObjective(m, Min, sum(z))
	solve(m)

	return(getValue(w)[:], getValue(b))
end

# predict the labels y = -1/+1
function predict_svm(X_test, w, b)
	y_pred = sum(X_test .* w',2) - b .>= 0
	
	return 2*y_pred-1
end

function calc_accu(X_test, Y_test, w, b)
	return countnz(Y_test .== predict_svm(X_test, w, b))/length(Y_test)
end

println("-----------Regular SVM-----------")
##### Nominal
tic()
w, b = svm(X_train, Y_train)
accu = calc_accu(X_test, Y_test, w, b)
println("Nominal: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)
##### Classical SVM
tic()
w, b = svm(X_train, Y_train, model="classical", C=1)
accu = calc_accu(X_test, Y_test, w, b)
println("Classical SVM: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)
#####  Robust X
tic()
w, b = svm(X_train, Y_train, model="robX", rho=0.1)
accu = calc_accu(X_test, Y_test, w, b)
println("Robust-X SVM: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)
#####  Robust Y
tic()
w, b = svm(X_train, Y_train, model="robX", gamma_perc=0.1)
accu = calc_accu(X_test, Y_test, w, b)
println("Robust-Y SVM: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)
#####  Robust in Both
tic()
w, b = svm(X_train, Y_train, model="robXY", rho=0.1, gamma_perc=0.1);
accu = calc_accu(X_test, Y_test, w, b)
println("Robust-in-both SVM: \tTime = ", toq(), "\tOS-Accuracy = ", accu)

println("\n-----------Discrete SVM-----------")
##### Nominal
tic()
w, b = dsvm(X_train, Y_train)
accu = calc_accu(X_test, Y_test, w, b)
println("Nominal: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)
#####  Robust X
tic()
w, b = dsvm(X_train, Y_train, model="robX", rho=0.1);
accu = calc_accu(X_test, Y_test, w, b)
println("Robust-X DSVM: \t\tTime = ", toq(), "\tOS-Accuracy = ", accu)