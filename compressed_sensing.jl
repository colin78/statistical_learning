using JuMP, Gurobi, Distributions

function compressed_senser(A, b; TimeLimit=60, OutputFlag=1, L1_norm=false, M=1000)
	model = Model(solver=GurobiSolver(TimeLimit=TimeLimit, OutputFlag=OutputFlag))
	m,n = size(A)

	@defVar(model, x[1:n])

	@addConstraint(model, A*x .== b)

	if(L1_norm)
		@defVar(model, y[1:n])
		@addConstraint(model, y_pos[i=1:n], y[i] >= x[i])
		@addConstraint(model, y_neg[i=1:n], y[i] >= -x[i])
		setObjective(model, :Min, sum(y))
		solve(model)
		return(norm(getValue(x),0), getValue(x))
	else
		@defVar(model, z[1:n], Bin)
		@addConstraint(model, x_lt[i=1:n], x[i] <= M*z[i])
		@addConstraint(model, x_gt[i=1:n], x[i] >= -M*z[i])
		setObjective(model, :Min, sum(z))
		solve(model)
		return(round(getObjectiveValue(model)), getValue(x))
	end
end