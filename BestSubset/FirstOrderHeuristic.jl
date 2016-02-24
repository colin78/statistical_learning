
# #first order heuristic for warm starts

function WarmStart(X, y, K)

	Lipsh = norm(X)^2
	betaLS = X\y
	
	#make it feasible - Find K largest (absolute) values of betas
	indices_to_keep = reverse(sortperm(abs(betaLS)))[1:K]					
	betak = betaLS*0
	betaknew = betak #initialize
	betak[indices_to_keep] = betaLS[indices_to_keep]
	
	maxiter = 100
	obj_vals = zeros(maxiter)
	LB_vals = zeros(maxiter)
	sparsity = zeros(maxiter)
	
	for i=1:maxiter
		grad = -X'*(y - X*betak)
		betakold = betak
		vec = betak - grad/Lipsh
		indices_to_keep = reverse(sortperm(abs(vec)))[1:K]
		betak = betaLS*0
		betak[indices_to_keep] = vec[indices_to_keep]
		#now get better UB
		betaknew = betak
		DelT = betak - betakold
		numerator = (X*DelT)'*(y-(X*betakold))
		denominator = norm(X*DelT)^2
		alpha_step = 0
		try
			alpha_step = numerator/denominator
		catch
			alpha_step = 0
		end	

		betak = betakold + alpha_step.*DelT 
		sparsity[i] = sum(abs(betak .> 0.0000001))
		obj_vals[i] = 0.5*norm(y-X*betaknew)^2
		LB_vals[i] = 0.5*norm(y-X*betak)^2
		if obj_vals[i] - LB_vals[i] < 0.00001
			break
		end	
	end
	
	return(sum(isnan(betaknew)) > 0 ? betaLS*0 : betaknew)

end




