function sorted_correlations(cor_matrix)
	c = copy(abs(cor_matrix))
	p = length(cor_matrix[1,:])
	num_pairs = convert(Int64,p*(p-1)/2)
	pair_list = zeros(Int64, num_pairs, 2)
	magnitude = zeros(num_pairs)

	# Set lower triangular correlation values = 0
	for i=1:p
		for j=1:i
			c[i,j] = 0
		end
	end

	for i=1:num_pairs
		ind = indmax(c)
		col = floor(ind/p) + 1
		row = ind % p
		magnitude[i] = c[ind]
		pair_list[i,1] = row
		pair_list[i,2] = col
		c[ind] = 0
	end

	return(pair_list, magnitude)
end

function linear_model(X_train, y_train, soln)
	K_opt = countnz(soln)
	df_train = DataFrame([X_train[:,soln] y_train])
	rename!(df_train, names(df_train)[end], :y)

	if K_opt == 1
		return lm(y ~ x1, df_train)
	elseif K_opt == 2
		return lm(y ~ x1+x2, df_train)
	elseif K_opt == 3
		return lm(y ~ x1+x2+x3, df_train)
	elseif K_opt == 4
		return lm(y ~ x1+x2+x3+x4, df_train)
	elseif K_opt == 5
		return lm(y ~ x1+x2+x3+x4+x5, df_train)
	elseif K_opt == 6
		return lm(y ~ x1+x2+x3+x4+x5+x6, df_train)
	elseif K_opt == 7
		return lm(y ~ x1+x2+x3+x4+x5+x6+x7, df_train)
	end
	println("Error in statistical significance test: K_opt = $K_opt > 6")
end

# Check if all features in linear regression are
# statistically significant
function stat_sig(X_train, y_train, soln)
	K_opt = countnz(soln)
	df_train = DataFrame([X_train[:,soln] y_train])
	rename!(df_train, names(df_train)[end], :y)

	if K_opt == 0
		return false
	elseif K_opt == 1
		model = lm(y ~ x1, df_train)
	elseif K_opt == 2
		model = lm(y ~ x1+x2, df_train)
	elseif K_opt == 3
		model = lm(y ~ x1+x2+x3, df_train)
	elseif K_opt == 4
		model = lm(y ~ x1+x2+x3+x4, df_train)
	elseif K_opt == 5
		model = lm(y ~ x1+x2+x3+x4+x5, df_train)
	elseif K_opt == 6
		model = lm(y ~ x1+x2+x3+x4+x5+x6, df_train)
	elseif K_opt == 7
		return lm(y ~ x1+x2+x3+x4+x5+x6+x7, df_train)
	else
		println("Error in statistical significance test: K_opt = $K_opt > 7")
		return false
	end

	conf_int = confint(model)[2:end,:]

	for i=1:length(conf_int[:,1])
		# Check if this confidence interval contains 0
		if conf_int[i,1] < 0 && 0 < conf_int[i,2]
			return false
		end
	end

	return true
end