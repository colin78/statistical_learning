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