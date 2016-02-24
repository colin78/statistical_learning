include("hw1_functions.jl")

seed = 0
m = 15
n = 20
k = 14



# x0 = [ones(k);zeros(n-k)]
# b = A[1:m,:]*x0
# nonzeros_L0[m, k], x_opt = compressed_senser(A[1:m,:], b, L1_norm=false, OutputFlag=1)

# srand(seed)
# A = rand(Normal(), n, n)
# nonzeros_L0 = zeros(Int, n, n)
# nonzeros_L1 = zeros(Int, n, n)

# for m=1:n
# 	for k=1:m
# 		x0 = [ones(k);zeros(n-k)]
# 		b = A[1:m,:]*x0
# 		nonzeros_L0[m, k], x_opt = compressed_senser(A[1:m,:], b, L1_norm=false, OutputFlag=0)
# 		nonzeros_L1[m, k], x_opt = compressed_senser(A[1:m,:], b, L1_norm=true, OutputFlag=0)
# 		println("m = $m, k = $k")
# 	end
# end

df_L0 = convert(DataFrame, nonzeros_L0)
df_L1 = convert(DataFrame, nonzeros_L1)

writetable("nonzeros_L0.csv", df_L0, header=false)
writetable("nonzeros_L1.csv", df_L1, header=false)