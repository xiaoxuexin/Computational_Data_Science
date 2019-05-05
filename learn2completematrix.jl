
using Images, FileIO, Plots, Colors, Interact, Missings
plotly(;
    label="",
    markerstrokecolor="white",
    markerstrokewidth=0.2
)

A = [1;3;5]*[2;3;1]'
Atil = allowmissing(A)

Atil[[2,4,5,9]] = missing; ## these are linear indices
Atil

?completeAmanual

"""
    Ac = completeAmanual(A)
    
Inputs:
* `A` is a rank-one 3 x 3 matrix of the form
```
[a11  ?  a13
  ?   ?  a23 
 a31 a32  ? ]
```
where ? denotes a missing value.

Output: `Ac` is a completed rank-one 3 x 3 matrix
"""
function completeAmanual(A::Matrix)
    # Extract values
    a11, a31, a32, a13, a23 = A[1,1], A[3,1], A[3,2], A[1,3], A[2,3]
    
    # Calculate missing values manually
    a21 = (a23/a13)*a11
    a12 = (a32/a31)*a11
    a22 = (a21/a31)*a32
    a33 = (a31/a11)*a13
    
    return [a11 a12 a13; a21 a22 a23; a31 a32 a33]
end

Ac = completeAmanual(Atil)

@show A-Ac;

p = 0.5
Atil2 = allowmissing(A)
Atil2[rand(size(A)) .>= p] = missing
Atil2##TODO: Fill in the ??

completeAmanual(Atil2)


using Missings

"""
Inputs:
* `A`: matrix with missing entries -- type Missing, allowmissing
* `k`: rank of matrix
* `iters`: Number of iterations to run
* `error_tol`: error tolerance

Outputs:
* `A_hat`: Completed Matrix
* `err`: vector of relative Frobenius norm errors at each iteration
"""
function completeAsvd(A::Matrix, k::Number; iters::Integer = 1000, error_tol::Number = 1e-9)
    MissingEntries = ismissing.(A) # Location of missing entries
    NotMissingEntries = .!MissingEntries
    
    error = Inf
    err = Array{Float64}(0)
    
    idx = 1 # Iteration index
    
    A_hat = collect(Missings.replace(A, 0)) # Estimate
    
    while (error > error_tol && idx <= iters)
        
        A_hat[NotMissingEntries] = A[NotMissingEntries] # Force Known Entries: Projection Step
        
        UsV = svds(A_hat, nsv = k)[1]
        U, s, V = UsV[:U], UsV[:S], UsV[:V]
       
        A_hat = U * Diagonal(s) * V' # Rank k fit
        
        error = sqrt(sum(abs2, skipmissing(A - A_hat)) / sum(abs2, skipmissing(A))) # Normalized error on known entries
        push!(err,error)
        
        idx = idx + 1
        
    end
 
    return A_hat, err
end

@show Ac
Ac, error = completeAsvd(Atil,1,iters = 1)
@show error[end]
Ac

Ac, error = completeAsvd(Atil,1,iters = 2)
@show error[end]
Ac

Ac, error = completeAsvd(Atil,1,iters = 3000)
@show error[end]
Ac

scatter(error, yscale=:log10, xlabel="Iterations", ylabel="Error")

image = load("mitlogo.png")
image_matrix_gray = float(Gray.(image));
heatmap(image_matrix_gray, color =:grays, yflip =:true, ticks = [], clim=(0, 1))

p = 0.7
image_missing = allowmissing(image_matrix_gray)
image_missing[rand(size(image_matrix_gray)) .>= p] = missing
image_missing_zeros = collect(Missings.replace(image_missing, 0.0));
heatmap(image_missing_zeros, color=:grays, yflip=:true, ticks=[], clim=(0, 1))

k = 4
image_hat, error = completeAsvd(image_missing, k; iters=1)
heatmap(image_hat, color=:grays, yflip=:true, ticks=[], title="Error = $(round(error[end],5))")

image_hat, error = completeAsvd(image_missing, k; iters=2)
heatmap(image_hat, color=:grays, yflip=:true, ticks=[], title="Error = $(round(error[end],5))")

image_hat, error = completeAsvd(image_missing, k; iters=10)
heatmap(image_hat, color=:grays, yflip=:true, ticks=[], title="Error = $(round(error[end],5))")

@manipulate for p in [0.001, 0.2, 0.5, 0.8], k in [1, 2, 3, 4], num_iters in [1, 2, 10, 100]
    image_missing = allowmissing(image_matrix_gray) 
    image_missing[rand(size(image_matrix_gray)) .>= p] = missing
    image_hat, error = completeAsvd(image_missing, k; iters = num_iters)
    image_missing_zeros = collect(Missings.replace(image_missing,0.0));
    p1 = heatmap(image_missing_zeros, color=:grays, yflip=:true, ticks=[],
         title = "Matrix with missing entries")
    p2 = heatmap(image_hat, color=:grays, yflip=:true, ticks=[], 
        title="k = $k, Error = $(round(error[end],6))")
    plot(p1, p2, layout = (1,2))
end

m, n  = 100, 200 
x = rand(m)/sqrt(m)
y = rand(n)/sqrt(n)
A = x*y';

@show maximum(svdvals(A))
@show vecnorm(x)*vecnorm(y)
scatter(svdvals(A), xlabel="index", ylabel="singular value")

scatter(svdvals(A), yscale=:log10, xlabel="index", ylabel="singular value")

@show rank(A);

println("Click on the highlighted link to examine the function")
@which rank(A)

UsV = svds(A, nsv=1)[1]
u1 = UsV[:U]
v1 = UsV[:V]
@show abs.(u1'*x/vecnorm(x))
@show abs.(v1'*y/vecnorm(y))

showfull(io, x) = show(IOContext(io; compact=false, limit=false), x)
println("Inner-product between u1 and x/norm(x)")
println("")
showfull(STDOUT, abs.(u1'*x/vecnorm(x)))

@show svdvals(A)[2]

##TODO:
p = 0.5
n_list = [10, 50, 100, 200, 400]
maxAs = []
sigma1Deltas = []
for n in n_list
    x, y = sign.(randn(n)), sign.(randn(n))
    x, y = x/norm(x), y/norm(y)
    A = x*y'
    maxA = maximum(A)
    maxAs = push!(maxAs, maxA)
    Atil = allowmissing(A)
#     Atil = A[] ## make it missing
    Atil[rand(size(A)) .> p] = missing
    Ahat = collect(Missings.replace(Atil,0.0)) ## 
    Delta =Ahat - p.*A
    sigma1Delta = maximum(svdvals(Delta))
    sigma1Deltas = push!(sigma1Deltas, sigma1Delta)
end

p1 = scatter(n_list, maxAs, xlabel = "n", xscale=:log10, yscale=:log10, label="maxA")
p2 = scatter(n_list, sigma1Deltas, xlabel = "n", xscale=:log10, yscale=:log10, label="sigma1(Delta)")
plot(p1, p2, layout = (2,1))

p = 0.5
n_list = [10, 50, 100, 200, 400 ]
maxAs = []
sigma1Deltas = []
for n = n_list
    x, y = sign.(randn(n)), sign.(randn(n))
    x, y = x/norm(x), y/norm(y)
    A = x*y'
    maxA = maximum(A)
    maxAs = push!(maxAs, maxA)
    Atil = allowmissing(A)
#     Atil = A[??] ## make it missing
#     Ahat = collect(Missings.replace(Atil(,??))) ## 
#     Delta = ?? - ??
#     sigma1Delta = ??
#     sigma1Deltas = push!(sigma1Deltas, sigma1Delta) 
# end

    Atil[rand(size(A)) .> p] = missing
    Ahat = collect(Missings.replace(Atil,0.0)) ## 
    Delta =Ahat - p.*A
    sigma1Delta = maximum(svdvals(Delta))
    sigma1Deltas = push!(sigma1Deltas, sigma1Delta)
end


p1 = scatter(n_list, maxAs, xlabel = "n", xscale=:log10, yscale=:log10, label="maxA")
p2 = scatter(n_list, sigma1Deltas, xlabel = "n", xscale=:log10, yscale=:log10, label="sigma1(Delta)")
plot(p1,p2, layout = (2,1))

function generate_velocity_sensor_data(n::Integer, v₀=0, a=1, t₀=1, t₁=2)
    t = collect(linspace(t₀, t₁, n))
    s = v₀*t + 0.5*a*t.^2
    v = v₀*t + a*t
    return s, v, t
end

n = 100 
v₀, a = 0.5, 1
s, v, t = generate_velocity_sensor_data(n, v₀)


##TODO: Construct the matrix X from the variables
## Hint: To get the subscript 0 in v₀ type  "\_0" (without the quotes) followed by the "Tab" key
v0 = v₀ * ones(1,100)
A = a*ones(1,100)
X =  vcat(v',A,s',t',v0)

#TODO: Dispaly the singular values of X as a scatter plot and put in the title the rank of X
# Hint use the title = "$(rank(X))" keyword within the scatter command
singvalues = svdvals(X)
scatter(singvalues, title = "$(rank(X))")

v00 = 0
a = 1
A = ones(1,100)
s, v, t = generate_velocity_sensor_data(n, v00)
X1 =  vcat(v',A,s',t',zeros(1,100))
Xtil = X1[1:4,:]
Z = vcat((1/a)*Xtil[1,:]', a* ones(1,100), 2/a* sqrt.(Xtil[3,:])', Xtil[4,:]') ##TOD0

@show rank(Z) ## Is it equal to 2? 

using Images, FileIO, Plots, Colors, Interact
plotly()
image = load("mitlogo.png")

k = 4
p = 0.5 #0.5
picture = zeros(2)
image_matrix_gray = float(Gray.(image));
image_missing = allowmissing(image_matrix_gray) 
image_missing[rand(size(image_matrix_gray)) .>= p] = missing

num = 200
eV = zeros(num);eU = zeros(num)
UsV = svds(image_matrix_gray, nsv = k)[1]
U, s, V = UsV[:U], UsV[:S], UsV[:V]
for num_iters = 1:num
    image_hat, error = completeAsvd(image_missing, k; iters = num_iters)
    UsV_hat = svds(image_hat, nsv = k)[1]
    U_hat, s_hat, V_hat = UsV_hat[:U], UsV_hat[:S], UsV_hat[:V]
    eV[num_iters] = 1/4 * sum((diag(V_hat[:,1:4]'*V[:,1:4])).^2)
    eU[num_iters] = 1/4 * sum((diag(U_hat[:,1:4]'*U[:,1:4])).^2)
end
scatter(1:num,eV, label = "eV", marker=:square, ylabel = "Value", xlabel = "Iteration")
scatter!(1:num,eU, label = "eU ")
plot!(title = " p = $p")

"""
    Xh = optshrink1(Y, r)

Inputs:
* `Y`: 2D array where Y = X + noise and goal is to estimate X
* `r`: estimated rank of X

Output: `Xh`: rank-r estimate of X using OptShrink weights for SVD components

Perform rank-r denoising of data matrix Y using OptShrink
using the method described in the IEEE Transactions on Information paper:
http://doi.org/10.1109/TIT.2014.2311661

*This version works only if the size of Y is sufficiently small,
because it performs calculations involving arrays roughly of
size(Y'*Y) and size(Y*Y') so neither dimension of Y can be large.*
"""
function optshrink1{T<:Number}(Y::Array{T,2}, r::Number)

    (U, s, V) = svd(Y, thin=true)

    (m, n) = size(Y)
    r = minimum([r, m, n]) # ensure r <= min(m,n)

    # make rectangular diagonal "S" (\hat{\Sigma}_{\hat{r}}) (m-r)x(n-r) per paper
    if m >= n # tall
        # [(n-r)x(n-r); (m-n)x(n-r)] -> [(m-r)x(n-r)]
        S = [diagm(s[(r + 1):n]); zeros(m - n, n - r)]
    else # wide
        # [(m-r)x(m-r), (m-r)x(n-m)] -> [(m-r)x(n-r)]
        S = [diagm(s[(r + 1):m]) zeros(m - r,n - m)]
    end
    #@show size(S) # [(m-r) x (n-r)]

    w = zeros(r)
    for k=1:r
        (D, Dder) = D_transform_from_matrix(s[k], S)
        w[k] = -2*D/Dder
    end

    Xh = U[:,1:r] * Diagonal(w)*V[:,1:r]' # (m,r) (r,r) (n,r)'

    return Xh
end

"""
* `X` is n x m, where this internal n, m differ from n, m in calling routine

*this version makes "big" matrices so it is impractical for big data*
"""
function D_transform_from_matrix(z, X)
    (n, m) = size(X)
    In = eye(n)
    Im = eye(m)


    D1 = ??
    D2 = ??

    D = D1*D2 # eq (16a) in  paper

    # derivative of D transform
    D1_der = ??
    D2_der = ??

    D_der = D1*D2_der + D2*D1_der # eq (16b) in paper

    return (D, D_der)
end
