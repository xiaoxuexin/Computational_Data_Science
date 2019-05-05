
using Plots, Interact, Distributions, StatsBase, ProgressMeter
gr(
    label="",
    markerstrokecolor="white",
    alpha=0.7,
    markerstrokewidth=0.3,
    color="black"
)

X = Bernoulli(0.5)
plot(x -> cdf(X, x), -1, 2, xlabel="x", ylabel="F(x)", label="cdf of Bernoulli") 

N = 10 # number of samples
x = rand(Bernoulli(0.5), N)
p = plot(ecdf(x), minimum(x) - 1, maximum(x) + 1, xlabel="x", ylabel="F(x)", label="empirical cdf")
plot!( x-> cdf(Bernoulli(0.5),x), -1, 2, color =:red, xlabel="x", ylabel="F(x)", label="theoretical cdf ")

plot(x-> cdf(Normal(0,1),x), -10, 10, xlabel="x", ylabel="F(x)", label="Gaussian cdf ") 

@manipulate for N = [10, 50, 200, 1000, 5000] # number of samples
    x = rand(Normal(0, 1), N)
    
    plot(
        x -> cdf(Normal(0, 1), x), -3, 3, 
        color=:red, 
        xlabel="x", 
        ylabel="F(x)", 
        label="theoretical cdf "
    )
    plot!(
        ecdf(x), -3, 3 , 
        xlabel="x", 
        color=:blue, 
        ylabel="F(x)",
        label="empirical cdf ", 
        legend=:bottomright)
    plot!(title="N = $N")
end

x = rand(Normal(0, 1), 5000)
histogram(x, normalize=:true, line=false, alpha=0.8)
plot!(x-> pdf(Normal(0, 1), x), -3, 3, linewidth = 2.0, 
    color =:red, xlabel = "x", ylabel = "f(x)", label="theoretical pdf ")

q = 0.75
quantile(Normal(0, 1), q)

@show quantile(Normal(0, 1), 0.5)
@show median(randn(1000));

N = 5000
q = rand(N)
gsample = quantile.(Normal(3, 1), q) # note the use of the "." operator after quantile
histogram(gsample,normalize=:true, line=false)
plot!(x-> pdf(Normal(3, 1), x), 0, 8, color="red", linewidth = 2.0, ylabel="density", xlabel="x")

"""
    res = kappa(x, order)

Compute cumulants of orders between 1 and 6

Inputs:
* `x`: vector of data
* `order`: order of cumulant
"""
function kappa(x, order::Int)

    if (order < 1 || order > 6)
        error("Order out of defined bounds")
    end

    xc = x - mean(x) # Centered data
    mu(n) = mean((xc).^n) # use this function below

    if order == 1 # First cumulant
        res = mu(1) # Mean
    elseif order == 2 # Second cumulant
        res = mu(2) # Variance
    elseif order == 3 # Third cumulant
        res = mu(3) # Third central moment
    elseif order == 4 # Fourth cumulant
        res = mu(4) - 3 * mu(2) ^ 2
    elseif order == 5 # Fifth cumulant
        res = mu(5) - 10 * mu(2) * mu(3)
    elseif order == 6 # Sixth cumulant
        res = mu(6) - 15 * mu(4) * mu(2) - 10 * mu(3) ^ 2 + 30 * mu(2) ^ 3
    end

    return res
end

@show kappa(randn(10000), 3);

trials = 4000
kappa4 = zeros(trials)
kappa6 = zeros(trials)
for idx = 1:trials,
    kappa4[idx] = kappa(randn(10000), 4)
    kappa6[idx] = kappa(randn(10000), 6)
    
end

@show mean(kappa6)
@show mean(kappa4)
histogram(kappa6,color=:red, label="kappa6", normalize=:true, alpha=0.5)
histogram!(kappa4, color=:blue, label="kappa4",normalize=:true, alpha=0.5)

include("pymanopt_deps.jl") # loads pymanotpt and the various solvers
using ForwardDiff # this is a Julia package for computing gradients

absk4(y) = abs(kappa(y,4))
function optimq1(A::Matrix, maxiters::Integer=1000, x0=normalize(randn(size(A, 1))); obj=absk4)
    m, n = size(A) 
    
    solver = ConjugateGradient(maxiter=maxiters)
    
    problem = Problem(
         manifold = Sphere(m),
         cost = x -> - obj(x'*A), ## cost function to be minimized. TODO: Match up with math
         egrad = x -> ForwardDiff.gradient(x -> -obj(x'*A),x),
         verbosity = 0 # suppreses solver intermediate output
        )
    xopt = solve(solver, problem) 
    return xopt
end

ϕ = π*45/180 ## this produces axes that are aligned at 45 degrees and 130 degrees
Q = [cos(ϕ) sin(ϕ);
     sin(ϕ) -cos(ϕ)];

X = rand(2, 5000) ## elements of X are independent
Y = Q*X

scatter(Y[1,:], Y[2,:], alpha=0.1, color=:cyan, aspect_ratio=:equal)
quiver!(
    [0, 0], [0, 0],
    quiver=[(Q[1,1], Q[1,2]), (Q[2,1], Q[2,2])], 
    title="Q[:,1] and Q[:,2] in red and black",
    color=["blue", "red" ,"black"]
)

qic = optimq1(Y,obj = x-> abs(kappa(x,4)))

@show qic'*Q;

abs.(mapslices(kurtosis, X, 2))

abs(kurtosis(qic'*Y))

θ_range = linspace(0, 2π, 1000)
κ(A, θ, order::Int) = kappa([cos(θ) sin(θ)]*A, order)
cum_order = 4 
@manipulate for n in[100, 500, 1000, 10000]
    Y = Q*rand(2, n)
    qpc = optimq1(Y, obj = x-> abs(kappa(x, cum_order)))
    plot(θ -> abs(κ(Y,θ,cum_order)), θ_range, proj=:polar, title="n = $n", label="|kappa4|")
    plot!(ylims=(0, ylims()[2]))
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.2, color=:blue)
    plot!(title = "qic'*Q = $(qic'*Q)")
end

θ_range = linspace(0, 2π, 1000)
κ(A, θ, order::Int) = kappa([cos(θ) sin(θ)]*A, order)
cum_order = 3 
@manipulate for n in[100, 500, 1000, 10000]
    Y = Q*rand(2, n)
    qpc = optimq1(Y, obj = x-> abs(kappa(x, cum_order)))
    plot(θ -> abs(κ(Y,θ,cum_order)), θ_range, proj=:polar, title="n = $n", label="|kappa3|")
    plot!(ylims=(0, ylims()[2]))
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.2, color=:blue)
    plot!(title = "qic'*Q = $(qic'*Q)")
end

θ_range = linspace(0, 2π, 1000)
cum_order = 2 # PCA 
@manipulate for n = [100, 500, 1000, 10000]
    Y = Q*rand(2, n) 
    qic = optimq1(Y, obj = x-> kappa(x, cum_order))
    plot(θ -> κ(Y, θ, cum_order), θ_range, proj=:polar, title="n = $n", label="|kappa$(cum_order)|")
    plot!(ylims=(0,ylims()[2]))
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.4, color=:red)
    plot!(title="qic'*Q = $(qic'*Q)")
end

θ_range = linspace(0, 2π, 1000)
cum_order = 6 
@manipulate for n = [100, 500, 1000, 10000]
    Y = Q*rand(2, n) 
    qic = optimq1(Y, obj = x-> abs(kappa(x, cum_order)))
    plot(θ -> abs(κ(Y, θ, cum_order)), θ_range, proj=:polar, title="n = $n", label="|kappa$(cum_order)|")
    plot!(ylims=(0, ylims()[2]))
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.4, color=:blue)
    plot!(title="qic'*Q = $(qic'*Q)")
end

θ_range = linspace(0, 2π, 1000)
cum_order = 4
@manipulate for n in [100, 500, 1000, 10000]
    Y = Q*randn(2, n) 
    qic = optimq1(Y, obj = x-> abs(kappa(x, cum_order)))
    plot(θ -> abs(κ(Y, θ, cum_order)), θ_range, proj=:polar, title="n = $n", label="|kappa$(cum_order)|")
    plot!(ylims=(0,ylims()[2]))
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.4, color =:blue)
    plot!(title = "qic'*Q = $(qic'*Q)")
end

θ_range = linspace(0, 2π, 1000)
cum_order = 3
@manipulate for n in [100, 500, 1000, 10000]
    Y = Q*vcat(rand(1, n),rand(1, n)) 
    qic = optimq1(Y, obj = x -> abs(kappa(x, cum_order)))
    plot(θ ->  abs(κ(Y, θ, cum_order)), θ_range, proj=:polar, title="n = $n", label="|kappa$(cum_order)|")
    plot!(ylims=(0,ylims()[2]))
    ##quiver!([0],[ylims()[1]],quiver=[(ϕ,ylims()[2]-ylims()[1])],proj=:polar)
    y1, y2 = ylims()
    plot!([0, ϕ], [y1, y2 - y1], seriestype=:path, proj=:polar, arrow=0.4, color =:blue)
    plot!(title = "qic'*Q = $(qic'*Q)")
end

## TODO: plot error in estimating ICs using kappa4 versus kappa6 as a function 
## of n averaged over 500 trials each

# n = 100:50:1000
# trail = 500
# len = size(n)[1]
# err4 = zeros(len)
# err6 = zeros(len)

# for i in 1:len
#     error4 = zeros(trail)
#     error6 = zeros(trail)
#     for j in 1:trail
#         Y = Q * rand(2, n[i])
#         ic4 = optimq1(Y, obj = x-> abs(kappa(x, 4)))
#         ic6 = optimq1(Y, obj = x-> abs(kappa(x, 6)))
#         error4[j] = vecnorm(abs.(ic4) - [1,1]/sqrt(2))
#         error6[j] = vecnorm(abs.(ic6) - [1,1]/sqrt(2))
#         err4[i] = mean(error4)
#         err6[i] = mean(error6)
#     end
# end

# plot(n, err4, color =:blue, label = "kappa 4", title = "error of estimation")
# plot!(n, err6, color =:red, label = "kappa 6")

const SphereComplementSubspace = mani.SphereSubspaceComplementIntersection
absk4(y) = abs(kappa(y,4))

function optimQ(A, k, maxiters=1000, x0=normalize(randn(size(A, 1))); obj=absk4)
    m, n = size(A) 
    q1 = optimq1(A, maxiters, x0; obj=obj)
    Q  = reshape(q1, :, 1)
    opt_obj_values = obj(q1'*A)
    for i in 2:min(k, m - 1)
        manifold = SphereComplementSubspace(m, reshape(Q, :,  i-1))
                
        solver = ConjugateGradient(maxiter=maxiters)
        problem = Problem(
        manifold = manifold,
        cost = x-> -obj(x'*A), # cost function to be minimized
        egrad = x -> ForwardDiff.gradient(x -> - obj(x'*A),x), ## gradient of cost function
        verbosity = 0 # suppreses solver intermediate output
        )       
        xopt = solve(solver, problem)
        opt_obj_values = vcat(opt_obj_values, obj(xopt'*A))
        Q = hcat(Q, xopt)
    end
    
    if k == m 
        qm = (I - Q*Q')*randn(m)
        qm = qm/vecnorm(qm)
        opt_obj_values = vcat(opt_obj_values, obj(qm'*A))
        Q = hcat(Q, qm)
    end
    
    ## Sort the components in descending order with respect to the objective function
    obj_sorted = sortperm(opt_obj_values, rev=true)
    Q = Q[:,obj_sorted]
    
    return Q
end

function ica_factorization(Y, k, maxiters=1000; obj = x -> abs(kappa(x,6)))
    m, n = size(Y)
    μy = mean(Y, 2)
    Ymean = μy * ones(1, size(Y, 2))
    Ytil = Y - Ymean
    
    UsV = svds(Ytil; nsv=k)[1]
    U = UsV[:U]
    s = UsV[:S]
    V = UsV[:V]
    S = Diagonal(s)

    Qica = optimQ(sqrt(n)*V',k,maxiters, obj) 
    Vica = V * Qica ## TODO: Fill in formula for Vica'
    Wica = U * S * Qica   ## TODO: Fill in formula (1) for Wica
    Xica = pinv(Wica)*Ymean + Vica'
    return Wica, Xica, Qica
end

A = [1 0.5;0.25 0.75]
U, s, V = svd(A)
X = rand(2,1000)
Y = A*X
Wica, Xica, Qica = ica_factorization(Y, 2)
include("pca_factorization.jl") ## code using SVD
Xpca = pca_factorization(Y,2)[2];

py = scatter(Y[1,:], Y[2,:], title="Y")
pica = scatter(Xica[1,:], Xica[2,:], title=" ICA coordinates")
ppca = scatter(Xpca[1,:], Xpca[2,:], title=" PCA coordinates")
plot(py,pica,ppca,layout=(1, 3))

using Colors, Images, FileIO, Plots
plotly()

image = "hedgehog.jpg";
# image = "panda.jpg";

I1 = Float64.(Gray.(load(image)))
I2 = randn(size(I1))/sqrt(max(size(I1)[1], size(I1)[2]))

p1 = heatmap(I1, yflip=true, color=:grays, aspect_ratio=:equal, title="I1")
p2 = heatmap(I2, yflip=true, color=:grays, aspect_ratio=:equal, title="I2")
plot(p1, p2; size=(800, 300))

m, n = size(I1)
S = hcat(vec(I1), vec(I2))'
I1 = 0.0
I2 = 0.0 ## this clears I1 and I2 from memory
gc()

# A = [0.5 0.5; 0.5 -0.5]
A = randn(2,2) ## non-orthogonal mixing matrix

Y = A*S

mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), m,n)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), m,n)

p3 = heatmap(mixed1, yflip=true, color=:grays, aspect_ratio=:equal, title="Mixed Image 1")
p4 = heatmap(mixed2, yflip=true, color=:grays, aspect_ratio=:equal, title="Mixed Image 2")
plot(p3, p4; size=(800, 300))

mixed1 = 0.0
mixed2 = 0.0
W, Sica = ica_factorization(Y, 2, obj = x -> abs(kappa(x,6)))
Y = 0.0
gc()
 
unmixed1 = reshape(Sica[1,:]/maximum(Sica[1, :]), m,n)
unmixed2 = reshape(Sica[2,:]/maximum(Sica[2, :]), m,n)
Sica = 0.0
gc()
p5 = heatmap(abs.(unmixed1), yflip=true, color=:grays, aspect_ratio=:equal, title="Unmixed Image 1")
p6 = heatmap(abs.(unmixed2), yflip=true, color=:grays, aspect_ratio=:equal, title="Unmixed Image 2")
unmixed1 = 0.0
unmixed2 = 0.0
gc()
plot(p3, p4, p5, p6; layout=4, size=(800, 800))

##TODO: How close to independent are these images in terms of the cumulants
## Compute and plot difference between the cumulants for  order 1, ..,6 
## Comment on whether this is a good measure of independence 
image_hedge = "hedgehog.jpg";
image_panda = "panda.jpg";

I1 = Float64.(Gray.(load(image_panda)))
I2 = Float64.(Gray.(load(image_hedge)))
vec1 = vec(I1)
vec2 = vec(I2)

sum_of_cumu = zeros(6)
cumu_of_sum = zeros(6)
for i in 1:6
    sum_of_cumu[i] = kappa(vec1, i) + kappa(vec2, i)
    cumu_of_sum[i] = kappa(vec1 + vec2, i)
end

@show sum_of_cumu
@show cumu_of_sum

plot(1:6, sum_of_cumu, color =:blue, label = "sum of cumulants")
plot!(1:6, cumu_of_sum, color =:red, label = "cumulants of sum of images")

function extractpatches(image,patchsize,stride=1)
    return Matrix{eltype(image)}[image[i:i+patchsize-1,j:j+patchsize-1]
                    for i in 1:stride:size(image,1)-(patchsize-1),
                        j in 1:stride:size(image,2)-(patchsize-1)]
end


image = "panda.jpg"
img = Float64.(Gray.(load(image)))
Gray.(img)

include("patch_displaying_utils.jl")
patches = extractpatches(img,8)
patchmat = hcat(vec.(patches)...); #Form a `64 x # patches` matrix from $8 \times 8$ patches extracted from img


@show size(patchmat)

viewpatches(rand(patches,(3,18)))

@time Wpca, Xpca = pca_factorization(patchmat)
viewpatches(reshape([normpatch(reshape(Wpca[:,i],(8,8))) for i in 1:64],(8,8)))

batch_size = 1000
npatches = size(patchmat,2) 
rand_idx = randperm(npatches)[1:batch_size]
rand_patchmat = patchmat[:,rand_idx]

@time Wica, Xica = ica_factorization(rand_patchmat,64);
viewpatches(reshape([normpatch(reshape(Wica[:,i],(8,8))) for i in 1:64],(8,8)))


@manipulate for idx = 1:64 # type in different numbers 
    p1 = plot(Xica[idx,:],st=:histogram, normalize=:true, alpha = 0.4, label = "ICA $idx", title = "ICA")
    p2 = plot(Xpca[idx,:],st=:histogram, color=:green, normalize=:true, alpha = 0.4, label = "PCA $idx", title = "PCA")
    plot(p1,p2)
end

## Repeat the same computationa  

image = "hedgehog.jpg"
img = Float64.(Gray.(load(image)))
Gray.(img)

##TODO: Your code to extract patches and make it a patch matrix 

##TODO: Your code to  display PCA factorization output

##TODO: Your code to  display ICA factorization on randomly selected patches

batch_size = 1000
npatches = size(patchmat,2) 
rand_idx = randperm(npatches)[1:batch_size]
rand_patchmat = patchmat[:,rand_idx]
@time Wica, Xica = ica_factorization(??,64);

##TODO: Your code displaying the differences between PCA and ICA coordinates on the hedgehog
