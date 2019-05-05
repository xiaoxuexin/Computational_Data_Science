
using Plots, Interact, ProgressMeter
gr(
    label="",
    markerstrokewidth=0
)
include("deps.jl") ## temporary fix for GR bug

using PyCall # this allows us to import Python packages 
@pyimport pymanopt #import pymanopt
@pyimport pymanopt.manifolds as mani 
@pyimport pymanopt.solvers as solvers
@pyimport autograd.numpy as np
const Sphere = mani.Sphere
const NelderMead = solvers.NelderMead

const Problem = pymanopt.Problem
solve(solver, problem) = solver[:solve](problem)

function xthatMaximizesAxonSphere(
        A::Matrix,
        maxiters::Integer=1000
    )

    m, n = size(A) 
    
    solver = NelderMead(maxiter=maxiters)
    manifold = Sphere(n); ## spherical constraint -- x is n dimensional vector
    
    problem = Problem(
         manifold = manifold,
         cost = x -> -sum(abs2, A*x), ## cost function to be minimized. TODO: Match up with math
         verbosity = 0 # suppreses solver intermediate output
        )
    
    xopt = solve(solver, problem) 
    return xopt
end

A = randn(3, 3)
xopt = xthatMaximizesAxonSphere(A, 10)
U, s, V = svd(A)
u₁, v₁ = U[:,1], V[:,1] ## TODO: Match up with theory
@show xopt
@show v₁
@show v₁'xopt

maxiters = [10, 20, 40, 80, 160, 240, 320, 500, 1000, 1250, 1500]
innerproduct = zeros(length(maxiters))
@showprogress for idx in 1:length(maxiters)
   xopt = xthatMaximizesAxonSphere(A, maxiters[idx]); 
   innerproduct[idx] = (xopt'*v₁);
end

plot(
    maxiters,
    innerproduct,
    yaxis=(-1,1),
    xlabel="maxiters",
    ylabel="inner product",
    m=:circle
)

plot(
    maxiters,
    abs.(innerproduct),
    yaxis=(0,1), 
    label="abs. inner product", 
    xlabel="maxiters",
    m=:square, 
    color=:"red", 
)

xopt = xthatMaximizesAxonSphere(A') 
@show xopt'u₁
xopt_alt = xthatMaximizesAxonSphere(-A') 
@show xopt_alt'u₁

const SphereComplementSubspace = mani.SphereSubspaceComplementIntersection

function manoptV(A::Matrix, k::Integer, maxiters::Integer=1000)
 
    m, n = size(A) 
    xopt = xthatMaximizesAxonSphere(A, maxiters)
    V = xopt
    
    for i in 2:min(k, n-1)
        manifold = SphereComplementSubspace(n, reshape(V, :,  i-1))
                
        solver = NelderMead(maxiter=maxiters)
        problem = Problem(
        manifold = manifold,
        cost = x-> -sum(abs2,A*x), # cost function to be minimized
        verbosity = 0 # suppreses solver intermediate output
        )       
        xopt = solve(solver, problem)
        V = hcat(V, xopt)
    end
    
    if k == n 
        vn = (I - V*V')*randn(n)
        vn = vn/vecnorm(vn)
        V = hcat(V, vn)
    end
    return V
end

k = 3 ## extract k singular vectors
Vmanopt = manoptV(A, k)
Usvd, ssvd, Vsvd = svd(A)
@show Vmanopt'*Vsvd

# flipdim is temporary workaround to GR bug 
heatmap(flipdim(Vmanopt'*Vsvd,1), ticks=[1,2,3])

#flipdim is temporary workaround to GR bug
heatmap(flipdim(abs.(Vmanopt'*Vsvd), 1), ticks=[1,2,3])

Umanopt = manoptV(A', k)

# flipdim is temporary workaround to GR bug
heatmap(flipdim(abs.(Usvd'*Umanopt),1), ticks=[1,2,3])

function manoptPCs(A::Matrix, k::Integer, maxiters::Integer=1000)
     m, n = size(A)
     centeredA = A .- mean(A, 2) ##TODO: Fill in ??
    return manoptV(centeredA', k, maxiters) ##TODO: Fill in ??
end

n = 2500 # number of samples
CovarianceA = Diagonal([1, 2])

# Columns are Normal(0, CovarianceA) distributed:
A = sqrtm(CovarianceA)*randn(2, n)
@show A*A'/n # approximates CovarianceA

"""
    p = scatter_quiver(A, U)

Given a 2-by-n matrix A, plot the second row coordinates against
the first row coordinates. Then add quiver arrows for the
coordinates encoded in U.
"""
function scatter_quiver(A::Matrix, Upc::Matrix; title="PC1 (red) & PC2 (black)", alpha = 0.2)
    p = scatter(
        A[1,:], A[2,:],
        alpha=alpha, 
        aspect_ratio=:equal,
    )
    quiver!(
        p, 
        [0, 0], [0, 0], 
        title=title, 
        quivers=[(Upc[1,1], Upc[1,2]), (Upc[2,1], Upc[2,2])], color=[:black, :red , :black]
    )
    return p
end

Upc = manoptPCs(A, 2)
scatter_quiver(A, Upc; alpha=0.15)

# centered A:
A = sqrtm(CovarianceA)*(rand(2,n) - 0.5*ones(2,n))*sqrt(12)
Upc = manoptPCs(A,2)
scatter_quiver(A, Upc)

"""
    Wpca, Xpca = pca_factorization(A)

* Input:  Data matrix A
* Output: Factor matrices Wpca and Xpca such that A = Wpca*Xpca
"""
function pca_factorization(A::Matrix)
    a_mean = mean(A, 2) 
    A_mean = a_mean * ones(1, size(A, 2))
    A_dm = A - A_mean
    
    U, s, V = svd(A_dm, thin = true)
    
    Wpca = U*Diagonal(s) ##TODO: Fill in ??
    Winv = inv(Wpca) 
    Xpca = Winv * a_mean *ones(1,size(A,2)) + V' ##TODO: Fill in ??
    
    return Wpca, Xpca
end


Y = randn(2, 1000)
Wpca, Xpca = pca_factorization(Y)

# is it small? if not, it is NOT a factorization:
vecnorm(Y - Wpca*Xpca)

n = 1000
A = randn(2,1000)
Upc = manoptPCs(A, 2)
scatter_quiver(A, Upc; title="PCs: Independent realization 1")

n = 1000
A = randn(2,1000)
Upc = manoptPCs(A,2)
scatter_quiver(A, Upc; title="PCs: Trial 2")

n = 1000
A = (rand(2,n) - 0.5*ones(2,n))*sqrt(12)
Upc = manoptPCs(A,2)
scatter_quiver(A, Upc; title="PCs: Trial 1")

n = 1000
A = (rand(2,n) - 0.5*ones(2,n))*sqrt(12)
Upc = manoptPCs(A,2)
scatter_quiver(A, Upc; title="PCs: Trial 2")

κ₄(x,A) = mean(x -> x^4, x'*A) - 3*mean(x -> x^2, x'*A)^2

function xthatMaximizesAxkurtonSphere(
        A::Matrix,
        maxiters::Integer=10000
    )
  
    m,n = size(A) 
    solver = NelderMead(maxiter = maxiters)
    
    manifold = Sphere(m) ## spherical constraint
    problem = Problem(
        manifold = manifold,
        cost = x -> -abs(κ₄(x, A)), ##TODO: Fill in ??
        verbosity = 0 # suppreses solver intermediate output
        )
    xopt = solve(solver, problem) 
    return xopt
end


function manoptICs(A::Matrix, k::Integer, maxiters::Integer=1000)
 
    m, n = size(A) 
    xopt = xthatMaximizesAxkurtonSphere(A,maxiters)
    U = xopt
    for i = 2 : min(k, m-1)
        
        manifold = SphereComplementSubspace(m, reshape(U, :, i-1))
                
        solver = NelderMead(maxiter=  maxiters)
        problem = Problem(
        manifold = manifold,
        cost = x-> -abs(κ₄(x, A)), ## TODO: Fill in ??
        verbosity = 0 # suppreses solver intermediate output
        )       
        xopt = solve(solver, problem)
        U = hcat(U, xopt)
    end
    
    if k == m 
        um = (I - U*U')*randn(m)
        um = um/vecnorm(um)
        U = hcat(U, um)
    end
    return U
end

@show Uic = manoptICs(A,2)
scatter_quiver(A, Uic; title="ICs")

n = 10000 # number of samples
Q = qr(randn(2, 10))[1]
println("Q = $Q")

# Columns are Normal(0,CovarianceA) distributed
A = Q*sqrt(12)*(rand(2, n) - 0.5)

Upc = manoptPCs(A, 2)
println("Upca = $Upc")

Uic= manoptICs(A, 2)
println("Uica = $Uic")

plot(
    scatter_quiver(A[:,1:10:end], Upc; title="PCA"),
    scatter_quiver(A[:,1:10:end], Uic; title="ICA")
)

function ica_factorization(Y)
    μy = mean(Y, 2)

    Ymean = μy * ones(1, size(Y, 2))

    Ytil = Y - Ymean

    U, s, V = svd(Ytil, thin=true)
    S = Diagonal(s)
    ## Fact: If Ytil*1  = 0 then V'*1 = 0
    ## => Ytil = U*S*V'

    Qica = manoptICs(V',size(Y,1))
    Vica = (Qica'*V')' ## TODO: Match this with equations above 
    W = U*S*Qica
    Xica = inv(W)*Ymean + Vica' ## TODO: Match this with equations
    return W, Xica
end

W, Xica = ica_factorization(Y)

# is it small? if not, it is NOT a factorization
vecnorm(Y - W*Xica)

using Colors, Images, FileIO, Plots
gr(
    label="",
    markerstrokewidth=0
)

image1 = "images/images/skyline1.jpeg"; image2 = "images/images/skyline2.jpeg"; ## buildings
#image1 = "images/images/bunny.jpg"; image2 = "images/images/pig.jpg";
I1 = Float64.(Gray.(load(image1)))
sizeI1 = size(I1)
I2 = Float64.(Gray.(load(image2)))
#I2 = randn(sizeI1)

p1 = imshow(I1, color=:grays, aspect_ratio=:equal, title="I1")
p2 = imshow(I2, color=:grays, aspect_ratio=:equal, title="I2")
plot(p1, p2, size=(900, 250))

# run this cell only once
s1 = vec(I1) 
s2 = vec(I2) 
I1 = 0.0
I2 = 0.0 ## this clears I1 and I2 from memory
gc()

A = [0.5 0.5; 0.5 -0.5];
S = [s1 s2]';
Y = A*S;
S = 0.0; # clear S variable
mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), sizeI1)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), sizeI1)

p3 = imshow(
    mixed1, 
    color=:grays, 
    aspect_ratio=:equal, 
    title="mixed image 1"
)
p4 = imshow(
    mixed2, 
    color=:grays, 
    aspect_ratio=:equal, 
    title="mixed image 2 "
)
plot(p3, p4, size=(900, 250))

mixed1 = 0.0
mixed2 = 0.0 # clear mixed1 and mixed2 variables
gc()

Wica, Sica = ica_factorization(Y)
Sica = abs.(Sica) ## make sign positive since it's an image

Y = 0.0; # clear Y variable
gc() 

unmixed1 = reshape(Sica[1,:]/maximum(Sica[1, :]), sizeI1)
unmixed2 = reshape(Sica[2,:]/maximum(Sica[2, :]), sizeI1)
Sica = 0.0; # clear Sica variable
gc()
p5 = imshow(unmixed1, color=:grays, aspect_ratio=:equal, title="ica Unmixed Image 1")
p6 = imshow(unmixed2, color=:grays, aspect_ratio=:equal, title="ica Unmixed Image 2")
unmixed1 = 0.0
unmixed2 = 0.0
gc()
plot(p3, p4, p5, p6, layout=4, size=(900, 600))

plot(p5)

##TODO: Unmix using PCA 
mixed1 = 0.0
mixed2 = 0.0 # clear mixed1 and mixed2 variables
gc()

Wpca, Xpca = pca_factorization(Y)
Xpca = abs.(Xpca) ## make sign positive since it's an image

Y = 0.0; # clear Y variable
gc() 

unmixed1 = reshape(Xpca[1,:]/maximum(Xpca[1, :]), sizeI1)
unmixed2 = reshape(Xpca[2,:]/maximum(Xpca[2, :]), sizeI1)
Xpca = 0.0; # clear Sica variable
gc()
p5 = imshow(unmixed1, color=:grays, aspect_ratio=:equal, title="pca Unmixed Image 1")
p6 = imshow(unmixed2, color=:grays, aspect_ratio=:equal, title="pca Unmixed Image 2")
unmixed1 = 0.0
unmixed2 = 0.0
gc()
plot(p3, p4, p5, p6, layout=4, size=(900, 600))


scatter(rand(1000), randn(1000), alpha=0.3, xlabel="x", ylabel="y")

x = randn(1000)
y = randn(1000)
scatter(x, x + y, alpha=0.3, xlabel="x", ylabel="y")

subset = randperm(length(s1))[1:1000]
scatter(
    s1[subset], s2[subset], 
    alpha=0.3,
    xlabel="Image 1 pixel values",
    ylabel="Image 2 pixel values"
)

using FileIO, WAV

# create mixed sources & save as .wav file
source_num = [2 5]; # pick two -- from 1, 2, 3, 4 ,5
# try different sources here!
source1 = "audio/audio/source$(source_num[1]).wav"
source2 = "audio/audio/source$(source_num[2]).wav"

s1, fs = load(source1)
s2, fs2 = load(source2)
min_s1s2_size = min(length(s1), length(s2))
# strip to same length
s1 = s1[1:min_s1s2_size]
s2 = s2[1:min_s1s2_size]
S = [s1 s2]'

A = [2 1.5; 1.5 2]; # mixing matrix
# Try different A matrices!

Y = A*S # mix the signals together
wavwrite(Y[1, :], "mixed1.wav", Fs=fs)
wavwrite(Y[2, :], "mixed2.wav", Fs=fs)
## open and play the .wav file in the directory

Sica = ica_factorization(Y)[2]

invSica = (1.0 ./ maximum(abs.(Sica), 2))[:, 1] ## normalize to same peak amplitude
Sica = diagm(invSica) * Sica

wavwrite(Sica[1, :], "unmixed1.wav", Fs=fs)
wavwrite(Sica[2, :], "unmixed2.wav", Fs=fs)
## play the saved .wav files in the directory

s1, s2, S  = 0, 0 , 0 
gc()

##TODO: Unmix using PCA -- your code here

Spca = pca_factorization(Y)[2]

invSpca = (1.0 ./ maximum(abs(Spca), 2))[:, 1] ## normalize to same peak amplitude
Spca = diagm(invSpca) * Spca 
wavwrite(Spca[1, :], "pca_unmixed1.wav", Fs=fs)
wavwrite(Spca[2, :], "pca_unmixed2.wav", Fs=fs)
Y  = 0
gc()

sawtooth(t, width = 0.5) = mod(t, width)
square(t, width = 0.5)  = sign(sin(t/width))

t=-5:0.01:5
A = [1 0.5; 1 1]
@manipulate for s1type ∈ ["sin","square","sawtooth"], s2type ∈ ["cos","square","sawtooth"]
    if s1type == "sin"
        s1 = sin.(2*t)
    elseif s1type == "square"
        s1 = square.(t,1.25)
    elseif s1type == "sawtooth"
        s1 = sawtooth.(t,2.3)
    end
    
    if s2type == "cos"
        s2 = cos.(3*t)
    elseif s2type == "square"
        s2 = square.(t)
    elseif s2type == "sawtooth"
        s2 = sawtooth.(t)
    end
    
    S = [s1 s2]'
    Y = A*S;
    Sica =  ica_factorization(Y)[2]
    
    p1 = plot(t,S[1,:], color=:red, title="Signal 1")
    p2 = plot(t,S[2,:], color=:blue, title="Signal 2")

    p3 = plot(t,Y[1,:], color=:red, title="Mixed Signal 1")
    p4 = plot(t,Y[2,:], color=:blue, title="Mixed Signal 2")

    
    p5 = plot(t,Sica[1,:], color=:red, title="Unmixed Signal 1")
    p6 = plot(t,Sica[2,:], color=:blue, title="Unmixed Signal 2")

    plot(p1,p2,p3,p4,p5,p6,layout=(3,2))
end

t=-5:0.01:5
s1 = sin.(2*t)
s2 = square.(t)
# subset1 = randperm(length(s1))
# subset2 = randperm(length(s2))
scatter(
    s1, s2, 
    alpha=0.3,
    xlabel="signal 1 values",
    ylabel="signal 2 values"
)

## TODO code here to unmix with PCA
t=-5:0.01:5
A = [1 0.5; 1 1]
@manipulate for s1type ∈ ["sin","square","sawtooth"], s2type ∈ ["cos","square","sawtooth"]
    if s1type == "sin"
        s1 = sin.(2*t)
    elseif s1type == "square"
        s1 = square.(t,1.25)
    elseif s1type == "sawtooth"
        s1 = sawtooth.(t,2.3)
    end
    
    if s2type == "cos"
        s2 = cos.(3*t)
    elseif s2type == "square"
        s2 = square.(t)
    elseif s2type == "sawtooth"
        s2 = sawtooth.(t)
    end
    
    S = [s1 s2]'
    Y = A*S;
    Sica =  pca_factorization(Y)[2]
    
    p1 = plot(t,S[1,:], color=:red, title="Signal 1")
    p2 = plot(t,S[2,:], color=:blue, title="Signal 2")

    p3 = plot(t,Y[1,:], color=:red, title="Mixed Signal 1")
    p4 = plot(t,Y[2,:], color=:blue, title="Mixed Signal 2")

    
    p5 = plot(t,Sica[1,:], color=:red, title="Unmixed Signal 1")
    p6 = plot(t,Sica[2,:], color=:blue, title="Unmixed Signal 2")

    plot(p1,p2,p3,p4,p5,p6,layout=(3,2))
end

image1 = "images/images/3.jpg"; image2 = "images/images/2.jpg"; 

# Load images
I1 = Float64.(Gray.(load(image1)))
I2 = Float64.(Gray.(load(image2)))

sizeI1 = size(I1)
S = [vec(I1) vec(I2)]';

I1, I2 = 0, 0
gc()
# Mix images
A = [0.5 0.5; 0.5 -0.5];
Y = A*S

gc()

mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), sizeI1)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), sizeI1)

## TODO: attempt to unmix
Sica = ica_factorization(Y)[2]
unmixed1 = reshape(Sica[1,:],sizeI1)
unmixed2 = reshape(Sica[2,:],sizeI1)
Sica , Y = 0, 0 
gc()


p1 = imshow(mixed1,color=:grays, aspect_ratio=:equal, title = "Mixed Image 1 ")
p2 = imshow(mixed2,color=:grays,  aspect_ratio=:equal, title = "Mixed Image 2")
p3 = imshow(unmixed1,color=:grays,  aspect_ratio=:equal, title = "ica Unmixed Image 1")
p4 = imshow(unmixed2,color=:grays,  aspect_ratio=:equal, title = "ica Unmixed Image 2")

plot(p1,p2,p3,p4,layout=4)

I1 = Float64.(Gray.(load(image1)))
I2 = Float64.(Gray.(load(image2)))
s1 = vec(I1) 
s2 = vec(I2) 
I1 = 0.0
I2 = 0.0 ## this clears I1 and I2 from memory
gc()
subset = randperm(length(s1))[1:2000]
scatter(
    s1[subset], s2[subset], 
    alpha=0.3,
    xlabel="Image 1 pixel values",
    ylabel="Image 2 pixel values"
)

##TODO: Code for displaying approximate independence of images where ICA successfully unmixes images  

image1 = "images/images/Unknown-4.jpeg"; image2 = "images/images/Unknown-5.jpeg"; 

# Load images
I1 = Float64.(Gray.(load(image1)))
I2 = Float64.(Gray.(load(image2)))[1:size(I1)[1], 1:size(I1)[2]]

sizeI1 = size(I1)
S = [vec(I1) vec(I2)]';

I1, I2 = 0, 0
gc()
# Mix images
A = [0.5 0.5; 0.5 -0.5];
Y = A*S

gc()

mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), sizeI1)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), sizeI1)

## TODO: attempt to unmix
Sica = ica_factorization(Y)[2]
unmixed1 = reshape(Sica[1,:],sizeI1)
unmixed2 = reshape(Sica[2,:],sizeI1)
Sica , Y = 0, 0 
gc()


p1 = imshow(mixed1,color=:grays, aspect_ratio=:equal, title = "Mixed Image 1 ")
p2 = imshow(mixed2,color=:grays,  aspect_ratio=:equal, title = "Mixed Image 2")
p3 = imshow(unmixed1,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 1")
p4 = imshow(unmixed2,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 2")

plot(p1,p2,p3,p4,layout=4)

##TODO: Code for displaying approximate independence of images where ICA succesfully unmixes signals

## Load and mix the images here 
S = ?? 
subset = randperm(length(S[1,:]))[1:1000]  ## select a random subset of the pixels to make plot less cluttered 
scatter(S[1,subset],S[2,subset], 
    label="", 
    color=:green, 
    alpha = 0.1, 
    xlabel ="Image 1 pixel values", 
    ylabel = "Image 2 pixel values")

##TODO: Code displaying approximate independence of images where ICA succesfully unmixes audio signals

##TODO: Generate two signal waveforms (they cannot be the same waveform :-)  -- mix them and show that ICA fails to recover them.
##      Use the same visualization as before to illustrate visually the dependence structure 



image1 = "images/images/face1.jpg"; image2 = "images/images/face2.jpg"; ## what are the original faces?

# Load images
I1 = Float64.(Gray.(load(image1)))
I2 = Float64.(Gray.(load(image2)))

sizeI1 = size(I1)
S = [vec(I1) vec(I2)]';

I1, I2 = 0, 0
gc()
# Mix images
A = [0.5 0.5; 0.5 -0.5];
Y = A*S

gc()

mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), sizeI1)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), sizeI1)

## TODO: attempt to unmix
Sica = ica_factorization(Y)[2]
unmixed1 = reshape(Sica[1,:],sizeI1)
unmixed2 = reshape(Sica[2,:],sizeI1)
Sica , Y = 0, 0 
gc()


p1 = imshow(mixed1,color=:grays, aspect_ratio=:equal, title = "Mixed Image 1 ")
p2 = imshow(mixed2,color=:grays,  aspect_ratio=:equal, title = "Mixed Image 2")
p3 = imshow(unmixed1,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 1")
p4 = imshow(unmixed2,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 2")

plot(p1,p2,p3,p4,layout=4)

subset = randperm(length(S[1,:]))[1:1000]
scatter(
    S[1,subset], S[2,subset], 
    alpha = 0.15, 
    xlabel="Image 1 pixel values", 
    ylabel="Image 2 pixel values"
)

function ica_factorization1(W,Y)
    μy = mean(Y, 2)

    Ymean = μy * ones(1, size(Y, 2))

    Ytil = Y - Ymean

    U, s, V = svd(Ytil, thin=true)
    S = Diagonal(s)
    ## Fact: If Ytil*1  = 0 then V'*1 = 0
    ## => Ytil = U*S*V'

    Qica = manoptICs(V',size(Y,1))
    Vica = (Qica'*V')' ## TODO: Match this with equations above 
    Xica = inv(W)*Ymean + Vica' ## TODO: Match this with equations
    return Xica
end

## Hard  Problem: Code to unmix images above -- 
## Hint: can we relax the notion of independent "everywere" to independent "somewhere" and suceeed?  
## Hint: You will have to find the sub-region that is nearly indpendent, determine the unmixing matrix there 
##       and apply it to the rest of the image. 
image1 = "images/images/face1.jpg"; image2 = "images/images/face2.jpg";
I1 = Float64.(Gray.(load(image1)))
I2 = Float64.(Gray.(load(image2)))

sizeI1 = size(I1)
sizeI2 = size(I2)
S = [vec(I1) vec(I2)]';

S1 = S[1,:][(S[1,:].>0.5) .& ((S[1,:].<0.7)) .& (S[2,:].>0.3) .& (S[2,:].<0.7)]'
S2 = S[2,:][(S[1,:].>0.5) .& ((S[1,:].<0.7)) .& (S[2,:].>0.3) .& (S[2,:].<0.7)]'
subS = vcat(S1,S2)

gc()
# Mix images
A = [0.5 0.5; 0.5 -0.5];
Y1 = A*subS

gc()
## TODO: attempt to unmix
W = ica_factorization(Y1)[1]

# Mix images
Y = A*S
Sica = ica_factorization1(W,Y)

mixed1 = reshape(Y[1, :]/maximum(Y[1, :]), sizeI1)
mixed2 = reshape(Y[2, :]/maximum(Y[2, :]), sizeI1)
unmixed1 = reshape(Sica[1,:],sizeI1)
unmixed2 = reshape(Sica[2,:],sizeI1)
Sica , Y = 0, 0 
gc()

p1 = imshow(mixed1,color=:grays, aspect_ratio=:equal, title = "Mixed Image 1 ")
p2 = imshow(mixed2,color=:grays,  aspect_ratio=:equal, title = "Mixed Image 2")
p3 = imshow(unmixed1,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 1")
p4 = imshow(unmixed2,color=:grays,  aspect_ratio=:equal, title = "Unmixed Image 2")

plot(p1,p2,p3,p4,layout=4)
