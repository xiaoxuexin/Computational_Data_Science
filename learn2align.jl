
using Plots, Interact, JLD, Images
gr(
    label="",
    markersize=5,
    markerstrokewidth=0.3,
    markerstrokecolor="white"
)
include("deps.jl")

l = ["a","b","c","d"]
Y = [0.25 0.25; 1 0; 1 1; 0  1]
scatter(Y[:,1], Y[:,2], color=:red, markershape=:circle, label="", aspect_ratio=:equal, xlim = (-0.5, 1.5), ylim = (-0.5, 1.5))
annotate!(1.05*Y[:,1], 1.05*Y[:,2], l)


θ = π/4
@show Q = [cos(θ) -sin(θ); sin(θ) cos(θ)]
X = Y*Q

plot(Y[:, 1], Y[:, 2], color=:red, markershape=:square, label="Y", legend=:bottomleft)
annotate!(1.05*Y[:, 1], 1.05*Y[:,2], l)
plot!(X[:,1], X[:,2], color=:blue, markershape=:circle, label="X", xlim = (-0.5, 1.5), ylim = (-1.5, 1.5),aspect_ratio=:equal)
annotate!(1.05*X[:,1], 1.05*X[:,2], l)


α = 0.5
μ = [1;1 ] 
X = α*Y*Q .+ μ' 
plot(Y[:, 1], Y[:, 2], color=:red, markershape=:square, label="Y", legend=:bottomleft)
annotate!(1.05*Y[:, 1], 1.05*Y[:,2], l)
plot!(X[:,1], X[:,2], color=:blue, markershape=:circle, label="X", xlim = (-0.5, 2.5), ylim = (-1.5, 2.5),aspect_ratio=:equal)
annotate!(1.05*X[:,1], 1.05*X[:,2], l)


"""
    Ya = procrustes(X, Y)

Returns Ya = alpha * (Y - muY) * Q + muX, where muX and muY are the m x n
matrices whose rows contain copies of the centroids of X and Y, and alpha
(scalar) and Q (m x m orthogonal matrix) are the solutions to the Procrustes
+ scaling problem

Inputs: `X` and `Y` are m x n matrices

Output: `Ya` is an m x n matrix containing the Procrustes-aligned version
of Y aligned to X and Q the optimal orthogonal matrix

\min_{alpha, Q: Q^T Q = I} \|(X - muX) - alpha * (Y - muY) Q\|_F
"""
function procrustes(X::Matrix, Y::Matrix)

    muX = mean(X, 1)
    muY = mean(Y, 1)
    
    X0 = X .- muX 
    Y0 = Y .- muY 

    # Procrustes rotation
    U, _, V = svd(X0'*Y0, thin=true) ## TODO: Fill in ?? Hint: Use answer from quiz above. 
    Q = V*U' ## TODO: Fill in ?? Use answer from quiz above.

    # Optimal scaling
    alpha = trace(X0'*Y0*Q)/trace(Y0'*Y0)  ## TODO: Fill in ?? Use answer from quiz above.

    # Align data
    Ya = alpha*(Y0*Q) .+ muX
   
    return Ya, Q

end

using MAT
d = matread("misaligned_digit1.mat")
X, Y = d["X"], d["Y"]
@show size(X)
@show size(Y);

ll = ["a","b","c","d","e","f","g","h","i","j"]
plot(X[:, 1], X[:, 2], color=:red, markershape=:circle, label="target", legend=:bottomleft)
annotate!(1.07*X[:, 1], X[:, 2], ll)
plot!(Y[:, 1], Y[:, 2], color=:blue, markershape=:circle,label="misaligned")
annotate!(1.02*Y[:, 1], 1.02*Y[:, 2], ll)

Ya = procrustes(X, Y)[1] ## TODO: Fill in ?? X or X', Y or Y' ?
plot(X[:, 1], X[:, 2], color=:red, markershape=:circle, label="target", legend=:bottomleft)
annotate!(1.07*X[:, 1], X[:, 2], ll)
plot!(Y[:, 1], Y[:, 2], color=:blue, markershape=:circle,label="misaligned")
annotate!(1.07*Y[:, 1], Y[:, 2], ll)
plot!(Ya[:, 1], Ya[:, 2], color=:cyan, markershape=:circle,label="aligned")

d = matread("misaligned_digit2.mat")
X, Y = d["X"], d["Y"]

plot(X[:, 1], X[:, 2], color=:red, markershape=:circle, label="target", legend=:bottomright)
plot!(Y[:, 1], Y[:, 2], color=:blue, markershape=:circle,label="misaligned")
annotate!(1.07*X[:, 1], X[:, 2], ll)

Ya = procrustes(X,Y)[1] ## TODO: Fill in ??
plot(X[:, 1], X[:, 2], color=:red, markershape=:circle, label = "target", legend=:bottomright)
plot!(Y[:, 1], Y[:, 2], color=:blue, markershape=:circle,label = "misaligned")
plot!(Ya[:, 1], Ya[:, 2], color=:cyan, markershape=:circle,label = "aligned")
[plot!(annotations = (1.07*X[i,1],X[i,2],text(ll[i]))) for i = 1 : length(ll)]
plot!()

moleculeData = matread("moleculeData.mat")["moleculeData"];
@show size(moleculeData);

function plot_molecule(moleculeData; kwargs...)
    return plot(
        moleculeData[1,:,1], 
        moleculeData[2,:,1],
        moleculeData[3,:,1];
        markershape=:circle,
        alpha=0.5,
        kwargs...
    )
end

function plot_molecule!(moleculeData; kwargs...)
    return plot!(
        moleculeData[1,:,1], 
        moleculeData[2,:,1],
        moleculeData[3,:,1];
        markershape=:circle,
        alpha=0.5,
        kwargs...
    )
end

@manipulate for moleculeNum = 100
    plot_molecule(moleculeData[:,:,1]; color=:red, label="reference")
    plot_molecule!(moleculeData[:,:,moleculeNum]; color=:blue, label="misaligned")
end    

plot_molecule(moleculeData[:,:,1]; color=:red)
for moleculeNum = 2:size(moleculeData, 3) 
    plot_molecule!(moleculeData[:,:,moleculeNum]; color=:blue, alpha=0.1)
end
plot!()

meanMolecule = moleculeData[:,:,1]
for j in 2:size(moleculeData, 3)
    nextMoleculeAligned = procrustes(meanMolecule',moleculeData[:,:,j]')[1]' ## TODO: Fill in ?? See hint above
    meanMolecule = 0.5*meanMolecule + 0.5*nextMoleculeAligned  
end

plot_molecule(moleculeData[:,:,1]; color=:red, label="molecule 1")
plot_molecule!(meanMolecule; color=:blue, label="meanMolecule")

d = load("law.jld")
im1, im2 = d["law1"], d["law2"]
XY1, XY2 = d["XY1"], d["XY2"];

imshow(im1)

imshow(im2)

function imshow_shared_points(im1, im2, XY1, XY2; numbering::Bool=true)
    p1 = imshow(im1)
    scatter!(XY1[:,1],XY1[:,2]; markersize=10)
    numbering && annotate!(XY1[:, 1], XY1[:, 2], 1:size(XY1, 1))
    p2 = imshow(im2)
    scatter!(XY2[:,1],XY2[:,2]; markersize=10)
    numbering && annotate!(XY2[:, 1], XY2[:, 2], 1:size(XY2, 1))
    plot(p1, p2, layout = (1, 2), size=(950, 300))
end

imshow_shared_points(im1, im2, XY1, XY2)

function projective_transform(XY, XYtil)
#
# Syntax:       H = projective_transform(XY, XYtil)
#               
# Inputs:       XY and XYtil are n x 2 matrices containing (x, y) coordinates
#               for n corresponding points in two coordinate systems
#               
# Outputs:      H is the unique 3 x 3 projective transformation matrix that maps
#               XY to XYtil with H[3, 3] = 1. That is, in the ideal case, the
#               following relationship should hold:
#               
#               tmp = [XY ones(n)] * H
#               XYtil = tmp[:, 1:2] ./ tmp[:, 3]
#
    
   # Construct A matrix
    n = size(XY, 1)
    A = zeros(2n, 9)
     for i in 1:n
        alphai = transpose([XY[i, :]; 1])
        A[2i - 1, :] = [alphai'; zeros(3); -XYtil[i, 1] * alphai'] 
        A[2i, :] = [zeros(3); alphai'; -XYtil[i, 2] * alphai']
    end

    Atil = A[:,1:8]
    btil  = - A[:,9] 
    htil  = pinv(Atil)*btil
    
    h = vcat(htil,1) ## This sets H[3,3] = 1 according to convention
    
    H = reshape(h, 3, 3)

    return H
end

H21 = projective_transform(XY2, XY1) # 2 --> 1


H12 = projective_transform(XY1, XY2) # 1 --> 2

using WebStitcher
im1, im2 = load("law1.jpg"),load("law2.jpg")

I1 = [
    ImageStitcher(im1, eye(3)),
    ImageStitcher(im2, H21)
]
imS1 = stitchImages(I1, order = "natural")

## Stitch images from perspective 2

I2 = [
    ImageStitcher(im2, eye(3)),
    ImageStitcher(im1, H12)
]

# order = "natural": closet (to chosen persepctive) displayed on top
# order = "reverse": farthest displayed on top
imS2 = stitchImages(I2, order = "natural")



# im_name = [""] ## your image -- save it as im_name1.jpg and im_name2.jpg
inpath1 = "im_name1.jpg" # change file extension accordingly
inpath2 = "im_name2.jpg"

## Load images and convert to an Array
im1 = load(inpath1)
im2 = load(inpath2)
nothing



 n = 16 # correspondencs (>= 4)
(XY1_ref, XY2_ref), plot = getcorrespondences(im1, im2, n)
plot

if XY1_ref != nothing 
    XY1, XY2 = XY1_ref[], XY2_ref[]
end

H21 = projective_transform(XY2, XY1) # 2 --> 1
I1 = [
    ImageStitcher(im1, eye(3)),
    ImageStitcher(im2, H21)
]

# order = "natural": closet (to chosen persepctive) displayed on top
# order = "reverse": farthest displayed on top
imS1 = stitchImages(I1, order = "natural") 

H12 = projective_transform(XY1, XY2) # 1 --> 2 
I2 = [
    ImageStitcher(im2, eye(3)),
    ImageStitcher(im1, H12)
]

# order = "natural": closet (to chosen persepctive) displayed on top
# order = "reverse": farthest displayed on top
imS2 = stitchImages(I2, order = "natural")

save("$(im_name)_stitched1.png", imS1)
save("$(im_name)_stitched2.png", imS2)
open("$(im_name)_points$(n).jls", "w") do io
    serialize(io, Dict("XY1" => XY1, "XY2" => XY2))
end

function sym2sided_procrustes(A::Matrix,B::Matrix) 
    
    _, U = eig(A)
    _, V = eig(B)
    
    U .*= sign(U[findmax(abs(U), 1)[2]]) 
    V .*= sign(V[findmax(abs(V), 1)[2]]) 

    Qopt = ??*?? ## TODO: Fill in ??

    return Qopt
end
