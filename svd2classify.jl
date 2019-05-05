
using Images, Plots, Interact
gr(
    label="",
    markerstrokecolor="white",
    alpha=0.7,
    markerstrokewidth=0.3,
    color="black"
)
include("deps.jl") ## this file contains some useful plotting functions -- you can examine it to learn more
println("Ready!")

A = randn(10, 100)
U = svd(A, thin=true)[1]
b = randn(10)
PrangeA = U*U' 
error = vecnorm(b - PrangeA*b)

"""
    U = learn_nearest_ss(train,ktrain)

Learn a set of orthonormal bases for the nearest subspaces.

## Inputs
* `train` is an n x m x d array containing m training samples (of dimension n) for each class
* `ktrain` in [1, min(n, m)] is the number of singular vectors to save
 
## Output
* `U` is an n x ktrain x d array containing a set of `ktrain` (orthonormal) basis vectors (of dimension n) for each of d classes
"""
function learn_nearest_ss(train::Array, ktrain::Integer)
    n, N, d = size(train)

    # Compute and save basis vectors
    U = zeros(n, ktrain, d)
    for j = 1:d
          Uj = svd(train[:,:,j], thin=true)[1]  # TODO: fill in ??
          U[:, :, j] =  Uj[:,1:ktrain,:]     # TODO: fill in ??
    end
    return U
end

"""
    labels = classify_nearest_ss(test, U, k=size(U,2))

Classifies the input test vectors by which subspace is nearest to each.

## Inputs
* `test` is an n x t matrix whose columns are vectors to be classified
* `U` is an n x ktrain x d array containing a set of ktrain (orthonormal) basis vectors (of dimension n) for each of d classes
* `k` in [1, ktrain] is the number of basis vectors to use during classification (default: ktrain)
           
## Outputs
* `labels` is a vector of length t containing which subspace each test vector is closest to
"""
function classify_nearest_ss(test::Matrix, U::Array, k::Integer=size(U,2))
    n, t = size(test)
    d = size(U, 3)
    
    # Construct projection matrices
    P = zeros(n, n, d)
    for j = 1:d
          P[:, :, j] =  U[:, 1:k, j]*U[:, 1:k, j]'  # TODO: Fill in ??. Hint: How is U organized?
    end

    # Calculate projection errors
    err = zeros(d, t)
    for j = 1:d
        err[j, :] = sum((test - P[:,:,j]*test).^2, 1)  # TODO: Fill in ?? Hint: Use formula you derived.
    end

    # Classify each vector by which subspace is nearest
    labels = vec(mapslices(indmin, err, 1))   # TODO: Fill in ?? Hint: Which dimension are we computing error over?
    
    return labels
end

classify_nearest_ss(
    test::Vector,
    U::Array,
    k::Integer=size(U,2)
    ) = classify_nearest_ss(reshape(test, (length(test), 1)), U, k)[1]

# Load training data
using MAT
vars = matread("train_digits.mat")
train = matopen("train_digits.mat") do file
    read(file,"train_data")
end

vars = matread("test_digits.mat")
file = matopen("test_digits.mat")
test = read(file,"test_data")
test_label = read(file,"test_label")
close(file)


m, n, _ = size(train)
T = length(test_label)
p = Int64(sqrt(m))

@show size(train)
@show T

@manipulate for digit in [0:9...]
    d = 1 + digit
    X = reshape(train[:, rand(1:750, 70), d], p, p, 70)
    imshow(
        tileSlices(X),
        title="Digit = $digit",
    )
end

# Adjust the sliders to see each digit

digits = collect(0:9)
subspace_label = digits + 1 # digit "0" corresponds to first subspace and so on.

digit2subspace = Dict(string(digits[i]) => subspace_label[i] for i in 1:length(digits) )
subspace2digit = Dict(value => key for (key, value) in digit2subspace)

# Generate a nice range of ranks to try
num_k = 5
test_image = test[:, rand(1:size(test, 2))]
kmax = 784
krange = unique(Array{Int64,1}(round.(logspace(0, log10(kmax), num_k))))

# Perform classification
trainU = learn_nearest_ss(train, maximum(krange))
@manipulate for k in krange
    # Classify image
    subspace_label = classify_nearest_ss(test_image, trainU, k)
    label = subspace2digit[subspace_label]
    # Plot results
    imshow(
        reshape(test_image, 28, 28),
        title = "k = $k: classified as: $label",
    )
end

# Try different values of k to see what happens 

function pcorrect(predicted_label, test_label) 
    return sum(predicted_label .== test_label)/length(test_label)
end

@time trainU = learn_nearest_ss(train, 784);

predicted_labels = classify_nearest_ss(test, trainU, 1) - 1
@show pcorrect(predicted_labels,test_label)

predicted_labels = classify_nearest_ss(test, trainU, 2) - 1
@show pcorrect(predicted_labels, test_label)

predicted_labels = classify_nearest_ss(test, trainU, 10) - 1
@show pcorrect(predicted_labels, test_label)

klist = [1, 2, 3, 5, 7, 9, 11, 15]

trainU = learn_nearest_ss(train, maximum(klist))
pcorrect_svd = zeros(length(klist))
@time for idx in 1:length(klist)
    pcorr = pcorrect(classify_nearest_ss(test, trainU, klist[idx]) - 1, test_label)
    pcorrect_svd[idx] = pcorr
    IJulia.clear_output(true)
    display(scatter(
            klist[1:idx],
            pcorrect_svd[1:idx],
            ylims=(0, 1),
            xlabel="subspace rank",
            ylabel="Pcorrect", 
            label="error"
            ))
end

function classify_nearest_ss_alt(test::Matrix,U,k=size(U,2))
    n, p = size(test)
    d = size(U,3)
    
    # Calculate projection errors
    err = zeros(d, p)
    for j = 1 : d
        Uj = U[:, 1 : k, j] #TODO fill in ??
        err[j, :] = sum((test - Uj * (Uj'*test)).^2, 1)   
    end

    # Classify each vector by which subspace is nearest
    labels = vec(mapslices(indmin,err,1)) 
    
    return labels
end

## Redoing the experiment from the previous cell we get
pcorrect_svd = zeros(length(klist))
@time for idx in 1:length(klist)
    pcorr = pcorrect(classify_nearest_ss_alt(test, trainU, klist[idx]) - 1, test_label)
    pcorrect_svd[idx] = pcorr
    IJulia.clear_output(true)
    display(scatter(
            klist[1:idx],
            pcorrect_svd[1:idx], 
            ylims=(0,1), 
            xlabel="subspace rank", 
            ylabel="Pcorrect", 
            label="error"
            ))
end

Udem = svd(randn(784, 1000),thin=true)[1]
@time (Udem[:,1:100]*Udem[:,1:100]')*randn(784);
@time Udem[:,1:100]*(Udem[:,1:100]'*randn(784));

using BenchmarkTools
@benchmark (Udem[:,1:100]*Udem[:,1:100]')*randn(784)

@benchmark Udem[:,1:100]*(Udem[:,1:100]'*randn(784))

## Let us now try for a larger range of values for k
klist = [
    1, 2, 3, 5, 7, 9, 11, 15, 20, 
    25, 30, 35, 40, 50, 75, 100, 
    150, 200, 300, 400, 500, 600, 
    700, 750, 784
]
##TODO: for all values of k from 1 to 784 -- for HW! This will take too long in class.
trainU = learn_nearest_ss(train, maximum(klist))
pcorrect_svd = zeros(length(klist))
@time for idx in 1:length(klist)
    pcorr = pcorrect(classify_nearest_ss_alt(test, trainU, klist[idx]) - 1,test_label)
    pcorrect_svd[idx] = pcorr
    IJulia.clear_output(true)
    display(scatter(
            klist[1:idx],
            pcorrect_svd[1:idx], 
            ylims=(0,1),
            xlabel="subspace rank", 
            ylabel="Pcorrect",
            label="error"
            ))
end

@show maximum(pcorrect_svd)
@show klist[indmax(pcorrect_svd)]

# Nearest-mean-digit classification
trainUmean = learn_nearest_ss(mean(train, 2), 1)
mean_pred_labels = classify_nearest_ss_alt(test, trainUmean, 1) -1 
mean_acc = pcorrect(mean_pred_labels, test_label)

# SVD classifier results
plot(
    klist, 100*pcorrect_svd,
    marker=:circle,
    linewidth=1,
    color=:blue,
    label="SVD classifier",
    xlabel="Rank (k)",
    ylabel="Accuracy (%)",
    xscale=:identity, # :log10, :identity
    ylims=(0, 100),
    title="Optimal k = $(klist[indmax(pcorrect_svd)]) with $(100*maximum(pcorrect_svd)) % accuracy"
)

# Plot mean-digit-based results
plot!(
    [1, m], 100*[mean_acc, mean_acc],
    linestyle=:dash,
    color=:green,
    label="Mean-digit",
)

scatter(svdvals(reshape(train,size(train,1),:)), xlabel = "index" , ylabel = "singular value", label = "")

##TODO: 
k_bulk_separate = 10 ## pick closest value from klist or re-run above simulation for a finer grid of values of k
Pcorrect_bulk_separate ## 
pcorrect(classify_nearest_ss_alt(test, trainU, 10) - 1,test_label)

@manipulate for digit in [0, 1, 2, 3, 4, 6, 7, 8, 9]
    error = mapslices(vecnorm, (eye(784) - trainU[:,:,1]*trainU[:,:,digit + 1]')*test, 1)[:]
    scatter(error, xlabel="test digit index", ylabel="Lenght of error vector")
end

trainUrandom = learn_nearest_ss(randn(784,1000,10), 784)
  
@manipulate for k in [1, 2, 10, 783, 784]
    which_digit = classify_nearest_ss_alt(randn(784, 800), trainUrandom, k) .-1 
    plot(
        which_digit,
        st=:histogram,
        normalize=true,
        title="k = $k"
    )
end

# Number of eigen-digits
k = 3

@manipulate for digit in [0:9...]
    # Compute eigenimages
    U = svd(train[:, :, digit + 1])[1][:, 1:k]
    X = cat(
        2, 
        [reshape(U[:, i], (p, p))' for i in 1:size(U, 2)]...
    )
    
    # Display
    plot(
        flipdim(X, 1),
        st = :heatmap,
        color = :balance_r,
        aspect_ratio = :equal,
        ticks = []
        )
end

# Choose different digits

@manipulate for digit in [0:9...], 
    α2 in -0.5:0.1:0.5, 
    α3 in -0.5:0.1:0.5
    
    # Eigen-digits
    U = svd(train[:, :, digit + 1])[1][:, 1:3]
    α1 = 1
    # Combined image
    u = U * [α1; α2; α3]
    
    # Format for display
    U = scaleto01(U, 1) # scale columns to [0, 1]
    u = scaleto01(u, 1) # scale columns to [0, 1]
    X = cat(
        2, 
        [reshape(U[:, i], (p, p))' for i in 1:size(U, 2)]...
    )
    x = reshape(u, (p, p))
    
    # Display
    gap = ","*repeat(".", 12)*","
    p1 = plot(
        flipdim(X, 1),
        st = :heatmap,
        title="a1 = $α1"*gap*"a2 = $α2"*gap*"a3 = $α3",
        color = :balance_r,
        aspect_ratio = :equal,
        ticks = []
        )
    p2 = imshow(
        x,
        title="u1*a1 + u2*a2 + u3*a3",
        color=:balance_r, # :balance_r, :grays_r
    )
    plot(
        p1, p2,
        layout=Plots.grid(2, 1, heights=[0.5, 0.5]),
    )
end

# Choose different digits
# Adjust the alpha sliders to combine the eigenimages in different ways

function predict_missing_entries_ls(y1,A)
    A1 = A[1:size(y1,1),:]      # TODO: Fill in ?? 
    A2 = A[size(y1,1)+1:end,:]  # TODO: Fill in ??
    y2 = A2 * pinv(A1) * y1     # TODO Fill in ?? Hint: Write y1 and y2 in terms of x
    
    return y2
end

test_idx = rand(1:size(test, 2))
test_image = test[:, test_idx]
test_image_label = test_label[test_idx]
test_image_label = Int(test_image_label)
test_image_portion = test_image[1:392]
imshow(reshape(test_image_portion, 28, 14))
plot!(title="Top half of image: Correct label = $test_image_label")

@manipulate for k = [1, 3, 5, 10, 20, 100, 200, 784]
    U = svd(train[:, :, test_image_label + 1])[1][:, 1:k]
    predicted_image = predict_missing_entries_ls(float(test_image_portion),U) 
    completed_image = vcat(test_image_portion,predicted_image)
    p1 = imshow(reshape(completed_image,28,28))
    plot!(title = "Reconstruction: k = $k")
    p2 = imshow(reshape(test_image,28,28))
    plot!(title = "True image")
    
    plot(p1, p2,
        layout = Plots.grid(1, 2, heights=[0.5, 0.5]),
    )
end
## TODO: which U should you use in ?? Hint: assume test_image_label is known
## Advanced TODO: What if test_image_label is not known? 

relu(z) = max(0,z)
linear(z) = z
@manipulate for k = [5, 10, 20], f = [linear, relu]
    U = svd(train[:, :, test_image_label + 1])[1][:, 1:k]
    predicted_image = f.(predict_missing_entries_ls(float(test_image_portion),U))  # Fill in the ??
    completed_image = (vcat(test_image_portion,predicted_image))
    p1 = imshow(reshape(completed_image,28,28))
    plot!(title = "Reconstruction: k = $k")
    p2 = imshow(reshape(test_image,28,28))
    plot!(title = "True image")
    
    plot(p1, p2,
        layout = Plots.grid(1, 2, heights=[0.5, 0.5]),
    )
end

