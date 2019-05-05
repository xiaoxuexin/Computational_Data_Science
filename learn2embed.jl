
include("deps.jl")
using JSON, MAT, Interact, Plots, Shapefile, PlotRecipes
gr(
    label="",
    markerstrokewidth=0.3,
    markerstrokecolor="white",
    markersize=6,
    alpha=0.8
)

"""
Function for convenient plotting of a digit, represented as an n-by-2 matrix.
"""
function plot_digit(X::Matrix, point_labels::Vector=string.(1:size(X, 1)); kwargs...)
    plot(
        X[:, 1], X[:, 2];
        marker=:circle, 
        aspect_ratio=:equal,
        kwargs...)
    annotate!(X[:, 1] + 1.5, X[:, 2], point_labels)
end
function plot_digit!(X::Matrix, point_labels::Vector=string.(1:size(X, 1)); kwargs...)
    plot!(
        X[:, 1], X[:, 2];
        marker=:circle, 
        aspect_ratio=:equal,
        kwargs...)
    annotate!(X[:, 1] + 1.5, X[:, 2], point_labels)
end

ll = split("a b c d e f g h i j")
d = matread("misaligned_digit1.mat")
X = d["X"]
@show size(X);
plot_digit(X, ll; xlim=(10, 40))

"""
    D = coords2dist(X, distance)

Input: `X` is an n x d matrix whose rows contains the d coordinates of the n objects 

Output:
* `D` is an n x n matrix such that D[i, j] is the distance from object i to object j
* `distance` is the function used to measure the distance between coordinates of objet i and objet j
*  default `distance`  is `vecnorm` which computes the usual Euclidean norm
"""
function coords2dist(X::Matrix, distance::Function=vecnorm)
    n, d = size(X)
    D = [distance(X[i,:] - X[j,:]) for i in 1:(n), j in 1:(n)]
    return D
end

D = coords2dist(X)

plot(
    hmap(D, ll),
    plot_digit(X, ll; xlim=(10, 40)); 
    layout=(1,2),
    size=(700, 350)
)

# shift by +5 in x dimension, and +10 in y dimension
μ = [5 10]
X_shifted = X .+ μ
plot_digit(X, ll; xlim=(10, 45))
plot_digit!(X_shifted, ll)

D_shifted = coords2dist(X_shifted)
@show vecnorm(D_shifted - D)
hmap(D_shifted - D, ll; title="D_shifted - D")

θ = π/4
@show Q = [cos(θ) -sin(θ); sin(θ) cos(θ)]
X_rotated = (X .+ μ)*Q
plot_digit(X, ll; xlim=(10, 70), ylim=(-10, 50))
plot_digit!(X_rotated, ll; marker=:square)

D_rotated = coords2dist(X_rotated)
@show vecnorm(D_rotated - D)
hmap(D_rotated - D, ll; title="D_rotated - D")

l1_distance(x) = sum(abs, x)
D_l1 = coords2dist(X, l1_distance)
D_l1_rotated = coords2dist(X_rotated, l1_distance)
@show vecnorm(D_l1_rotated - D)
hmap(D_l1_rotated - D_l1, ll; title="D_l1_rotated - D_l1", colorbar=true)

"""
    Xr = dist2locs(D, d)

Inputs:
* `D` is an n x n matrix such that D[i, j] is the distance from object i to
 object j
* `d` is the desired embedding dimension. The actual `d` value used should be
 capped at the number of positive eigenvalues of K

Output: `Xr` is an n x d matrix whose rows contains the relative coordinates
 of the n objects and e the n eigenvalues of K
 
Note: MDS is only unique up to rotation and translation, so we
enforce the following conventions on Xr:
             
* [ORDER] Xr[:, i] corresponds to ith largest eigenpair of K
* [EIG] Actual d used is min(d, # positive eigenvalues of K)
* [CENTER] The rows of Xr have zero centroid
* [SIGN] The largest magnitude element of Xr[:, i] is positive
"""
function dist2coords(D::Matrix, d::Integer)
    n = size(D, 1)

    # Compute correlation matrix
    S = D.^2
    P = eye(n) - ones(n,1) * ones(1,n) / n
    K = -0.5 * P * S * P' 
    K = 0.5 * (K + K') # Force symmetry to eliminate any round-off errors

    # Compute relative coordinates
    e, V = eig(K)
    d = min(d, countnz(e .> 0)) # Apply [EIG] 
    idx = sortperm(e, rev=true)
    Xr = V[:, idx[1:d]] * Diagonal(sqrt.(e[idx[1:d]])) # Apply [ORDER] 

    # Apply [CENTER]
    Xr .-= mean(Xr, 1) ## TODO: Fill in ?? -- do we average over rows or columns?

    # Apply [SIGN]
    Xr .*= sign(Xr[findmax(abs(Xr), 1)[2]]) ## TODO: Fill in ??

    return Xr, e
end

Xest, eigK = dist2coords(D, 2)
plot_digit(Xest, ll; label="MDS")
plot_digit!(X; marker=:square, label="Known", legend=:topleft)

include("myprocrustes.jl") # copy/paste your working `procrustes` function into the myprocrustes.jl file
using myprocrustes 

Xest_aligned, _ = procrustes(X, Xest)

plot_digit(Xest_aligned, ll; label="Aligned MDS", alpha=0.8)
plot_digit!(X, ll; marker=:square, label="Known", alpha=0.4, legend=:bottomleft)

scatter(eigK, xlabel="index", ylabel="eig(K)")

println("Eigenvalues of K equal \n")
eigK

using Distances # package for computing Distances efficiently
D_city = pairwise(Cityblock(), X');
print(D_city)
X_city_est, eigK = dist2coords(D_city, 2)

plot_digit(X_city_est, ll; label="CityBlock MDS")
plot_digit!(X, ll; marker=:square, label="Known", legend=:topleft)

X_l1_est, eigK = dist2coords(D_l1, 2)
plot_digit(X_l1_est, ll; label="l1 MDS")
plot_digit!(X, ll; marker=:square, label="Known", legend=:topleft)

X_l1_aligned, _ = procrustes(X, X_l1_est)

pdigit_l1 = plot_digit(X_l1_aligned, ll; legend=:topright, label="l1 aligned MDS")
plot_digit!(X, ll; marker=:square, ylim=(10, 60), xlim=(0, 50), label="Known")

plot(
    hmap(D - D_l1, ll; title="D_l2 - D_l1"),
    pdigit_l1,
    layout=(1, 2),
    size=(700, 350)
)

cities = ["Atl","Chi","Den","Hou","LA","Mia","NYC","SF","Sea","WDC"]

D = [    0  587 1212  701 1936  604  748 2139 2182   543;
       587    0  920  940 1745 1188  713 1858 1737   597;
      1212  920    0  879  831 1726 1631  949 1021  1494;
       701  940  879    0 1374  968 1420 1645 1891  1220;
      1936 1745  831 1374    0 2339 2451  347  959  2300;
       604 1188 1726  968 2339    0 1092 2594 2734   923;
       748  713 1631 1420 2451 1092    0 2571 2408   205;
      2139 1858  949 1645  347 2594 2571    0  678  2442;
      2182 1737 1021 1891  959 2734 2408  678    0  2329;
       543  597 1494 1220 2300  923  205 2442 2329     0];

"""
Conveniently n-dimensional plot points encoded in a matrix.
Better than scatter(X[:, 1], X[:, 2], X[:, 3]) etc.
"""
function scatter_nd(X; kwargs...)
    scatter((X[:, i] for i in 1:size(X, 2))...; kwargs...)
end

function scatter_nd!(X; kwargs...)
    scatter!((X[:, i] for i in 1:size(X, 2))...; kwargs...)
end

Xest = dist2coords(D, 2)[1]
scatter_nd(
    Xest;
    aspect_ratio=:equal, 
    legend=:topright
)
annotate!(
    Xest[:,1],
    Xest[:,2] + 80,
    cities,
    font("Sans", 9)
)
plot!(
    xlims=extrema(1.2*Xest[:,1]), 
    ylims=extrema(1.2*Xest[:,2])
)

Xest = Xest*Diagonal([-1, -1])
scatter_nd(Xest; aspect_ratio=:equal)
annotate!(
    Xest[:,1],
    Xest[:,2] + 80,
    cities,
    font("Sans", 9)
)
plot!(
    xlims=extrema(1.15*Xest[:,1]), 
    ylims=extrema(1.15*Xest[:,2])
)

@show eigK
scatter(eigK, label="eig(K)", legend=:topleft)

# import functions for getting distances between cities from Google
include("map_querying_dependencies.jl")

cities = [
    "Ann Arbor, MI", 
    "Detroit, MI", 
    "Toledo, OH", 
    "Cleveland, OH",
    "Buffalo, NY", 
    "Chicago, IL"
    ]
country_code = "US"
Dgoogle = city_distances(cities)

# Compute relative coordinates via MDS
Xest = dist2coords(Dgoogle, 2)[1]
scatter_nd(Xest; label="raw MDS w/ Dgoogle")
annotate!(
    Xest[:,1],
    Xest[:,2] + 10,
    cities,
    font("Sans", 9)
)
plot!(
    xlims=extrema(1.5*Xest[:,1]), 
    ylims=extrema(1.5*Xest[:,2])
)

# Get (lat, long) coordinates for our cities
latlong = map(cities) do city
    getLatLonViaYql(city, country_code)
end
filter!(x -> x[1] != 0, latlong)

# Collect into [long, lat] matrix
#     longitude = x coordinate
#     latitude  = y coordinate
citieslonglat = cat(2,
    map(last, latlong),
    map(first, latlong),
)

Zgoogle = procrustes(citieslonglat, Xest)[1] ##TODO: Fill in ??

# US shape file map downloaded from  https://www.census.gov/geo/maps-data/data/cbf/cbf_nation.html 
united_states = open("cb_2017_us_nation_20m.shp") do fd
    read(fd, Shapefile.Handle) 
end

Zgoogle = procrustes(citieslonglat, Xest)[1] ##TODO: Fill in ??

# Display aligned coordinates
scatter(
    Zgoogle[:, 1], Zgoogle[:, 2],
    label="MDS w/ Dgoogle",
    xaxis="Longitude",
    yaxis="Latitude",
    text=cities,
    marker=:square,
    color=:red,
)
# Display true geo coordinates
scatter(
    citieslonglat[:, 1], citieslonglat[:, 2], 
    text=cities,
    label="True geo coords.",
    marker=:square,
    color=:red,
    legend=:bottomright
)

# geographic distances between cities in miles based on their (lat, long) coordinates
Dgeo = [
    lldistkm(
        citieslonglat[idx1, 2:-1:1], # flip to [lat, long]
        citieslonglat[idx2, 2:-1:1], # flip to [lat, long]
    )[1] for idx1 in 1:length(cities), idx2 in 1:length(cities)
] ./ 1.60934

cities_short = map(s -> split(s, ",")[1], cities)
hmap(Dgoogle - Dgeo, cities_short; title="Dgoogle - Dgeo")

cities_short = map(s -> split(s, ",")[1], cities)
p1 = hmap(Dgoogle - Dgeo, cities_short; title="Dgoogle - Dgeo")

p2 = plot(
    united_states, 
    xlim=(-90, -77),
    ylim = (40, 47),
    color="green", 
    alpha=0.2,
    ticks=[],
    xaxis="Longitude",
    yaxis="Latitude"
)
scatter!(
    citieslonglat[:, 1], citieslonglat[:, 2], 
    color=:blue,
    legend=:topright
)
annotate!(
    citieslonglat[:,1],
    citieslonglat[:,2] + 0.2,
    cities,
    font("Sans", 8)
)

plot(layout=(1, 2), size=(950, 350), p1, p2)

X1 = randn(2,400) 
X2 = randn(2,400) 
X = hcat(X1.+ [10;5], X2.+ [0;5])
scatter_nd(X')

function simple_kmeans(
        X::Matrix, 
        k::Integer, 
        iters::Integer=100,
        cluster_idx::Vector=rand(1:k, size(X, 2))
    )
    
    # Input: X = d x n data matrix where n is number of samples and X[:,i] is a d-dimensional vector contain
    #       containing the d coordinates of the i-th sample 
    #        k = # clusters
    #        iters = # iterations to run the algorithm
    #        cluster_idx = initial assignment of data into clusters
    # Output: cluster_idx, C = matrix of Centroids corresponding to clusters and intra_cluster_sq_error 
    
    
    # Initialize array of cluster centroids
    C = zeros(size(X, 1), k)
    within_cluster_sqerror = zeros(k)

    for iter_idx = 1:iters
        D = zeros(k, size(X, 2))

        
        # Step 1: 
        # Computer centroid of each cluster and store in columns of C
        # Also compute squared distance of each point to its cluster's centroid
        for i = 1 : k
            cluster_i_idxs = find(cluster_idx .== i)
            C[:, i] = mean(X[:, cluster_i_idxs], 2)
            within_cluster_sqerror[i] = sum(abs2.(X[:, cluster_i_idxs] .- C[:,i]))
        end
        
        # Step 2:
        # Compute distance from each point to cluster centroids 
        # D[i,j] = distance of point j to cluster centroid i
        
        for i = 1 : k 
            for j = 1 : size(X,2)
                D[i,j] = vecnorm(C[:, i] - X[:, j])
            end
        end
    
        # Step 3:
        # Compute new cluster assignment based on which centroid each point is closer to
        cluster_idx = mapslices(indmin, D, 1)
    end
    
    intra_cluster_sqerror = sum(within_cluster_sqerror)
    
    return cluster_idx, C, intra_cluster_sqerror  
end

X1 = randn(2, 400)
X2 = randn(2, 400)

@manipulate for iters in [0, 1, 10, 100, 1000], Δ = [1, 0.1, 0.01], k = [2, 1, 3]
    X = hcat(X1.+ Δ*[10;5], X2 .+Δ*[0;5])
    
    # prevent colors from flipping
    srand(1)
    cluster_idx = rand(1:k, size(X, 2))
    cluster_idx = simple_kmeans(X, k, iters, cluster_idx)[1][:]
    scatter_nd(X[:, cluster_idx .== 1]')
    scatter_nd!(X[:, cluster_idx .== 2]')
    scatter_nd!(X[:, cluster_idx .== 3]')
end

# Synthetic distance matrix for three equidistant points
D = ones(3, 3) - eye(3)

# Compute three-dimensional relative coordinates via MDS
Xcoords = dist2coords(D, 3)[1]

# Visualize coordinates
scatter_nd(
    Xcoords;
    marker=:square,
    title="Global coords"
)

# Visualize first two coordinate dimensions only
scatter_nd(
    Xcoords[:, 1:2];
    marker=:square,
    title="Global coords"
)

# Synthetic distance matrix for four equidistant points
D = ones(4, 4) - eye(4)

# Compute three-dimensional relative coordinates via MDS
Xcoords = dist2coords(D, 3)[1]

# Visualize first two relative coordinates
scatter_nd(
    Xcoords[:, 1:2];
    marker=:square,
    title="Global coords"
)

# Visualize first three relative coordinates
plotly()
scatter_nd(
    Xcoords,
    marker=:square,
    title="Global coords"
)

train_data = matread("train_digits.mat")["train_data"]
digits = [0, 5, 7]
X = hcat(hcat(train_data[:,:,digits[1]+1], train_data[:,:,digits[2]+1]), train_data[:,:,digits[3]+1])
X = float(X[:,1:5:end]);

using Distances # package for computing Distances efficiently

using Distances # package for computing Distances efficiently
D_digits = pairwise(Euclidean(), X);
Xdigit = dist2coords(D_digits,2)[1]
X0 = Xdigit[1:160, :]
X1 = Xdigit[161:320, :]
X2 = Xdigit[321:480, :]
pdigits_truth_euclidean = scatter_nd(X0, label="$(digits[1])")
scatter_nd!(X1, label="$(digits[2])")
scatter_nd!(X2, label="$(digits[3])")

title!("Ground truth labels")

k = 3
@manipulate for iters = [0, 1, 10, 100]
    cluster_idx = simple_kmeans(Xdigit', k, iters)[1][:]
    pmds_digits_euclidean = scatter_nd(Xdigit; color=cluster_idx)
    title!("k-means after $(iters) iterations")
    plot(pmds_digits_euclidean, pdigits_truth_euclidean; layout=(1,2), size=(700, 350))
end

D_digits = pairwise(CosineDist(), X)
Xdigit = dist2coords(D_digits, 2)[1]
X0 = Xdigit[1:160, :]
X1 = Xdigit[161:320, :]
X2 = Xdigit[321:480, :]
pdigits_truth_cosine = scatter_nd(X0, label="$(digits[1])")
scatter_nd!(X1, label="$(digits[2])")
scatter_nd!(X2, label="$(digits[3])")

title!("Ground truth labels")

k = 3
@manipulate for iters = [0, 1, 10, 100]
    cluster_idx = simple_kmeans(Xdigit', k,iters)[1][:]
    pmds_digits_cosine = scatter_nd(Xdigit; color=cluster_idx)
    title!("k-means: Cosine distance")
    plot(pmds_digits_cosine,pdigits_truth_cosine; layout=(1,2), size=(700, 350))
end

##TODO: Your code to cluster three digits using k-means 
digits = [0, 5, 7]
X = hcat(train_data[:, :, digits[1] + 1], train_data[:, :, digits[2] + 1], train_data[:, :, digits[3] + 1])
X = float(X[:,1:5:end]);

@manipulate for iters in [0, 1, 10, 100],  dist in [Euclidean,CosineDist]
    D_digits = pairwise(dist(), X);
    #TODO: Plot ground truth labels versus cluseri
end


##TO: For the digit dataset plot the  MDS retrieved when you use the cosine distance. 
## Align the coordinates to the original coordinates and display the aligned coordinates. 



##TODO: Your code here for the distance function 

##TODO: your code here to plot the aligned MDS coordinates 

##TODO: Your code here for the distance function 

##TODO: your code here to plot the aligned MDS coordinates 

groceries = ["orange", "apple", "spinach", "kale"]
num_groceries = length(groceries)
println("0 is similar, 10 is very dissimilar.")
D = zeros(num_groceries, num_groceries)
for i in 1:length(groceries)
     for j in 1:i-1 
        println("How dissimilar is $(groceries[i]) to $(groceries[j]) on a scale of 0-10?")
        dij = readline(STDIN)
        D[i,j] = parse(Float64, dij)
    end
end

Dgroceries = D + D'

Xgroceries, _ = dist2coords(Dgroceries, 2)
# Visualize coordinates
scatter_nd(Xgroceries; title="Groceries as MDS coords", text=groceries)
