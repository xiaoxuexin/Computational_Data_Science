
Pkg.add("Suppressor")
using Suppressor
using Plots, Interact, Optim, LightGraphs, GraphPlot, Colors
gr(
    markerstrokewidth=0.3,
    markerstrokecolor="white",
    label="",
    markersize=4
)
include("deps.jl")

function optimxTAx(A::Matrix; maxiters::Integer=1000, x0=normalize(randn(size(A, 1))))
    m, n = size(A)
       opt = Optim.optimize(
        x -> -x' * (A * x), # objective function to be *minimized*  
        x0,       # Initialization vector
        Optim.ConjugateGradient(manifold=Optim.Sphere()), # optimizer and manifold 
        Optim.Options(iterations=maxiters)
    )
    xopt = opt.minimizer
    return xopt
end

θ = 2π * rand() # random angle between 0 and 2π
W = [cos(θ) sin(θ); sin(θ) -cos(θ)] # generate a random Q
A = W * Diagonal([1;2]) * W' ## generate a random symmetric A 

xopt = optimxTAx(A)

W' * xopt

q₁ = eigs(A; nev=1, which=:LR)[2]
@show q₁' * xopt;

num_nodes = 3
g = SimpleGraph(num_nodes)
add_edge!(g, 1, 2)
graph_plot(g)

nv(g)

ne(g)

add_edge!(g, 2, 3)
graph_plot(g)

for e in edges(g)
    println("source $(src(e)), destination $(dst(e))") 
end

@show degree(g);

add_edge!(g, 1, 3) ## Add edge to make it a complete graph
graph_plot(g)

g_complete = CompleteGraph(5)
graph_plot(g_complete; layout=spring_layout) ## run twice if layout looks too "clumpy"

@show degree(g_complete)';

graph_plot(g)

incidence_matrix(g)

full(incidence_matrix(g))

C_complete = full(incidence_matrix(g_complete))

A_complete = adjacency_matrix(g_complete)

full(A_complete)

hmap(A_complete; color=:Greys)

g_2clique = CliqueGraph(5,2)
nodelabel = [string(s) for s in vertices(g_2clique)]
gplot(g_2clique; layout=spring_layout, nodelabel=nodelabel)

A_2clique = adjacency_matrix(g_2clique)
hmap(A_2clique; color=:Greys)

num_vertices = length(vertices(g_2clique))
vertex_perm = randperm(num_vertices)
A_2clique_perm = A_2clique[vertex_perm, vertex_perm]
g_2clique_perm = SimpleGraph(A_2clique_perm)
nodelabel = [string(s) for s in vertices(g_2clique_perm)]
gplot(g_2clique_perm; layout=spring_layout, nodelabel=nodelabel)

hmap(A_2clique_perm; color=:Greys)

gplot(g_2clique; layout=spring_layout, nodelabel=nodelabel)

function adjacency2modularity(A)
    degree_vector = sum(A, 2) ##TODO: Fill in ??
    
    total_edges = sum(sum(A, 1)) ##TOD: Fill in ??
    
    B = A - degree_vector * degree_vector' / total_edges ## Hint: Express Pij as an outer-product!
    return B 
end

B_2clique_perm = adjacency2modularity(A_2clique_perm)
hmap(B_2clique_perm)

n = size(B_2clique_perm, 1)
@show ones(n)' * B_2clique_perm * ones(n);

function maxmodularity_eig(B::Matrix)
    n = size(B,1)
    seig= sqrt(n) * eigs(B; nev=1, which=:LR)[2] ## TODO: What should the seig be so it solves the spherical problem
    seig = sign(seig[1]) * seig # to make s unique
    sopt  = sign.(seig)
    return sopt, seig
end

sopt_eig, seig = maxmodularity_eig(B_2clique_perm)
scatter(seig; xlabel="vertex", ylabel="value", label = "sopt")
scatter!(sopt_eig; xlabel="vertex", label="seig",color=:red)


which_group(x) = sign(x) < 1 ? 1 : 2
membership = which_group.(sopt_eig[:])
ngroups = length(unique(membership))
nodecolor = distinguishable_colors(ngroups + 1)
nodelabel = [string(s) for s in vertices(g_2clique_perm)]
gplot(g_2clique_perm; nodelabel=nodelabel, nodefillc=nodecolor[membership + 1])

g_karate = graphfamous("karate")
nodelabel = [string(s) for s in collect(vertices(g_karate))]
gplot(g_karate; layout=random_layout, nodelabel=nodelabel)

A_karate = adjacency_matrix(g_karate)
hmap(A_karate; color=:Greys, size=(600, 500))

B_karate = adjacency2modularity(A_karate)
sopt_eig, s_eig = maxmodularity_eig(B_karate)
membership = which_group.(sopt_eig[:]) 
gplot(g_karate; nodelabel=nodelabel, nodefillc=nodecolor[membership + 1])

@show find(membership .== 1);
@show find(membership .== 2);

using DataFrames, CSV
header = [
    "date",
    "number",
    "day",
    "vt",
    "vl",
    "vn",
    "ht",
    "hl",
    "hn",
    "time",
    "postpone",
    "makeup_date"
]
mlb = CSV.read("mlbgamedata_2017.txt"; header=header)

all_teams = Vector{String}(unique(cat(1, mlb[:ht], mlb[:vt])))
nteams = length(all_teams)
s = enumerate(sort(all_teams))
collect(s)

league_membership = Vector(length(all_teams))
for (idx, s) in enumerate(sort(all_teams))
    i = findfirst(mlb[:ht], s)
    league_membership[idx] = mlb[:hl][i]
end

american_league = sort(all_teams)[league_membership .== "AL"]

national_league = sort(all_teams)[league_membership .== "NL"]

all_teams = cat(1, american_league, national_league) # sorted by league, then alphabetically

h = [findfirst(all_teams, s) for s in mlb[:ht]]
v = [findfirst(all_teams, s) for s in mlb[:vt]]
nrows = length(h)
vals = ones(Int, nrows)
X_mlb = sparse(h, v, vals, nteams, nteams)
team_labels = cat(1, [s * "-AL" for s in american_league], [s * "-NL" for s in national_league])
hmap(
    X_mlb, team_labels; 
    color=:Greys,
    xaxis=(rotation=90), 
    aspect_ratio=1.0,
    size=(600, 500)
)

X_mlb_g = Graph((X_mlb + X_mlb') / 2)
edgeweights = nonzeros(X_mlb)[1:ne(X_mlb_g)]

@suppress gplot(X_mlb_g,layout=spring_layout, nodelabel=team_labels, edgelinewidth=edgeweights) 
## (X_mlb+X_mlb')/2 because X_mlb appears not to be symmetric

g_mlb = SimpleGraph(X_mlb + X_mlb')
A_mlb = adjacency_matrix(g_mlb) # make sure A is 0-1
hmap(
    A_mlb, team_labels;
    color=:Greys,
    clim=(0,1),
    xaxis=(rotation=90), 
    aspect_ratio=1.0,
    size=(600, 500)
) # black & white only

B_mlb = adjacency2modularity(A_mlb)
sopt_mlb = maxmodularity_eig(B_mlb)[1]
membership = which_group.(sopt_mlb[:])
ngroups = length(unique(membership))
nodecolor = distinguishable_colors(ngroups + 1)
gplot(g_mlb, nodefillc=nodecolor[membership + 1], nodelabel=team_labels)

using Clustering

num_mlb_divisions = 6
λ, S_mlb = eigs(B_mlb; which=:LM, nev=num_mlb_divisions)
@show λ
membership_mlb = kmeans(S_mlb', num_mlb_divisions).assignments
ngroups_mlb = length(unique(membership_mlb))
nodecolor = distinguishable_colors(ngroups_mlb + 1)
gplot(g_mlb,nodefillc=nodecolor[membership_mlb + 1], nodelabel=team_labels, edgelinewidth=edgeweights)

num_mlb_divisions = 6
λ, S_mlb = eigs(B_mlb; which=:LR,nev=num_mlb_divisions)
@show size(S_mlb)
membership_mlb = kmeans(S_mlb', num_mlb_divisions).assignments
ngroups_mlb = length(unique(membership_mlb))
nodecolor = distinguishable_colors(ngroups_mlb + 1)
gplot(g_mlb,nodefillc=nodecolor[membership_mlb + 1], nodelabel=team_labels, edgelinewidth=edgeweights)

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

num_mlb_divisions = 6
λ, S_mlb = eigs(B_mlb; which=:LM,nev=num_mlb_divisions)
@show size(S_mlb)
membership_mlb = simple_kmeans(S_mlb', num_mlb_divisions)[1]
ngroups_mlb = length(unique(membership_mlb))
nodecolor = distinguishable_colors(ngroups_mlb + 1)
gplot(g_mlb, nodelabel=team_labels, edgelinewidth=edgeweights,nodefillc=nodecolor[membership_mlb'[:,1] + 1])

##TODO: Scatter plot of eigenvalues of B_mlb -- what do you notice about the plot? 

function mincut_eig(A::Matrix)
    d = sum(A, 1)
    D = Diagonal(d[:])
    L = D - A
    U = sqrt(size(A,1))*eigs(L; nev=2, which=:SR)[2]
    s = U[:,2] ## Hint: which order does eigs return the eigenvalues? 
    s = sign(s[1]) * s # to make s unique
    sopt  = sign.(s)
    return sopt, L
end

lapla = mincut_eig(full(A_karate))
membership = which_group.(lapla[:]) 
gplot(g_karate; nodelabel=nodelabel, nodefillc=nodecolor[membership + 1])

##TODO: Code for  partitioning MLB using laplacian and displaying the graph 
s_laplacian = mincut_eig(full(A_mlb))
membership = which_group.(s_laplacian[:]) 
ngroups = length(unique(membership))
nodecolor = distinguishable_colors(ngroups + 1)
gplot(g_mlb, nodefillc=nodecolor[membership + 1], nodelabel=team_labels)
## Plot the graph 

num_mlb_divisions = 6
B_mlb = adjacency2modularity(A_mlb)
S_mlb = mincut_eig(B_mlb)[2]
@show(size(S_mlb))
membership_mlb = kmeans(S_mlb', num_mlb_divisions).assignments
ngroups_mlb = length(unique(membership_mlb))
nodecolor = distinguishable_colors(ngroups_mlb + 1)
gplot(g_mlb,nodefillc=nodecolor[membership_mlb + 1], nodelabel=team_labels, edgelinewidth=edgeweights)

function hashimoto_eig(A::Matrix)
    n = size(A, 1)
    d = sum(A, 1)
    D = Diagonal(d[:])
    
    Abar = vcat(hcat(A, eye(n) - D), hcat(eye(n), zeros(n,n)))
    λbar, Ubar = eigs(Abar; maxiter = 5000, nev=2, which=:LM)
#     λbar, Ubar = eigs(Abar; maxiter = 5000, nev=7, which=:LM)

    s = real.(Ubar[1:n, 2])
#     s = real.(Ubar[1:n, 2:7])

    s = sign(s[1]) * s ## to make s unique
    sopt  = sign.(s)
    return sopt, Abar
end

B_karate = adjacency2modularity(A_karate)
hash = hashimoto_eig(B_karate)[1]
membership = which_group.(hash[:]) 
gplot(g_karate; nodelabel=nodelabel, nodefillc=nodecolor[membership + 1])

##TODO: Code for  partitioning MLB using Hashimoto matrix into two communities and displaying the graph 
## Plot the graph displaying communities  
# B_mlb = adjacency2modularity(A_mlb)
hash_mlb = hashimoto_eig(full(A_mlb))[1]
membership = which_group.(hash_mlb[:]) 
ngroups = length(unique(membership))
nodecolor = distinguishable_colors(ngroups + 1)
gplot(g_mlb, nodefillc=nodecolor[membership + 1], nodelabel=team_labels)

##TODO: Code for  partitioning MLB using Hashimoto matrix into six communities and displaying the graph 
## Hint: You cannot use sopt and will have to use Abar directly
num_mlb_divisions = 6
sopt, A_bar = hashimoto_eig(full(A_mlb))
@show λ
membership_mlb = kmeans(A_bar', num_mlb_divisions).assignments
ngroups_mlb = length(unique(membership_mlb))
nodecolor = distinguishable_colors(ngroups_mlb + 1)
gplot(g_mlb,nodefillc=nodecolor[membership_mlb + 1], nodelabel=team_labels, edgelinewidth=edgeweights)

include("simple_kmeans.jl") ## upload your code to the directory before running this

_, U = eigs(A_mlb')
cluster_idx = simple_kmeans(U',6)[1][:]
for idx = 1 : 6
    @show all_teams[cluster_idx .== idx]
end

## TODO: Display a plot of the communities found using your kmeans

_, U = eigs(X_mlb+X_mlb')
cluster_idx = simple_kmeans(U',6)[1][:]
for idx = 1 : 6
    @show all_teams[cluster_idx .== idx]
end

## TODO: Display a plot of the communities found using your kmeans

cin = 15
cout = 2
n = 20
g_sbm = stochastic_block_model(cin, cout, [n;n])
membership_sbm = Int.([ones(n); 2 * ones(n)])[:]
ngroups_sbm = length(unique(membership_sbm))
nodecolor = distinguishable_colors(ngroups_sbm + 1)
gplot(g_sbm; nodefillc=nodecolor[membership_sbm + 1])

##TODO: Compute and plot the communities displayed by the modularity, Laplacian and Hashimoto method



## TODO: Set cin + cout = 100 and let n = 1000
## Generate a SBM from above model
## We know that first half of vertices belong to group 1 and the remainder below to group 2
## Step 1: Use the spectral modularity method to partition the network and infer the group membership
## Step 2: Compute the probability that a vertexis correctly classified 
##      Tip: This is a bit tricky and you will want to ensure that Pcorrect is never below 0.5
## Step 3: Plot Pcorrect, averaged over 100 trials versus cin - cout 
##         for about 20 different values of cin-cout from 0 to n/2
prob_modularity_20 = []
n = 1000
for cin = 10:20
    cout = 20 - cin
    p = 0
    prob = 0
    g_sbm = stochastic_block_model(cin, cout, [n;n])
    membership_sbm = Int.([ones(n); 2 * ones(n)])[:]
    for i = 1:10
        
        A_2clique = adjacency_matrix(g_sbm)
        B_2clique_perm = adjacency2modularity(A_2clique)
        sopt_eig, s_eig = maxmodularity_eig(B_2clique_perm)
        membership = which_group.(sopt_eig[:]) 
        p = (sum(membership[1:1000,] .== 1) + sum(membership[1001:end,] .== 2))/2000
        if (p < 0.5)
            p = 1-p
        end
        prob += p/10
    end
    append!(prob_modularity_20,prob)

end
prob_modularity_20

##TODO: Repeat above for Laplacian matrix -- 
prob_Laplacian_20 = []
for cin = 10:20
    cout = 20 - cin
    n = 1000
    g_sbm = stochastic_block_model(cin, cout, [n;n])
    membership_sbm = Int.([ones(n); 2 * ones(n)])[:]
    p = 0
    prob = 0
    for i = 1:10
        A_2clique = adjacency_matrix(g_sbm)
        s_laplacian = mincut_eig(full(A_2clique))[1]
        membership = which_group.(s_laplacian[:]) 
        p = (sum(membership[1:1000,] .== 1) + sum(membership[1001:end,] .== 2))/2000
        if (p < 0.5)
            p = 1-p
        end
        prob += p/10
    end
    append!(prob_Laplacian_20,prob)

end
prob_Laplacian_20

##TODO: Repeat above for Hashimoto matrix 
prob_hash_20 = []
for cin = 5:10
    cout = 10 - cin
    n = 500
    g_sbm = stochastic_block_model(cin, cout, [n;n])
    membership_sbm = Int.([ones(n); 2 * ones(n)])[:]
    p = 0
#     prob = 0
#     for i = 1:10
#         @show i
    A_2clique = adjacency_matrix(g_sbm)
    s_hash = hashimoto_eig(full(A_2clique))[1]
    membership = which_group.(s_hash[:]) 
    p = (sum(membership[1:1000,] .== 1) + sum(membership[1001:end,] .== 2))/2000
    @show cin
    if (p < 0.5)
        p = 1-p
    end
#         prob += p/10
#     end
    append!(prob_hash_20,p)

end
prob_hash_20

##TODO: Super-impose all three plots -- which is better?
cin = 10:20
cout = 20 .- cin
plot(cin .- cout, prob_modularity_20, label = "modularity", legend =:topleft)
plot!(cin .- cout, prob_Laplacian_20, label = "Laplacian")
# plot!(cin .- cout, prob_Hashimoto_20, label = "Hashimoto")

## TODO: Compare the performance of the three algorithms for  cin + cout = 20 and let n = 1000

##TODO: Code and plot for modularity method 

##TODO: Code and plot for Laplacian method 

##TODO: Code and plot for Hashimoto method 

##TODO: Plot comparing all three 

## TODO: Compare the performance of the three algorithms for  cin + cout = 20 and let n = 1000

##TODO: Code and plot for modularity method 

##TODO: Code and plot for Laplacian method 

##TODO: Code and plot for Hashimoto method 

##TODO: Plot comparing all three 
