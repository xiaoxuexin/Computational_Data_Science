
using Plots, Interact, Colors, ProgressMeter
gr(
    markerstrokewidth=0.3,
    markerstrokecolor="white",
    label="",
    markersize=4,
)
include("deps.jl")

function generate_spikewaveform(n::Integer=1000, a::Number=50.0, b::Number=100.0)
    t = (0.0:1.0:(n - 1.0)) / n
    return exp.(-a * t) .* (sin.(b * t)), t
end

@manipulate for a in [20,50,100], b in [50,100,150]
    n = 1000
    x, t = generate_spikewaveform(n, a, b)
    plot(t, x; xlabel="t", ylabel="x(t)")
end

n = 1000
x, t = generate_spikewaveform(n, 50, 100)
x = x / vecnorm(x)
plot(t, x; xlabel="t", ylabel="x(t)")

m, k = 20, 5
sync_idx = [3, 7, 13, 18, 20]
X = zeros(m, n)
for idx in sync_idx
    X[idx, :] = x'
end
labels = ["sensor $i" for i in 1:m]
joyplot(X'; labels=labels, subplot_scale=1.0)

@show rank(X);

@show maximum(svdvals(X))
@show sqrt(k)
scatter(svdvals(X); xlabel="index", ylabel="singular value")

u₁ = svds(X; nsv=1)[1][:U][:]
@show maximum(abs.(u₁))
@show maximum(u₁)
@show 1 / sqrt(k)
scatter(u₁; xlabel="i", ylabel="U[i,1]")

u₁ = sign.(u₁[findmax(abs.(u₁))[2]])*u₁
@show extrema(u₁)
@show sortperm(u₁; rev=true)
@show sync_idx
scatter(u₁; xlabel="i", ylabel="U[i,1]")

function synced_vector_svd(A::Matrix)
    U = svds(A;nsv=1)[1][:U]
    U = U*sign.(U[findmax(abs.(U))[2]]) # to fix sign
    return U
end

@manipulate for σ = [0, 0.005, 0.05, 0.1, 0.5]
    X_noisy = X + σ * randn(size(X))
    u₁ = synced_vector_svd(X_noisy)[:]
    top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
    synced_vector_svd(X_noisy)
    plot(
        joyplot(X_noisy'; labels=labels),
        scatter(
            u₁; 
            title="top 5 coords. idx = $top5_coord_idx", 
            xlabel="i", 
            ylabel="U{i,1}",
            ylim=(-0.1,1)
            ),
        layout=(1,2),
        size=(900, 400)
    )
    end

m, k = 20, 5
sync_idx =  [3, 7, 13, 18, 20]; 
c = 0.1
Z = zeros(m, n)
for idx in 1:m  
    if idx in sync_idx 
        Z[idx, :] = x' 
    else
        Z[idx, :] = c * ones(n)'
    end
end
labels = ["sensor $i" for i in 1:m]
joyplot(Z'; labels=labels, plot_size=(600, 400))

u₁ = synced_vector_svd(Z)[:]
top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
scatter(
    u₁;
    title="top-5 coords. idx = $top5_coord_idx", 
    xlabel="i", 
    ylabel="U{i,1}", 
    ylim=(-0.1, 1)
)

@manipulate for c in [0, 0.1, 0.5], σ in [0, 0.05, 0.1, 0.5]
    Z = zeros(m, n)
    for idx in 1:m  
        if idx in sync_idx 
            Z[idx, :] = x' 
        else
            Z[idx, :] = c * ones(n)'
        end
    end
    Znoisy = Z + σ * randn(size(Z))
    u₁ = synced_vector_svd(Znoisy)[:]
    top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
    plot(
        joyplot(Znoisy'; labels=labels),
        scatter(
            u₁; 
            title="top-5 coords. idx = $top5_coord_idx", 
            xlabel="i", 
            ylabel="U{i,1}",
            ylim=(-0.1,1)
            ),
        layout=(1, 2),
        size=(800, 400)
    )
    end

@show rank(Z)
scatter(svdvals(Z); xlabel="index", ylabel="singular value")

function synced_vector_robust2mean(A::Matrix)
    c = mean(A, 2)
    A = A .- c
    U = svds(A; nsv=1)[1][:U]
    U = U*sign.(U[findmax(abs.(U))[2]]) # to fix sign
    return U
end

@manipulate for c in [0, 0.05, 0.1, 0.5], σ in [0, 0.05, 0.1, 0.5]
    Z = zeros(m, n)
    for idx in 1:m  
        if idx in sync_idx 
            Z[idx, :] = x' 
        else
            Z[idx,:] = c * ones(n)'
        end
    end
    Znoisy = Z + σ * randn(size(Z))
    u₁ =  synced_vector_robust2mean(Znoisy)[:]
    top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
    plot(
        joyplot(Znoisy'; labels=labels),
        scatter(u₁; title = "top-5 coords. idx = $top5_coord_idx", 
            xlabel="i", 
            ylabel="U{i,1}",
            ylim=(-0.1, 1)),
        layout=(1, 2),
        size=(800, 400)
    )
    end

function synced_vector(A::Matrix)
    m = size(A, 1)
    K = zeros(m, m)
    # Only compute lower triangle part
    for i in 1:m
        for j in 1:i
            K[i, j] = dot(A[i, :] .- mean(A[i, :]), A[j, :] .- mean(A[j, :]))
        end
    end

    # Complete the matrix
    K = K + K' - Diagonal(diag(K))
    u = eigs(K; nev=1, which=:LR)[2]
    u = u * sign.(u[find(abs.(u) .== maximum(abs.(u)))]) # to find sign
    return u, K
end

Kx = synced_vector(X)[2]
@show rank(Kx)
hmap(Kx; xticks=:off, yticks=:off)

@manipulate for c in [0, 0.05, 0.1, 0.5], σ in [0, 0.01, 0.05,0.5]
    Z = zeros(m, n)
    for idx in 1:m  
        if idx in sync_idx 
            Z[idx, :] = x' 
        else
            Z[idx, :] = c * ones(n)'
        end
    end
    Znoisy = Z + σ * randn(size(Z))
    u₁, K = synced_vector(Znoisy)
    u₁ = u₁[:]
    top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
    p = plot(
        joyplot(Znoisy'; labels=labels),
        scatter(
            u₁; 
            title="max_idx = $top5_coord_idx", 
            xlabel="i", 
            ylabel="U{i,1}",
            ylim=(-0.1, 1)
            ),
        hmap(K; xticks=:off, yticks=:off),
    layout=(1, 3),
    size=(900, 400)
    )
end

function randshift(X::Matrix, shift::Vector{Int}=rand(75:150, size(X, 1)))
    Xshift = zeros(size(X))
    for idx in 1:length(shift)
        Xshift[idx, :] = circshift(X[idx, :]', (0, shift[idx]))
    end
    return Xshift
end

shift_vector = Array{Int}(m)
shift_vector[sync_idx] = [50; 160; 200; 320; 420]
Xshift = randshift(X, shift_vector)
labels =["sensor $i" for i in 1:m]
joyplot(Xshift'; labels = labels)

@manipulate for c in [0, 0.05, 0.1, 0.5], σ in [0, 0.01, 0.05,0.5]
    Z = zeros(m, n)
    for idx in 1:m  
        if idx in sync_idx 
            Z[idx, :] = x' 
        else
            Z[idx, :] = c * ones(n)'
        end
    end
    Znoisy = Z + σ * randn(size(Z))
    shift_vector[sync_idx] = [50; 160; 200; 320; 420]
    Zshift_noisy = randshift(Znoisy, shift_vector)
    u₁, K = synced_vector(Zshift_noisy)
    u₁ = u₁[:]
    top5_coord_idx = sort(sortperm(u₁; rev=true)[1:5])
    p = plot(
        joyplot(Zshift_noisy'; labels=labels),
        scatter(
            u₁;
            title="max_idx = $top5_coord_idx", 
            xlabel="i", 
            ylabel="U{i,1}",
            ylim=(-0.1,1)),
        hmap(K; xticks=:off, yticks=:off), layout=(1, 3),
        size=(900, 400)
    )
end
