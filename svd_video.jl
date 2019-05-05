
using Colors, Interact, Plots
gr()

# deps.jl containts some plotting utilites and an important
# workaround for yflip=:true not working in GR v17.2 
# You can inspect the file deps.jl to learn more
include("deps.jl"); 

A = [ (1 <= j <= 10) || (25 <= j <= 35) ? 1 : 0  
    for i=1:100, j=1:100 ]
displaymatrix(A)

A = [ (1 <= j <= 10) || (25 <= j <= 35) ||  
    (40 <= i <= 60 && 50 <= j <= 70) ?  1 : 0  
    for i = 1:100, j=1:100 ]
displaymatrix(A)

x1 = ones(100)
y1 = [(25 <= j <= 35) || (1 <= j <= 10) ? 1 : 0 for j=1:100]
x2 = [(40 <= i <= 60) ? 1 : 0 for i=1:100]
y2 = [50 <= j <= 70 ? 1 : 0 for j=1:100]
Aalt = x1*y1' + x2*y2'
displaymatrix(Aalt)

vecnorm(A-Aalt)

A = [(1 <= j <= 10) || (25 <= j <= 35) ||
    (40 <= i <= 60 && 50 <= j <= 70) || 
    (80 <= j <= 90) ? 1 : 0  
    for i=1:100, j=1:100]
displaymatrix(A)

x = rand(1:5, 100)
y = rand(1, 100)
A = [ y[j]*x[i] for i=1:100, j=1:10 ]
displaymatrix(A)
size(A)

X = rand(1:5, 10, 10)
B = repmat(X, 1, 10)
displaymatrix(B)


A = repmat(x[:], 1, 10)
displaymatrix(A)

# this the package we can use to read MATLAB formated .mat files
using MAT
vars = matread("lobby.mat") 

# TODO: Fill in the ?? with the appropriate variable name from 
# the Dictionary
varname = "MovMat"
file = matopen("lobby.mat")
MovMat = read(file, varname) 
close(file)
@show size(MovMat)
@show typeof(MovMat)

# Convert to [0, 1] floating point, if necessary
MovMat = float(MovMat)
MovMat -= minimum(MovMat)
MovMat /= maximum(MovMat)
# Display first frame
Gray.(MovMat[:, :, 1])

@manipulate for frame = 100 
    plotframe(
        bg(MovMat[:, :, frame]),
        title = "Frame $frame of $(size(MovMat,3))",
    )
end

# Enter values in the box below to explore the video

# this is a package built & tested for bookalive that 
# cannot be added locally (for now)
using WebPlayer

playvideo([bg(MovMat)], "Original movie", 
    frames_per_second=120, width=200)
# move the slider to scan through the movie

"""
    MovMatk, MovMatResidual, Vk, sk = svdize_moviematrix(MovMat, k)
Inputs:
* `MovMat` is an m x n x numFrames array
* `k` is an integer
Outputs:
* `MovMatk` is an m x n x numFrames 3-D array
* `MovMatResidual` is an m x n x numFrames 3-D array
* `sk` is a length k vector

Given a movie `MovMat` formatted as a 3-D array and a rank `k`,
return a (rank `k`) truncated SVD version of the movie as `MovMatk`,
the resiudal as `MovMatResidual`, the right singular vectors as `Vk`,
and the singular values as `sk`.

*Hint: svds has a keyword argument `nsv` for the number of singular values.
To specify this argument, you use the keyword: `svds(..., nsv=...)`*
"""
function svdize_moviematrix(MovMat, k)
        # Reshape 3-D arrays into matrix where each column is a frame
        m, n, numFrames = size(MovMat) # MovMat is an m x n x numFrames 3-D array
        MovMatVec = reshape(MovMat, m*n, numFrames) # Convert it to a mn x numFrames matrix

        # Compute rank k truncated SVD and residual
        UsV = svds(MovMatVec,nsv= k)[1]
        Uk = UsV[:U]
        sk = UsV[:S]
        Vk = UsV[:V]

        MovMatk = Uk*Diagonal(sk)*Vk'
        MovMatResidual = MovMatVec - MovMatk

        # Reshape back into 3-D m x n x numFrames arrays
        MovMatk = reshape(MovMatk, m, n, numFrames)
        MovMatResidual = reshape(MovMatResidual, m, n, numFrames)

        return MovMatk, MovMatResidual, sk
end


# Desired rank
k = 1
# Compute truncated SVD-version of movie
MovMatk, MovMatResidual,  _ = svdize_moviematrix(MovMat, k)
m, n, num_frames = size(MovMat)

@manipulate for frame=325
    plotframe(
        bg(MovMatk[:, :, frame]),
        title = "Rank $k approx: Frame $frame of $(size(MovMat,3))",
    )
end

# Enter values in the box below to explore the video

playvideo([bg(MovMat), bg(MovMatk)], 
    ["Original movie","Rank $k approximation"], 
    frames_per_second=60, width=400)


playvideo([bg(MovMatk), fg(MovMatResidual), bg(MovMat)], 
    ["Rank $k approximation","Residual","Original Movie"], 
    frames_per_second=60, width=200)

# An interesting frame to compare
frame = 250

# Original movie
plotframe(
    bg(MovMat[:, :, frame]),
    title = "Original movie: frame = $frame",
)

# Rank k approx
plotframe(
    bg(MovMatk[:, :, frame]),
    title = "Rank $k approx: frame = $frame",
)

# Rank k residual
plotframe(
    fg(MovMatResidual[:, :, frame]),
    title = "Rank $k residual: frame = $frame",
)

# Largest right singular vector
UsV = svds(reshape(MovMat, :, 650), nsv=1)[1]
V = UsV[:V]
plot(
    V[:, 1],
    linestyle=:solid,
    marker=:circle, # :circle
    label="V[:, 1]",
    title="Largest singular vector of movie matrix",
    legend=:bottomleft,
    label=""
)


k = 1
@manipulate for frame = [100, 194, 397, 450]    
    # Plot first right singular vector
    p1 = plot(
        V,
        linestyle=:solid,
        title="First right singular vector",
        label=""
    )
    
    # Denote current frame
    plot!(
        (frame, V[frame, 1]),
        marker=:circle, # current point
        color=:black,
        label="Current frame"
    )

    # Plot movie frames
    p2 = plotframe(
        joinframes(
            frame,
            bg(MovMat),
            bg(MovMatk),
            fg(MovMatResidual),
        ),
        title = "Frame = $frame",
    )
    
    plot(p1, p2,
        layout = Plots.grid(2, 1, heights=[0.5, 0.5]),
    )
end

# Change frame to view different frames

@manipulate for k=[1, 2], frame=[100, 194, 397, 450] 
    # Decompose movie
    _backk, _resk = svdize_moviematrix(MovMat, k)
    UsV = svds(reshape(MovMat, :, 650), nsv=k)[1]
    _Vk = UsV[:V]

    # Plot right singular vector(s)
    p1 = plot(
        _Vk,
        linestyle=:solid,
        title="Leading right singular vector(s)",
        label=["V[:, 1]", "V[:, 2]"]
    )
    
    # Denote current frame
    for kk = 1:k
        plot!(
            (frame, _Vk[frame, kk]),
            marker=:circle, # current point
            color=:black,
            label=""
        )
    end
    
    # Plot movie frames
    p2 = plotframe(
        joinframes(
            frame,
            bg(MovMat),
            bg(_backk),
            fg(_resk),
        ),
        title="Frame = $frame",
    )
    
    plot(p1, p2,
        layout = Plots.grid(2, 1, heights=[0.5, 0.5]),
    )
end

# Change k to use a different rank approximation
# Change frame to view different frames

# Choose rank
k = 2

# Decompose movie
MovMatk, MovMatResidual = svdize_moviematrix(MovMat, k);
UsV = svds(reshape(MovMat,:,650), nsv=k)[1]
V = UsV[:V];

# An interesting frame
frame = 250

# Original movie
plotframe(
    bg(MovMat[:, :, frame]),
    title="Original movie: frame $frame",
)

# Rank k approx
plotframe(
    bg(MovMatk[:, :, frame]),
    title="Rank $k approx: frame $frame",
)

# Rank k residual
plotframe(
    fg(MovMatResidual[:, :, frame]),
    title="Rank $k residual: frame $frame",
)

println("Original movie vs Rank $k approximation vs Residual")
playvideo(
    [bg(MovMat), bg(MovMatk), fg(MovMatResidual)],
    ["Original movie", "Rank $k approximation", "Residual"]
)

plot(
    V,
    label=["V[:, $i]" for i in collect(1:k)'],
    title="$k leading right singular vectors of movie matrix", 
    xaxis="Frame",
    marker=:circle, 
)

@manipulate for frame = [150, 196, 300, 420, 500] 
    plotframe(
        bg(MovMat[:, :, frame]),
        title = "Original movie: frame = $frame",
    )
end

# Change frame to view different frames

## TODO: Your code here comparing videos of the k = 2 vs k = 3  reduced rank videos.
# Desired rank
k2 = 2
k3 = 3
# Compute truncated SVD-version of movie
MovMatk2, MovMatResidual2,  _ = svdize_moviematrix(MovMat, k2)
MovMatk3, MovMatResidual3,  _ = svdize_moviematrix(MovMat, k3)
m, n, num_frames = size(MovMat)
playvideo([bg(MovMatk2), bg(MovMatk3)], 
    ["Rank $k2 approximation","Rank $k3 approximation"], 
    frames_per_second=60, width=200)

@manipulate for frame = [100, 150, 300, 420, 500] 
    _back2, _res2 = svdize_moviematrix(MovMat, 2)
    _back3, _res3 = svdize_moviematrix(MovMat, 3)

    p2 = plotframe(
        joinframes(
            frame,
            bg(_back2),
            bg(_back3),
        ),
        title="k=2 v.s. k=3 for Frame = $frame",

    )
        
end

k = 3;MovMatk, MovMatResidual = svdize_moviematrix(MovMat, k);
UsV = svds(reshape(MovMat,:,650), nsv=k)[1]
V = UsV[:V];
plot(
    V,
    label=["V[:, $i]" for i in collect(1:k)'],
    title="$k leading right singular vectors of movie matrix", 
    xaxis="Frame",
    marker=:circle, 
)

## TODO: Your code here displaying what happends in the video at time instances corresponding 
##       to when something there are (say) jumps or anything interesting in the  third right singular vector
# Decompose movie
    k = 3
    _backk, _resk = svdize_moviematrix(MovMat, k)
    UsV = svds(reshape(MovMat, :, 650), nsv=k)[1]
    _Vk = UsV[:V]

@manipulate for frame=[100, 194, 397, 450] 
    
    # Plot right singular vector(s)
    p1 = plot(
        _Vk,
        linestyle=:solid,
        title="Leading right singular vector(s)",
        label=["V[:, 1]", "V[:, 2]", "V[:, 3]"]
    )
    
    # Denote current frame
    for kk = 1:k
        plot!(
            (frame, _Vk[frame, kk]),
            marker=:circle, # current point
            color=:black,
            label=""
        )
    end
    
    # Plot movie frames
    p2 = plotframe(
        joinframes(
            frame,
            bg(MovMat),
            bg(_backk),
            fg(_resk),
        ),
        title="Frame = $frame",
    )
    
    plot(p1, p2,
        layout = Plots.grid(2, 1, heights=[0.5, 0.5]),
    )
end

# Change k to use a different rank approximation
# Change frame to view different frames

##TODO: Plot the singular value of the reshaped video matrix. Tip: Use the "scatter" and the "svdvals" commands. 
m, n, numFrames = size(MovMat)
MovMatVec = reshape(MovMat, m*n, numFrames) # Convert it to a mn x numFrames matrix
matr = svdvals(MovMatVec)
scatter(matr)

