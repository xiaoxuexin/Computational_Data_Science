
A = ones(200, 400)
r = rank(A')

x = randn(200 - r)

Q = svd(A,thin=false)[3]

Q = Q[:, r + 1: 200]

x - Q * (Q' * x)

x = ones(2,3)
y = ones(3,3)
x[1,:]' * y[:,1]

using Plots; gr()
n = 100
x = randn(n); y = randn(n); A = x*y'
s = svdvals(A)
# @show rank(A)
# plot(s, marker = :circle, yscale = :log10, label= "",title = "singular values: log scale", xlabel="i",ylabel="sigma[i]")
@which rank(A)

sv = svdvals([9 9])
maximum(size([9 9]))*eps(sv[1])

a = 10

b = 10.0

typeof(a)

typeof(b)

c = "julia"
typeof(c)

convert(Int64,c)

c1 = "rocks"
c2 = c*c1


?string

string(c,c1)

a = "1"
b = parse(Int8,a)
typeof(b)

a = "1.0e-3"
b = parse(Float64,a)
@show  a # the @show macro display the value of the variable a
typeof(b)

c = "10.0"
d = "10"
e = parse(Float64, c*d)

a = 5 + im*3 # a = 5+3im # this also works 
typeof(a)


n = 5
x = randn(n)
@show typeof(randn(5,1))
randn(5,1)
# @show typeof(x)

@show typeof(x)

A = ones(2,3)
size(A,2)
length(A)
ndims(A)

@show A = rand(1.0:5.0,2,3)
@show B= A[1:2,1:2]
@show typeof(B)

@show [1:1:3]*ones(2)'
a = linspace(1,3,3)*ones(2)'
# b = ones(2)'*linspace(1,3,2)

@show A = rand(1.0:5.0,2,3)
@show A[:,1]
@show typeof(A[:,1])
@show A[:,[1]]
@show typeof(A[:,[1]])


@show A = reshape(rand(1.0:5.0,4*4),(4,4))
@show typeof(A)

@show A = reshape(rand(1.0:5.0,4*4),(4,4))

# @show A[:]
# @show typeof(A[:])
@show C = ones(3,4)
@show B = ones(2,4)
vcat(B,C)


for i = 1:5
    println(i)
end

a = randn(5,3,2)
size(a)

@show linspace(2,12,4)
collect(2.0:3.3333333333333335:12.0)
2:5:12

function f(x,y)
    return x+y
end

@show f(2,3)

function g(x,y)
    return x+y, x-y, x*y
end

a,b,c = g(2,3)
g(2,3)[[2]]

@show g(2,3)  
@show typeof(g(2,3))
@show g(2,3)[[1,3]]
@show typeof(g(2,3)[[1,2]])

function h(x,y,z)
    
    return x*y, y*z, x*z, x+y+z #TODO: fill in the ??'s 
end

A = [x*y- x+y for x= 1:5, y = 1:5]

function f1diff(i,j,n)
    fij = 0
    if i == n && j == 1
        fij = 1
    elseif i == j
        fij = -1
    elseif j-i == 1
        fij = 1
    
        
         # Hint: The command "if i == 5 && j == 5" checks if i equals 10 and j equals 5 .    
    end
          
    return fij
end

n = 4
A = [f1diff(i,j,n) for i = 1 : n, j = 1 : n]

@show A 
@show mapslices(maximum,A,1)
@show typeof(mapslices(maximum,A,1))

@show X = rand(1:4,3,3)
@show Y = (X .== 1)
@show sum(Y)
vec(Y)


using Plots
plotly()

x = linspace(0,2*pi,100)
y = sin.(x)
# plot(x,y,label="sin",xlabel="x",ylabel="y(x)")
@show typeof(y)
collect(y)

x = linspace(0,2*pi,100)
y = sin.(x)
scatter(x,y,label="sin",xlabel="x",ylabel="y(x)")

x = linspace(0,2*pi,50)
y = sin.(x)
plot(x,y,label="sin",xlabel="x",ylabel="y(x)",linewidth=7,color="blue")
scatter!(x,y,label="data",xlabel="x",ylabel="y(x)",markersize=5,marker=:circle,color="black")

using Interact

x = linspace(0,2*pi,100)
@manipulate for f in [sin,cos] # TODO: modify this to [sin,cos,tan] 
    y = f.(x)
    scatter(x,y,label="$f",xlabel="x",ylabel="$(f)(x)")
end


n = 4 
A = [f1diff(i,j,n) for i = 1 : n, j = 1 : n]
heatmap(A)

@manipulate for n = [4, 10, 30]
A = [f1diff(i,j,n) for i = 1 : n, j = 1 : n]
heatmap(A)
end


@manipulate for n = [4, 10, 30]
A = [f1diff(i,j,n) for i = 1 : n, j = 1 : n]
heatmap(A,yflip = false, aspect_ratio=1.0,) # aspect_ratio = 1.0 makes the x and y axes equal  
end
