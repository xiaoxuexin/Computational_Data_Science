
using Plots, Interact
using Flux, Flux.Data.MNIST ## this is the Julia package for deep learning 
using Flux: onehotbatch, argmax, crossentropy, throttle, mse
using Base.Iterators: repeated, partition
include("./deps.jl")

function generatedata_circle(r1,r2,N,σ=0.1)
    ϕ1 = linspace(0,2*π,N)
    ϕ2 = linspace(0,2*π,N)
    rx1 = r1.+σ*randn(N)
    rx2 = r2.+σ*randn(N)
    X1 = [rx1.*cos.(ϕ1) rx1.*sin.(ϕ1)]
    X2 =  [rx2.*cos.(ϕ2) rx2.*sin.(ϕ2)]
    return X1', X2'
end

X1c, X2c = generatedata_circle(2,0.5,100)
p1 = scatter(X1c[1,:],X1c[2,:], color = "red", label = "Class 1", aspectratio=:1.0)
scatter!(X2c[1,:],X2c[2,:], color = "blue", label = "Class 2",legend =:bottomright)

X = [X1c X2c]

Y = [ones(1,size(X1c,2)) -ones(1,size(X2c,2))]

loss_fn = mse
m = Chain(Dense(2, 1)) ## TODO: Replace ?? -- Hint: What should it be for the dataset? 
loss(x, y) = loss_fn(m(x), y) 
evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), 500)
opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.01))


function display_decision_boundaries(X1c,X2c,m,x1range,x2range,τ=0.0)
    D = [(m([x1;x2]).tracker.data)[1] for x2 = x2range, x1 = x1range] 
    heatmap(x1range,x2range,sign.(D.-τ),color=:grays) 
    scatter!(X1c[1,:],X1c[2,:], color = "red", label = "Class 1",aspectratio=:1.0)
    scatter!(X2c[1,:],X2c[2,:], color = "blue", label = "Class 2")
end


x1range = linspace(-3,3,100)
x2range = linspace(-3,3,100)
display_decision_boundaries(X1c,X2c,m,x1range,x2range)

@show params(m)[1];
@show params(m)[2];

loss_fn = mse
iters = 5000
m = Chain(Dense(2, 1,relu)) ## TODO: Replace ?? to have the relu activation function 
loss(x, y) = loss_fn(m(x), y) 
evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.1))


display_decision_boundaries(X1c,X2c,m,x1range,x2range)

loss_fn = mse
iters = 5000
m = Chain(Dense(2, 1,sigmoid)) ## TODO: Replace ?? 
loss(x, y) = loss_fn(m(x), y) 
evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.1))

display_decision_boundaries(X1c,X2c,m,x1range,x2range)

loss_fn = mse
n = 6  ## number of neurons in hidden layer 
iters = 10000
active_fun = σ # vs "relu"

m = Chain(Dense(2,n,active_fun),Dense(n,1)) ##TODO: Replace ?? 

loss(x, y) = loss_fn(m(x), y) ## replace with mse
evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))

lossXY = loss(X,Y).tracker.data
display_decision_boundaries(X1c,X2c,m,x1range,x2range)
plot!(title = "Loss = $(round(lossXY,5))")

loss_fn = mse
@manipulate for n = [4, 8, 16, 32], iters = [10, 100, 1000, 10000], active_fun = [σ,relu]
    m = Chain(Dense(2, n,active_fun),Dense(n,1)) ##TODO: Replace ?? with the same values you used above
    loss(x, y) = loss_fn(m(x), y) 
    evalcb = () -> @show([loss(X,Y)])
    dataset = Base.Iterators.repeated((X, Y), iters)
    opt = ADAM(params(m)) 
    Flux.train!(loss, dataset, opt)
    display_decision_boundaries(X1c,X2c,m,x1range,x2range)
    lossXY = loss(X,Y).tracker.data
    plot!(title = "n = $n, Loss = $(round(lossXY,5)), Iters = $iters")
end

loss_fn = mse
n = 3 ##TODO: Replace ??
iters = 10000

active_fun = σ # vs "relu"b
m = Chain(Dense(2,n,active_fun),Dense(n,n),Dense(n,1))  

loss(x, y) = loss_fn(m(x), y) ## replace with mse

evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) ## replace with SGD, Nesterov

Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))
lossXY = loss(X,Y).tracker.data
display_decision_boundaries(X1c,X2c,m,x1range,x2range)
plot!(title = "Loss = $(round(lossXY,5))")

loss_fn = mse
n = 3
iters = 10000
active_fun = σ # vs "relu"b

m = Chain(Dense(2, n,active_fun),Dense(n,n,active_fun),Dense(n,1)) ##TODO: Replace ?? 

loss(x, y) = loss_fn(m(x), y) ## replace with mse

evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) ## replace with SGD, Nesterov

Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))
lossXY = loss(X,Y).tracker.data
display_decision_boundaries(X1c,X2c,m,x1range,x2range)
plot!(title = "Loss = $(round(lossXY,5))")

loss_fn = mse
n = 3 ## TODO: Change till network learns 
iters = 10000
active_fun = relu

m = Chain(Dense(2, n,active_fun),Dense(n,n,active_fun),Dense(n,1,relu)) 
loss(x, y) = loss_fn(m(x), y) ## replace with mse

evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)
opt = ADAM(params(m)) ## replace with SGD, Nesterov

Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))
lossXY = loss(X,Y).tracker.data
display_decision_boundaries(X1c,X2c,m,x1range,x2range)
plot!(title = "Loss = $(round(lossXY,5))")

Y = [zeros(1,size(X1c,2)) ones(1,size(X2c,2))]

loss_fn = mse
n = 10
iters = 10000
active_fun = relu
m = Chain(Dense(2, n,active_fun),Dense(n,n,active_fun),Dense(n,1,active_fun)) ##TODO: Replace ?? 
loss(x, y) = loss_fn(m(x), y) 
evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)

opt = ADAM(params(m)) 

Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))
lossXY = loss(X,Y).tracker.data
display_decision_boundaries(X1c,X2c,m,2*x1range,2*x2range)
plot!(title = "Loss = $(round(lossXY,5))")


function generatedata_spiral(a,r,n)
    theta = linspace(0,4*pi,n)
    x = zeros(2,n)
    for i = 1:n
        x[1,i] = a*(theta[i]^(1/r))*sin(theta[i])
        x[2,i] = a*(theta[i]^(1/r))*cos(theta[i])
    end
    
    return x
end


X1c = generatedata_spiral(1,1,100)
X2c = generatedata_spiral(-1,1,100)
p1 = scatter(X1c[1,:],X1c[2,:], color = "red", label = "Class 1", aspectratio=:1.0)
scatter!(X2c[1,:],X2c[2,:], color = "blue", label = "Class 2",legend =:bottomright)

##TODO: Create training dataset
X = [X1c X2c]
Y = [1*ones(1,size(X1c,2)) 0*ones(1,size(X2c,2))] ## Fill in ?? Hint: Utilize insights from above
τ = 0.5 # decision boundary -- TODO: Fill in ?? based on how you populate Y and insights from earlier exercise
x1range = linspace(-20,20,100)
x2range = linspace(-20,20,100)

loss_fn = mse
n = 16
iters = 50000

active_fun = tanh

m = Chain(Dense(2,n,active_fun),Dense(n,n,active_fun), Dense(n,1,active_fun)) ##TODO: Replace ?? 

loss(x, y) = loss_fn(m(x), y) 

evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)

opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 0.5))
lossXY = loss(X,Y).tracker.data

display_decision_boundaries(X1c,X2c,m,x1range,x2range,τ)
plot!(title = "Loss = $(round(lossXY,5))")

active_fun = sigmoid # can also type σ and it would work 

##TODO: Use sane network topology that you used above to classify sprial -- 
##      Except change the activation function. Insert code below

m = Chain(Dense(2,n,active_fun),Dense(n,n,active_fun), Dense(n,1,active_fun))
loss(x, y) = loss_fn(m(x), y) 

evalcb = () -> @show([loss(X,Y)])
dataset = Base.Iterators.repeated((X, Y), iters)

opt = ADAM(params(m))
Flux.train!(loss, dataset, opt,cb = throttle(evalcb, 1))
lossXY = loss(X,Y).tracker.data

display_decision_boundaries(X1c,X2c,m,x1range,x2range,τ)
plot!(title = "Loss = $(round(lossXY,5))")

imgs = MNIST.images()
labels = MNIST.labels()
X = hcat(float.(reshape.(imgs, :))...) 

test_X = hcat(float.(reshape.(MNIST.images(:test), :))...) 
test_Y = onehotbatch(MNIST.labels(:test), 0:9);


@manipulate for sample_num = [1,2,3,4,5,6,7,8,9,10]
    imshow(reshape(X[:,sample_num],28,28),color=:grays, yticks =:none, xticks =:none, aspect_ratio=:equal)
    plot!(title = "Label =  $(labels[sample_num])")
end

Y  = onehotbatch(labels, 0:9) 
Y[:,1:10]

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

 evalcb = () -> @show([loss(X,Y), accuracy(test_X, test_Y)])

loss_fn = crossentropy

m = Chain(Dense(28^2, 10),softmax) ##TODO: Replace the ?? Hint: How many labels are there?
batches = 100

loss(x, y) = loss_fn(m(x), y) ## replace with mse
dataset = Base.Iterators.repeated((X, Y), batches)
opt = ADAM(params(m)) ## replace with SGD, Nesterov

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 5)) 
## it takes a while for output to appear -- be patient :) 

loss_fn = crossentropy

m = Chain(Dense(28 ^2, 40,relu),Dense(40,10),softmax) ##TODO: Replace the ?? Hint: How many labels are there?
batches = 500

loss(x, y) = loss_fn(m(x), y) ## replace with mse
dataset = Base.Iterators.repeated((X, Y), batches)
opt = ADAM(params(m)) ## replace with SGD, Nesterov

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

@manipulate for idx = [1,2,3,4,5,6,7,8,9,10]
imshow(reshape(params(m)[1].tracker.data[idx,:],28,28),color=:grays,aspect_ratio=:equal)
end

loss_fn = mse
m = Chain(Dense(28^2, 10),softmax) ##TODO: Replace ?? 

batches = 100
loss(x, y) = loss_fn(m(x), y) ## replace with mse
dataset = Base.Iterators.repeated((X, Y), batches)
opt = ADAM(params(m))
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

loss_fn = crossentropy
m = Chain(
         Dense(28^2, 20, relu), 
         Dense(20, 10, relu),
         softmax) ##TODO: Fill in ??

loss(x, y) = loss_fn(m(x), y) ## replace with mse
dataset = Base.Iterators.repeated((X, Y), 100)
opt = ADAM(params(m))
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))


loss_fn = crossentropy
batches = 500
m = Chain(
         Dense(28^2, 50, relu), 
         Dense(50,50,tanh),
         Dense(50,50,sigmoid),
         Dense(50,10,tanh),
          softmax)  ##TODO: Fill in ??
loss(x, y) = loss_fn(m(x), y) ## replace with mse

dataset = Base.Iterators.repeated((X, Y),batches)
opt = ADAM(params(m)) 
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

loss_fn = crossentropy
batches = 100
m = Chain(
         Dense(28^2, 32, relu), 
         Dense(32,160,relu),
         Dense(160, 10,relu),
         softmax)  
loss(x, y) = loss_fn(m(x), y) ## replace with mse

dataset = Base.Iterators.repeated((X, Y),batches)

opt = ADAM(params(m),0.01) 


Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

loss_fn = crossentropy
m = Chain(Dense(28^2, 10),softmax)
batches = 50
loss(x, y) = loss_fn(m(x), y) ## replace with mse
dataset = Base.Iterators.repeated((X, Y), batches)
opt = ADAM(params(m),1) 
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

