
using Plots, Interact
include("deps.jl") #workaround for bug in GR package where yflip=true in heatmap does not work 

gr(;
    :color=>"black",
    :markerstrokecolor=>"white", 
    :markerstrokewidth=>0,
    :alpha=>0.6,
    :label=>""
)

function g(X, W, b, f_a)
    # X is a d x N Array{Float64,2} -- even when n or d equal 1
    # W is an n x d Array{Float64,2} 
    # b is n element Array{Float64,1}
    return sum(f_a.(W*X .+ b),1) # type Array{Float64,2}
end

function grad_loss_1layer_1output(f_a, df_a, x, y, W, b)          
    # x is a d x N Array{Float64,2} -- even when d= 1
    # y is a 1 x N element Array{Float64,2}
    # W is an n x d Array{Float64,2} 
    # b is n element Array{Float64,1}
    
    n, d = size(W) 
    N = size(x, 2)
    
    dW = zeros(W)
    db = zeros(b)
    loss = 0.0
    
    for k = 1:N
        error = (y[k] - sum(f_a.(W*x[:,k] + b)))
        for p = 1:n
            for q = 1:d
                # TODO: Fill in the ??
                dW[p,q] = dW[p,q] - 2/N*error*df_a(W[p,:]'*x[:,k] + b[p])*x[q,k]
            end
                #TODO: Fill in the ??
            db[p] = db[p] - 2/N*error*df_a(W[p,:]'*x[:,k] + b[p])
        end
        
        ## TODO: Fill in the ??
        loss =  loss + (1/N)*error^2
    end
    return dW, db, loss
end
        

function learn2classify_sgd_1layer(f_a, df_a, grad_loss, x, y, W0, b0,
        mu=1e-3, iters=500, batch_size=10)

    # x is a d x N Array{Float64,2} -- even when = 1
    # y is a 1 x N element Array{Float64,2}
    # W is an n x d Array{Float64,2} 
    # b is n element Array{Float64,1}
    
    n, d = size(W0) #number of inputs
    N = size(x, 2) # number of training samples
 
    W = W0
    b = b0
    
    loss = zeros(iters)
    for i = 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]
        
        dW, db, loss_i = grad_loss(f_a, df_a, x[:,batch_idx], y[:,batch_idx], W, b)
        
        W = W - mu*dW
        b = b - mu*db

        loss[i] = loss_i
    end

    return W, b, loss
end


function learn2classify_asgd_1layer(
        f_a, df_a, grad_loss,x, y, W0, b0,
        mu=1e-3, iters=500, batch_size=10
    )

    d = size(W0, 2) #number of inputs
    n = size(W0, 1) # number of neurons
    N = size(x, 2) # number of training samples
 
    W = W0
    b = b0
    
    loss = zeros(iters)


    lambdak = 0
    qk = W
    pk = b
    for i = 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]
        
        dW, db, loss_i = grad_loss(f_a, df_a, x[:,batch_idx], y[:,batch_idx], W, b)
        
        qkp1 = W - mu*dW
        pkp1 = b - mu*db

        lambdakp1 = (1 + sqrt(1 + 4*lambdak^2))/2
        gammak = (1 - lambdak)/lambdakp1

        W = (1 - gammak)*qkp1 + gammak*qk
        b = (1 - gammak)*pkp1 + gammak*pk

        qk = qkp1
        pk = pkp1
        lambdak = lambdakp1

        loss[i] = loss_i
    end

    return W,b, loss
end


function dtanh(z)
    return 1-tanh(z)^2 ##TODO: fill ?? which is dtanh(z)/z
 end 

linear(z) = z

function dlinear(z) 
     return 1  ##TODO: fill ?? which is dlinear(z)/z
end

N = 200
x = convert(Array{Float64,2}, collect(linspace(-2, 2, N))')
y = abs.(x)
x

f_a, df_a = tanh, dtanh
n, d = 8, 1
iters, batch_size = 5000, 20
W0, b0  = randn(n,d), randn(n)
mu = 1e-2 # make lower till network learns
@time what,bhat, empirical_loss = learn2classify_sgd_1layer(f_a,df_a,grad_loss_1layer_1output,x,y,W0,b0,mu,iters,batch_size)
plot_idx = 1:5:iters
scatter(plot_idx,empirical_loss[plot_idx],yscale=:log10,ylabel="training loss",xlabel="iterations",label="")

yhat = g(x, what, bhat, f_a) ##TDO: Fill in ??
@show loss = mean((yhat - y).^2)
scatter(x[:], yhat[:], 
    label="training data", 
    xlabel="x", 
    ylabel="value", 
    title="SGD"
)
scatter!(x[:], y[:], color=:red, label="predictions")

n, d = 16, 1
W0, b0  = randn(n,d), randn(n)
@time what,bhat, empirical_loss = learn2classify_sgd_1layer(f_a,df_a,grad_loss_1layer_1output,x,y,W0,b0,mu,iters,batch_size)
plot_idx = 1:5:iters
scatter(plot_idx,empirical_loss[plot_idx],yscale=:log10,ylabel="training loss",xlabel="iterations",label="")

yhat = g(x,what,bhat,f_a) ##TDO: Fill in ??
@show loss = mean((yhat-y).^2)
scatter(x[:],yhat[:],label="training data",xlabel="x",ylabel="value",title="SGD")
scatter!(x[:],y[:],color=:red,label="predictions")

@time what, bhat, empirical_loss = learn2classify_asgd_1layer(
        f_a, df_a, grad_loss_1layer_1output,
        x, y, W0, b0, mu, iters, batch_size
)
yhat = g(x, what, bhat, f_a)
@show loss = mean((yhat - y).^2)
scatter(empirical_loss[1:5:end], scale=:log10)

@show loss = mean((yhat - y).^2)
scatter(x[:], yhat[:], 
    label="training data",
    xlabel="x",
    ylabel="value",
    title="ASGD"
)
scatter!(x[:], y[:], color=:red, label="predictions")

function generatedata(μ1, μ2, Σ1, Σ2, N)
    d = size(μ1, 1)
    X1 = μ1 .+ sqrtm(Σ1)*randn(d, N)
    X2 = μ2 .+ sqrtm(Σ2)* randn(d, N)
    return X1, X2
end   

X1, X2 = generatedata([1;0], [-1;0], Diagonal([0.01,1]), Diagonal([0.01,1]), 100)
p1 = scatter(X1[1,:], X1[2,:], color="red", label="Class 1", title="", legend=:bottom)
scatter!(p1, X2[1,:], X2[2,:], color="blue", label="Class 2")
plot(p1)

xtest = [0.55, 0.5] ## test vector
p1 = scatter(X1[1,:], X1[2,:], color="red", label="Class 1", title ="", legend=:bottom)
scatter!(p1,X2[1,:], X2[2,:], color="blue", label="Class 2")
scatter!(p1,xtest[1,:],xtest[2,:], color = "cyan", marker =:square, label = "xtest = point to be classified")
plot(p1)

x1range = linspace(-5,5,100)
x2range = linspace(-5,5,100)
@manipulate for w₁ = -1:0.5:1,
                w₂ = -1:0.5:1,
                b = -1:0.25:1,
                show_sign_only = true 
                
                  D =( [x1*w₁ + x2*w₂ + b for x2 = x2range, x1 = x1range])
                 if show_sign_only
                    D = sign.(D)
                 end
    
                 heatmap(x1range,x2range,D,title="",color=:grays)
                 scatter!(X1[1,:], X1[2,:], color="red", label="class 1")
                 scatter!(X2[1,:], X2[2,:], color="blue", label="class 2")
                 scatter!(xtest[1,:], xtest[2,:], color="cyan", marker=:square, label="x = point to be classified",legend=:bottomleft)
                 
end

trainX = hcat(X1,X2)
class_vector = [-ones(1,size(X1,2)) ones(1,size(X2,2))]
n, d = 1, 2
mu = 0.0001 ## reduce this till network learns
W0 = randn(n,d)
b0 = rand(1)
what,bhat, empirical_loss = learn2classify_asgd_1layer(linear,dlinear,grad_loss_1layer_1output,trainX,class_vector,W0,b0,mu,iters,batch_size)
plot_idx = 1:5:iters
scatter(plot_idx,empirical_loss[plot_idx],yscale=:log10,ylabel="training loss",xlabel="iterations",label="")

D = [ sign.((x1*what[1]+x2*what[2]+bhat)[1]) for x2 = x2range, x1 = x1range]
heatmap(x1range,x2range,D,title="",color=:grays)
scatter!(X1[1,:],X1[2,:], color = "red", label = "class 1")
scatter!(X2[1,:],X2[2,:], color = "blue", label = "class 2")
scatter!(xtest[1,:],xtest[2,:], color = "cyan", marker =:square, label = "x = point to be classified",legend=:bottomleft)

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

trainXc = hcat(X1c,X2c)
n,d = 8, 2  ##TODO: d = ?
W0, b0 = rand(n,d), rand(n)
f_a, df_a = tanh, dtanh
mu = 0.0001 ## make smaller till network learns
what,bhat, empirical_loss = learn2classify_asgd_1layer(f_a,df_a,grad_loss_1layer_1output,trainXc,class_vector,W0,b0,mu,iters,batch_size)
plot_idx = 1:5:iters
scatter(plot_idx,empirical_loss[plot_idx],yscale=:log10,ylabel="training loss",xlabel="iterations",label="")

D = [g([x1;x2],what,bhat,f_a)[1] for x2 = x2range, x1 = x1range] 
heatmap(x1range,x2range,sign.(D),color=:grays) 
scatter!(X1c[1,:],X1c[2,:], color = "red", label = "Class 1",aspectratio=:1.0)
scatter!(X2c[1,:],X2c[2,:], color = "blue", label = "Class 2")

n,d = 16, 2
W0, b0 = rand(n,d), rand(n)
f_a, df_a = tanh, dtanh
what,bhat, empirical_loss = learn2classify_asgd_1layer(f_a,df_a,grad_loss_1layer_1output,trainXc,class_vector,W0,b0,mu,iters,batch_size)
plot_idx = 1:5:iters
scatter(plot_idx,empirical_loss[plot_idx],yscale=:log10,ylabel="training loss",xlabel="iterations",label="")

D = [g([x1;x2],what,bhat,f_a)[1] for x2 = x2range, x1 = x1range] ##TODO fill in ??
heatmap(x1range,x2range,sign.(D),color=:grays) 
scatter!(X1c[1,:],X1c[2,:], color = "red", label = "Class 1",aspectratio=:1.0)
scatter!(X2c[1,:],X2c[2,:], color = "blue", label = "Class 2")

function grad_loss_1layer(f_a,df_a,x,y,W,b)

    # x is a d x N Array{Float64,2} -- even when d = 1
    # y is a n x N  Array{Float64,2}
    # W is an n x d Array{Float64,2} 
    # b is n element Array{Float64,1}
    
    n, d = size(W) ##TODO: ?? & quiz
    N = size(y,2) ## assume y is matrix of size n x N
    
    dW = zeros(W) 
    db = zeros(b)
    loss = 0.0
    
        
    for k = 1 : N
        for p = 1 : n
            error = y[p,k] - f_a(W[p,:]'*x[:,k]+b[p])
            common_term = error* df_a(W[p,:]'*x[:,k]+b[p])
            for q = 1 : d
                # TODO: Fill in the ??
                dW[p,q] = dW[p,q] - 2/N*common_term*x[q,k]
            end
            # TODO: Fill in ??
            db[p] = db[p] - 2/N*common_term
            # TODO: Fill in the ??
            loss = loss + 1/N*error^2
        end
    end
    
    return dW, db, loss
end        

function load_digit_data(digit,nx=28,ny=28,nrep=1000)
    file = "data"*string(digit)
    
    fp = open(file, "r")
    x = read(fp, UInt8, (nx,ny,nrep)) # what is the type of x0?
    close(fp)
    
   return x
end

digits = [7, 8, 3]
num_digits = length(digits)
x0 = load_digit_data(digits[1])
x1 = load_digit_data(digits[2])
x2 = load_digit_data(digits[3]);


class_label_vector = collect(1:num_digits)
class_encoding_matrix = eye(num_digits,num_digits)
digit2class = Dict(string(digits[i]) => class_label_vector[i] for i = 1 : length(digits) )
class2digit = Dict(value=> key for (key,value) in digit2class)
class2digit

xtrain = hcat(reshape(x0[:,:,1:500],784,:),reshape(x1[:,:,1:500],784,:),reshape(x2[:,:,1:500],784,:))
xtest =   hcat(reshape(x0[:,:,501:1000],784,:),reshape(x1[:,:,501:1000],784,:),reshape(x2[:,:,501:1000],784,:))
test_labels = kron(1:num_digits,ones(500))
yvector = hcat(kron(ones(500)',class_encoding_matrix[:,1]),kron(ones(500)',class_encoding_matrix[:,2]),kron(ones(500)',class_encoding_matrix[:,3]))
yvector


function fun(mu)    
    pcorrect = zeros(1,9)    
    x0 = load_digit_data(0)
    xtrain = reshape(x0[:,:,1:500],784,:)
    xtest = reshape(x0[:,:,501:1000],784,:)
    for i in 2:10
        digits = 1:i
        num_digits = length(digits)
        x = load_digit_data(digits[i-1])
        class_label_vector = collect(1:num_digits)
        class_encoding_matrix = eye(num_digits,num_digits)
        digit2class = Dict(string(digits[i]) => class_label_vector[i] for i = 1 : length(digits) )
        class2digit = Dict(value=> key for (key,value) in digit2class)
        xtrain = hcat(xtrain, reshape(x[:,:,1:500],784,:))
        xtest = hcat(xtest, reshape(x[:,:,501:1000],784,:))
        test_labels = kron(1:num_digits, ones(500))
        yvector = kron(ones(500)',class_encoding_matrix[:,1])
        for j in 2:i
            yvector = hcat(yvector,kron(ones(500)',class_encoding_matrix[:,j]))
        end
        n = num_digits
        d = 784
        W0 = zeros(n,d)
        b0 = zeros(n)
        iters = 2000
        wtanh, btanh, loss_tanh = learn2classify_asgd_1layer(tanh,dtanh,grad_loss_1layer,xtrain,yvector,W0,b0,mu,iters)
        Ytanh = tanh.(wtanh*xtest .+ btanh)
        class_predictions_tanh = vec(mapslices(indmax,Ytanh,1))
        pcorrect[i-1] = sum(class_predictions_tanh .== test_labels)/length(test_labels)
    end
    return pcorrect
end

scatter(2:1:10,fun(1e-9)', color = "red", xlabel = "num of class", ylabel = "correct prob", label = "mu = 1e-9")
scatter!(2:1:10,fun(1e-10)', color = "green", xlabel = "num of class", ylabel = "correct prob", label = "mu = 1e-10")
scatter!(2:1:10,fun(1e-11)', color = "blue", xlabel = "num of class", ylabel = "correct prob", label = "mu = 1e-11")

n = num_digits
d = 784
@show mu = 1e-12 ## make lower if it does notlearn  
W0 = zeros(n,d) ## why did we initlalize with @0
b0 = zeros(n)
iters = 2000 
wlinear, blinear, loss_linear = learn2classify_asgd_1layer(linear,dlinear,grad_loss_1layer,xtrain,yvector,W0,b0,mu,iters)
scatter(1:20:iters,loss_linear[1:20:end],yscale=:log10,label="",xlabel="iterations",ylabel="training loss")

function disp_vector_as_img(img_vector,plot_title,nx=28,ny=28)
    showimg(vcat(mapslices(v -> [reshape(v,nx,ny)],img_vector,2)...)', title=plot_title,color=:grays,aspectratio=1.0)
end

disp_vector_as_img(wlinear,"weight vectors viewed as matries for linear") 

Ylinear = linear.(wlinear*xtest .+ blinear)
class_predictions_linear = vec(mapslices(indmax,Ylinear,1)) 
@show pcorrect = sum(class_predictions_linear .== test_labels)/length(test_labels)
scatter(class_predictions_linear,label="predicted label",title="linear")
scatter!(test_labels,color="red",label="correct label")


wtanh, btanh, loss_tanh = learn2classify_asgd_1layer(tanh,dtanh,grad_loss_1layer,xtrain,yvector,W0,b0,mu,iters)
scatter(1:20:iters,loss_tanh[1:20:end],color=:red,yscale=:log10,label="",xlabel="iterations",ylabel="training loss")

Ytanh = tanh.(wtanh*xtest .+ btanh)
class_predictions_tanh = vec(mapslices(indmax,Ytanh,1))
@show pcorrect = sum(class_predictions_tanh .== test_labels)/length(test_labels)
scatter(class_predictions_tanh,label="predicted label",title="tanh")
scatter!(test_labels,color="red",label="correct label")


disp_vector_as_img(wtanh,"weight vectors viewed as images for tanh") 

using WebWidgets, Blink, Colors, InteractNext, Base.Test

function predict_digit(my_img_from_app,class2digit,f_a,w,b)
    my_img_from_app = Gray.(my_img_from_app[])
    my_img_from_app = float(my_img_from_app)
    my_img = 255*(my_img_from_app');
    my_img_vector = (my_img[:])
    class_prediction = mapslices(indmax,f_a.(w*my_img_vector+b),1) ##TODO fill in ??
    my_digit_prediction = class2digit[class_prediction[]]
    return my_img, my_digit_prediction, f_a.(w*my_img_vector+b)
end

app, my_img_from_app = drawnumber(brushsize = 14, resolution = (250,250))
app

my_img, my_digit_prediction, vector = predict_digit(my_img_from_app,class2digit,tanh,wtanh,btanh) 
@show vector
showimg(my_img,
    title = "Network predicts $my_digit_prediction",
    transpose =:true,
    color=:grays,
    aspectratio=1.0)

##TDO: Your code for computing Pcorrect over all digits
digits = 0:1:9
num_digits = length(digits)
x0 = load_digit_data(digits[1])
x1 = load_digit_data(digits[2])
x2 = load_digit_data(digits[3])
x3 = load_digit_data(digits[4])
x4 = load_digit_data(digits[5])
x5 = load_digit_data(digits[6])
x6 = load_digit_data(digits[7])
x7 = load_digit_data(digits[8])
x8 = load_digit_data(digits[9])
x9 = load_digit_data(digits[10])
class_label_vector = collect(1:num_digits)
class_encoding_matrix = eye(num_digits,num_digits)
digit2class = Dict(string(digits[i]) => class_label_vector[i] for i = 1 : length(digits) )
class2digit = Dict(value=> key for (key,value) in digit2class)
xtrain = hcat(reshape(x0[:,:,1:500],784,:),reshape(x1[:,:,1:500],784,:),reshape(x2[:,:,1:500],784,:),reshape(x3[:,:,1:500],784,:),reshape(x4[:,:,1:500],784,:),reshape(x5[:,:,1:500],784,:),reshape(x6[:,:,1:500],784,:),reshape(x7[:,:,1:500],784,:),reshape(x8[:,:,1:500],784,:),reshape(x9[:,:,1:500],784,:))
xtest = hcat(reshape(x0[:,:,501:1000],784,:),reshape(x1[:,:,501:1000],784,:),reshape(x2[:,:,501:1000],784,:),reshape(x3[:,:,501:1000],784,:),reshape(x4[:,:,501:1000],784,:),reshape(x5[:,:,501:1000],784,:),reshape(x6[:,:,501:1000],784,:),reshape(x7[:,:,501:1000],784,:),reshape(x8[:,:,501:1000],784,:),reshape(x9[:,:,501:1000],784,:))
test_labels = kron(1:num_digits,ones(500))
yvector = hcat(kron(ones(500)',class_encoding_matrix[:,1]),kron(ones(500)',class_encoding_matrix[:,2]),kron(ones(500)',class_encoding_matrix[:,3]),kron(ones(500)',class_encoding_matrix[:,4]),kron(ones(500)',class_encoding_matrix[:,5]),kron(ones(500)',class_encoding_matrix[:,6]),kron(ones(500)',class_encoding_matrix[:,7]),kron(ones(500)',class_encoding_matrix[:,8]),kron(ones(500)',class_encoding_matrix[:,9]),kron(ones(500)',class_encoding_matrix[:,10]))
n = num_digits
d = 784
@show mu = 1e-12 ## make lower if it does notlearn  
W0 = zeros(n,d) ## why did we initlalize with @0
b0 = zeros(n)
iters = 2000 
wlinear, blinear, loss_linear = learn2classify_asgd_1layer(linear,dlinear,grad_loss_1layer,xtrain,yvector,W0,b0,mu,iters)
scatter(1:20:iters,loss_linear[1:20:end],yscale=:log10,label="",xlabel="iterations",ylabel="training loss")
Ylinear = linear.(wlinear*xtest .+ blinear)
class_predictions_linear = vec(mapslices(indmax,Ylinear,1)) 
@show pcorrect = sum(class_predictions_linear .== test_labels)/length(test_labels)
# scatter(class_predictions_linear,label="predicted label",title="linear")
# scatter!(test_labels,color="red",label="correct label")

digits = 0:1:9
num_digits = length(digits)
x0 = load_digit_data(digits[1])
x1 = load_digit_data(digits[2])
x2 = load_digit_data(digits[3])
x3 = load_digit_data(digits[4])
x4 = load_digit_data(digits[5])
x5 = load_digit_data(digits[6])
x6 = load_digit_data(digits[7])
x7 = load_digit_data(digits[8])
x8 = load_digit_data(digits[9])
x9 = load_digit_data(digits[10])
class_label_vector = collect(1:num_digits)
class_encoding_matrix = eye(num_digits,num_digits)
digit2class = Dict(string(digits[i]) => class_label_vector[i] for i = 1 : length(digits) )
class2digit = Dict(value=> key for (key,value) in digit2class)
xtrain = hcat(reshape(x0[:,:,1:500],784,:),reshape(x1[:,:,1:500],784,:),reshape(x2[:,:,1:500],784,:),reshape(x3[:,:,1:500],784,:),reshape(x4[:,:,1:500],784,:),reshape(x5[:,:,1:500],784,:),reshape(x6[:,:,1:500],784,:),reshape(x7[:,:,1:500],784,:),reshape(x8[:,:,1:500],784,:),reshape(x9[:,:,1:500],784,:))
xtest =   hcat(reshape(x0[:,:,501:1000],784,:),reshape(x1[:,:,501:1000],784,:),reshape(x2[:,:,501:1000],784,:),reshape(x3[:,:,501:1000],784,:),reshape(x4[:,:,501:1000],784,:),reshape(x5[:,:,501:1000],784,:),reshape(x6[:,:,501:1000],784,:),reshape(x7[:,:,501:1000],784,:),reshape(x8[:,:,501:1000],784,:),reshape(x9[:,:,501:1000],784,:))
test_labels = kron(1:num_digits,ones(500))
yvector = hcat(kron(ones(500)',class_encoding_matrix[:,1]),kron(ones(500)',class_encoding_matrix[:,2]),kron(ones(500)',class_encoding_matrix[:,3]),kron(ones(500)',class_encoding_matrix[:,4]),kron(ones(500)',class_encoding_matrix[:,5]),kron(ones(500)',class_encoding_matrix[:,6]),kron(ones(500)',class_encoding_matrix[:,7]),kron(ones(500)',class_encoding_matrix[:,8]),kron(ones(500)',class_encoding_matrix[:,9]),kron(ones(500)',class_encoding_matrix[:,10]))
@show(yvector)
n = num_digits
d = 784
@show mu = 1e-12 ## make lower if it does notlearn
iters = 2000 
pcorrect = zeros(9)
W0 = zeros(n,d) ## why did we initlalize with @0
b0 = zeros(n)
for k = 2:10
    wtanh, btanh, loss_tanh = learn2classify_asgd_1layer(tanh,dtanh,grad_loss_1layer,xtrain[:,1:500*k],yvector[1:k,1:500*k],W0,b0,mu,iters)
    Ytanh = tanh.(wtanh*xtest .+ btanh)
    class_predictions_tanh = vec(mapslices(indmax,Ytanh,1)) 
    pcorrect[k-1] = sum(class_predictions_tanh .== test_labels)/length(test_labels)
end
scatter(digits[2:10],pcorrect,label = "1e-12")


