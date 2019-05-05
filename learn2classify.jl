
using Plots, Interact
gr()
include("./deps.jl") ## Dependences file for displaying an image

linear(z) = z

function dlinear(z)
    return 1 ##TODO: fill in ?? with the derivative of the linear activation function
end

dtanh(z)  = 1-tanh(z)^2

f₁(x, w, b,f_a) = f_a.(w*x .+ b) # Note the subscript 1 in the name!!

# Functions in Julia can have subscripts, greek letters,
# or any unicode characters! I entered "\_1" followed by 
# the "Tab" key to get ₁ -- try it yourself in the scratchpad!

# (Psst! You can access the scratchpad by clicking on the
# triangle in the lower right of your screen! 
# Then press shift+enter to execute your command)

x = linspace(-10,10,101) # 101 equally spaced numbers from -10 to 10
@manipulate for w = [-10,-5,-2.5,2.5,5,10], b = [-10,-5,-2.5,0,2.5,5,10], f_a = [tanh,linear]
   # plot x vs f1(x, w, b) for the w and b chosen by buttons:
    plot(x, f₁(x, w, b, f_a),
        label="", xlabel = "x", ylabel = "f1(x)", title = "$f_a activation", ylims=(-10,10))
end

function gn(x,w,b,f_a )
    ## TODO: Fill in ?? to return an Nx1 array with elements given by Equation (2) when x is an n X N array  
    
    return f_a.(x'*w .+ b)  

end

function grad_loss(f_a,df_a,x,y,w,b,normalize=true)
    
    dw = zeros(length(w))
    db = 0.0
    loss = 0.0
    for j = 1 : size(x,2)
        error =  y[j] - f_a.(w'*x[:,j]+b) ## TODO: fill in ??
        common_term = (error .* df_a.(w'*x[:,j]+b))
        dw = dw - 2 * common_term.*x[:,j] ## TODO: fill in ?? Ignore the 1/N part that is done below
        db = db - 2 * common_term *1 ## TODO: fill in ??
        loss = loss + error^2
     end
    
     if normalize
        dw = dw/length(y)
        db = db/length(y)
        loss = loss/length(y)
    end
    return dw, db, loss
 end

function learn2classify_gd(f_a,df_a,grad_loss,x,y,mu=1e-3,iters=500,show_loss=true,normalize=true,seed=1)
    
    n = size(x,1)
    
    if seed == false
        w = zeros(n)
        b = 0.0
    else
        srand(seed) #initialize random number generator
        w = randn(n)
        b = rand()
    end
    
    loss = zeros(iters) 
    for i = 1 : iters
      
        dw, db, loss_i = grad_loss(f_a,df_a,x,y,w,b,normalize) ## TODO: fill in ??
        w = w - mu*dw  ## TODO: fill in ??
        b = b - mu*db ## TODO: fill in ??
        loss[i] = convert(Float64,loss_i[1])
        
        if show_loss == true
            if(rem(i,100) == 0)
                IJulia.clear_output(true)
                loss_plot = scatter([1:50:i],loss[1:50:i],yscale=:log10, 
                                    xlabel = "iteration", 
                                    ylabel = "training loss", 
                                    title = "iteration $i, loss = $loss_i") 
            
                display(loss_plot)
                sleep(0.1)
            end
        end
    
    end
    
    return w,b, loss
end



randperm(10)

n = 10
N = 1000
x = randn(n,N)
w = randn(n)
b = rand(1)
y = gn(x,w,b,linear)

mu = 1 ## TODO: change mu till network learns
@time what, bhat, loss = learn2classify_gd(linear,dlinear,grad_loss,x,y,mu,1000,true);


smaller_mu = 0.01*mu
@time what, bhat, loss = learn2classify_gd(linear,dlinear,grad_loss,x,y,smaller_mu,1000,true);

function learn2classify_sgd(f_a,df_a, grad_loss,x,y,mu=1e-3,iters=500,batch_size=10,show_loss=true,normalize=true,seed=1)
    
    n = size(x,1)
    N = size(x,2) 
    
    if seed == false
        srand(1)
        w = zeros(n)
        b = 0.0
    else
        srand(seed) # Initialize random number generator
        w = randn(n)
        b = rand()
    end
    
    loss = zeros(iters) 
    for i = 1 : iters
      
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size,N)]
        
        xbatch = x[:,batch_idx]
        ybatch = y[batch_idx]
        dw, db, loss_i= grad_loss(f_a,df_a,xbatch,ybatch,w,b,normalize) ## TODO: fill in ??
        w = w - mu*dw
        b = b - mu*db
        loss[i] = convert(Float64,loss_i[1])
        
        if show_loss == true
            if(rem(i,100) == 0)
                IJulia.clear_output(true)
                loss_plot = scatter([1:50:i],loss[1:50:i],yscale=:log10, 
                                    xlabel = "iteration", 
                                    ylabel = "training loss", 
                                    title = "iteration $i, loss = $loss_i") 
            
                display(loss_plot)
                sleep(0.1)
            end
        end
    
    end
    
    return w,b, loss
end

x = [1,2,3,4,5,6,7,8,9,10]
batch_idx = randperm(10)
batch_idx = batch_idx[1:5]
x[batch_idx]
size(x,2)

mu = 0.1
@time what, bhat, loss = learn2classify_sgd(linear,dlinear,grad_loss,x,y,mu,500,10,true);


batch_size = 10
new_mu = mu ## does it still learn? Hint: how tp normalize with batch_size 
@time what, bhat, loss = learn2classify_sgd(linear,dlinear,grad_loss,x,y,new_mu,500,batch_size,true,false);


function learn2classify_asgd(f_a,df_a,grad_loss,x,y,mu=1e-3,iters=500,batch_size = 10,show_loss=true,normalize=true,seed=1)


    n = size(x,1)
    N = size(x,2)

    if seed == false
        srand(1)
        b = 0.0
        w = zeros(n)
    else
        srand(seed) # initiliaze random number generator
        w = randn(n)
        b = rand()
    end

    loss = zeros(iters)


    lambdak = 0
    qk = w
    pk = b
    for i = 1 : iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size,N)]
        
        xbatch = x[:,batch_idx]
        ybatch = y[batch_idx]

        dw, db, loss_i = grad_loss(f_a,df_a,xbatch,ybatch,w,b,normalize) 
        
        ## TODO: fill in ??

        qkp1 = w - mu*dw 
        pkp1 = b - mu*db 
        
        lambdakp1 = (1+sqrt(1+4*lambdak^2))/2  ## Expression in in terms of lambdak
        gammak = (1-lambdak)/lambdakp1

        w = (1-gammak)*qkp1 + gammak*qk 
        b = (1-gammak)*pkp1 + gammak*pk 

        qk = qkp1
        pk = pkp1
        lambdak = lambdakp1

        loss[i] = convert(Float64,loss_i[1])

        if show_loss == true
            if(rem(i,100) == 0)
                IJulia.clear_output(true)
                loss_plot = scatter([1:50:i],loss[1:50:i],yscale=:log10,
                                    xlabel = "iteration",
                                    ylabel = "training loss",
                                    title = "iteration $i, loss = $loss_i")

                display(loss_plot)
                sleep(0.1)
            end
        end

    end

    return w,b, loss
end


@time what, bhat, loss = learn2classify_asgd(linear,dlinear,grad_loss,x,y,mu,500,10,true,true,false);


function load_digit_data(digits,nx=28,ny=28,nrep=1000)
    file0 = "data"*string(digits[1])
    file1 = "data"*string(digits[2])

    fp = open(file0, "r")
    x0 = read(fp, UInt8, (nx,ny,nrep)) # what is the type of x0?
    close(fp)

    fp = open(file1, "r")
    x1 = read(fp, UInt8, (nx,ny,nrep))
    close(fp)
     
    return x0, x1
end





digits = [1,9]
nx, ny, nrep = 28, 28, 1000
x0, x1 = load_digit_data(digits);
x0  

@manipulate for idx = 1 : nrep
    showimg(x0[:,:,idx],transpose=true,color=:grays,title="sample $idx",aspect_ratio=1.0)
end

@manipulate for idx = 1 : nrep
    showimg(x1[:,:,idx], transpose=true, color=:grays,title="sample $idx",aspect_ratio=1.0)
end

function generate_test_train_set_from_datacube(data,percentage_train=50.0)
    num_samples = size(data,3)
    num_train = convert(Int,round(percentage_train/100.0*num_samples))
    num_test = num_samples - num_train
    
    rand_idx = randperm(num_samples)
    train_idx = rand_idx[1:num_train]
    test_idx = rand_idx[num_train+1:end]
    
    train_data = data[:,:,train_idx]
    test_data = data[:,:,test_idx]
    
    return train_data, test_data
end   



x0_train, x0_test = generate_test_train_set_from_datacube(x0)
x1_train, x1_test = generate_test_train_set_from_datacube(x1);


@show typeof(x0_train)
size(x0_train)

function datacube2matrix(data_cube)
       
    return float(reshape(data_cube,:,size(data_cube,3))) 
end

x0_train_matrix = datacube2matrix(x0_train) 
x1_train_matrix = datacube2matrix(x1_train)
x0_test_matrix = datacube2matrix(x0_test)
x1_test_matrix = datacube2matrix(x1_test);
size(x0_train_matrix)

function encode_class0_class1_data(class0_matrix,class1_matrix,encoding_vector=[0, 1])
    num_class0 = size(class0_matrix,2)
    num_class1 = size(class1_matrix,2)
    class_matrix = hcat(class0_matrix,class1_matrix)
    class_vector = vcat(encoding_vector[1]*ones(num_class0),encoding_vector[2]*ones(num_class1))
    return class_matrix, class_vector
end


class_encoding_vector = [-1.0,1.0] # this makes the first class equal to "-1" and the second class = "1"

digit2class = Dict(string(digits[i])=> class_encoding_vector[i] for i = 1 : length(digits) )



class2digit = Dict(value=> key for (key,value) in digit2class)
@show class2digit

class2digit[-1]

train_matrix, train_vector = encode_class0_class1_data(x0_train_matrix,x1_train_matrix,[-1,1])
test_matrix, test_vector = encode_class0_class1_data(x0_test_matrix,x1_test_matrix,[-1,1]);


scatter(train_vector)

f_a = tanh
df_a = dtanh
mu = 1e-14 ##TODO: select a mu
@time what, bhat, loss = learn2classify_sgd(f_a,df_a,grad_loss,train_matrix,train_vector,mu,5000,20,true,true,false);


@time what, bhat, loss = learn2classify_asgd(f_a,df_a,grad_loss,train_matrix,train_vector,mu,5000,20,true,true,false);


seed = 1 # for random number generator
what_alt, bhat_alt, loss_alt = learn2classify_asgd(f_a,df_a,grad_loss,train_matrix,train_vector,mu,5000,20,true,true,seed);


scatter(what,label="learned w vector") # elements of the learned w vector  

showimg(reshape(what,nx,ny),transpose=:true,color=:grays,aspect_ratio=1.0)

predicted_values = gn(test_matrix,what,bhat,f_a)
scatter(predicted_values,title="nn output on test data")


test_prediction = sign.(gn(test_matrix,what,bhat,f_a)) 
## How would we modify above if the encoding was not -1 and +1 but given by elements of class_vector
scatter(test_prediction,title="class encoding predictions on test data")


pcorrect = sum(test_prediction .== test_vector)/length(test_vector)*100
@show pcorrect

mu1 = 1e-11
mu2 = 1e-10
mu3 = 1e-9
f_a = tanh
df_a = dtanh
what1, bhat1, loss1 = learn2classify_gd(f_a,df_a,grad_loss,train_matrix,train_vector,mu1,500,false,true,false);
# what2, bhat2, loss2 = learn2classify_gd(f_a,df_a,grad_loss,train_matrix,train_vector,mu2,500,false,true,false);
# what3, bhat3, loss3 = learn2classify_gd(f_a,df_a,grad_loss,train_matrix,train_vector,mu3,500,false,true,false);

test_prediction1 = sign.(gn(test_matrix,what1,bhat1,f_a)) 
# test_prediction2 = sign.(gn(test_matrix,what2,bhat2,f_a)) 
# test_prediction3 = sign.(gn(test_matrix,what3,bhat3,f_a)) 

pcorrect1 = sum(test_prediction1 .== test_vector)/length(test_vector)*100
# pcorrect2 = sum(test_prediction2 .== test_vector)/length(test_vector)*100
# pcorrect3 = sum(test_prediction3 .== test_vector)/length(test_vector)*100

# R = scatter(pcorrect1,color="red", label = "Red Data", s = 40)
# G = scatter(class_encoding_vector,pcorrect2,color="blue", label = "Blue Data", s = 60)
# B = scatter(class_encoding_vector,pcorrect3,color="green", label = "Green Data", s = 80)

@manipulate for i = 1 : size(test_matrix,2)
    predicted_digit = class2digit[test_prediction[i]]
    showimg(reshape(test_matrix[:,i],nx,ny),
        title ="Classified as $(predicted_digit)",
        color=:grays,
        transpose=true,
        aspect_ratio=1.0)
end

wrong_predictions_matrix = test_matrix[:,test_prediction.!== test_vector]

@manipulate for i = 1: size(wrong_predictions_matrix,2)
    showimg(reshape(wrong_predictions_matrix[:,i],nx,ny),
        color=:grays,
        transpose=true,
        title="incorrectly classified ",
        aspect_ratio=1.0)
end

using WebWidgets, Blink, Colors, InteractNext, Base.Test

app, my_app_img = drawnumber(brushsize = 14, resolution = (250,250))
app


my_img_from_app = Gray.(my_app_img[])
my_img_from_app = float(my_img_from_app)
my_img = 255*(my_img_from_app');


showimg(my_img_from_app,
    title = "Image produced by app",
    transpose=:true,
    color=:grays,
    aspectratio=1.0)

showimg(my_img,
    title = "Image produced by app transformed into `my_img`",
    transpose=:true,
    color=:grays,
    aspectratio=1.0)

my_img_vector = (my_img[:])
class_prediction = sign.(gn(my_img[:],what,bhat,f_a))
my_digit_prediction = class2digit[class_prediction[]]
showimg(my_img,
    title = "Network predicts $my_digit_prediction",
    transpose=:true,
    color=:grays,
    aspectratio=1.0)


function compute_pcorrect(digits,f_a,df_a,class_encoding_vector=[-1,1]) 
    
    x0, x1 = load_digit_data(digits)
    
    ### TODO: complete based on code from above
    x0_train, x0_test = generate_test_train_set_from_datacube(x0)
    x1_train, x1_test = generate_test_train_set_from_datacube(x1);
    x0_train_matrix = datacube2matrix(x0_train) 
    x1_train_matrix = datacube2matrix(x1_train)
    x0_test_matrix = datacube2matrix(x0_test)
    x1_test_matrix = datacube2matrix(x1_test);
    train_matrix, train_vector = encode_class0_class1_data(x0_train_matrix,x1_train_matrix,[-1,1])
    test_matrix, test_vector = encode_class0_class1_data(x0_test_matrix,x1_test_matrix,[-1,1]);
#     grad_loss = grad_loss(f_a,df_a,x,y,w,b,normalize=true)
    show_loss = false
    batch_size = 20
    mu = 10e-14 # use the same one from earlier 
    iters = 5000
    what, bhat, loss = learn2classify_asgd(f_a,df_a,grad_loss,train_matrix,train_vector,mu,iters,batch_size,show_loss,true,false);
    test_prediction = sign.(gn(test_matrix,what,bhat,f_a)) 
    return pcorrect = sum(test_prediction .== test_vector)/length(test_vector)*100
end

function compute_pcorrect2(digits,f_a,df_a,class_encoding_vector=[0,1]) 
    
    x0, x1 = load_digit_data(digits)
    
    ### TODO: complete based on code from above
    x0_train, x0_test = generate_test_train_set_from_datacube(x0)
    x1_train, x1_test = generate_test_train_set_from_datacube(x1);
    x0_train_matrix = datacube2matrix(x0_train) 
    x1_train_matrix = datacube2matrix(x1_train)
    x0_test_matrix = datacube2matrix(x0_test)
    x1_test_matrix = datacube2matrix(x1_test);
    train_matrix, train_vector = encode_class0_class1_data(x0_train_matrix,x1_train_matrix,[0,1])
    test_matrix, test_vector = encode_class0_class1_data(x0_test_matrix,x1_test_matrix,[0,1]);
    show_loss = false
    batch_size = 20
    mu = 10e-14 # use the same one from earlier 
    iters = 5000
    what, bhat, loss = learn2classify_asgd(f_a,df_a,grad_loss,train_matrix,train_vector,mu,iters,batch_size,show_loss,true,false);
    test_prediction = sign.(gn(test_matrix,what,bhat,f_a) - 1/2)*(1/2)+1/2
    return pcorrect = sum(test_prediction .== test_vector)/length(test_vector)*100
end

 compute_pcorrect(digits,f_a,df_a) 

f_a = linear
df_a = dlinear
class_encoding_vector = [-1,1]
## TODO: what other parameters need to be defined

linear_classify_matrix_upper_diagonal = [digit1 < digit2 ? compute_pcorrect([digit1,digit2], f_a, df_a,class_encoding_vector) : 0.0 for digit1 = 0 : 9, digit2 = 0 : 9 ]
@show linear_classify_matrix_upper_diagonal
linear_classify_matrix = linear_classify_matrix_upper_diagonal + linear_classify_matrix_upper_diagonal' + 100*I 
# Julia figures out I dimension automatically!
@show linear_classify_matrix


showimg(linear_classify_matrix,yflip = true) ## what other arugments so it displays correctly?

f_a = tanh
df_a = dtanh
tanh_classify_matrix_upper_diagonal = [digit1 < digit2 ? compute_pcorrect([digit1,digit2],f_a,df_a) : 0.0 for digit1 = 0 : 9, digit2 = 0 : 9 ]
tanh_classify_matrix = tanh_classify_matrix_upper_diagonal + tanh_classify_matrix_upper_diagonal' + 100*I
showimg(tanh_classify_matrix,xflip = true) ## what other arugments so it displays correctly?



function sigmoid(z)
    
    return 1/(1+exp(-z)) ## 
    
end

function dsigmoid(z)
    sigmoid_z = 1.0/(1.0+exp(-z))
    return sigmoid_z*(1-sigmoid_z)
end

##TODO Compute the classifcation matrix
f_a = sigmoid
df_a = dsigmoid
@show sigm_classify_matrix_upper_diagonal = [digit1 < digit2 ? compute_pcorrect([digit1,digit2],f_a,df_a) : 0.0 for digit1 = 0 : 9, digit2 = 0 : 9 ]
@show sigm_classify_matrix = sigm_classify_matrix_upper_diagonal + sigm_classify_matrix_upper_diagonal' + 100*I
# showimg(sigm_classify_matrix,xflip = true)


##TODO Compute the classifcation matrix for the modified code
f_a = sigmoid
df_a = dsigmoid
sigm_classify_matrix_upper_diagonal = [digit1 < digit2 ? compute_pcorrect2([digit1,digit2],f_a,df_a) : 0.0 for digit1 = 0 : 9, digit2 = 0 : 9 ]
@show sigm_classify_matrix = sigm_classify_matrix_upper_diagonal + sigm_classify_matrix_upper_diagonal' + 100*I

