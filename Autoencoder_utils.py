import numpy as np, csv, matplotlib.pyplot as plt, random
import matplotlib

def Load_data(filepath='digitstrain.txt'):
    filepath = '/Users/Ankita/Documents/CMU_Acads/Deep_Leaning_10707/HW_1/NeuralNet/'+filepath
    reader = csv.reader(open(filepath, "rb"), delimiter=",")
    x = list(reader)
    digitstrain = np.array(x).astype("float")
    X = digitstrain[:,0:784]
    Y = digitstrain[:,784].astype(int)
    return X, Y

def sigmoid(z):
    """Computes the sigmoid function activation for a numpy array input z"""
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoidGradient(x):
    return x*(1-x)

# -------------------------Set Activation Function---------------------------
activ_func = sigmoid                   # can be sigmoid, ReLu, Tanh
activ_Grad_func = sigmoidGradient      # can be sigmoidGradient, ReLuGradient, TanhGradient


def display_data(X):
    """display_data Display 2D data in a nice 10x10 grid
   display_data(X) displays 2D data stored in X in a nice grid.
   """
    num_img = X.shape[0]
    #Select 100 random images:
    img_ix = random.sample(np.arange(0,num_img), 100);
    plt.figure(figsize=(10,10))
    for i in range(1,101):
        ax = plt.subplot(10,10,i)
        data = X[img_ix[i-1],:].reshape((28,28))
        plt.imshow(data, cmap=plt.cm.Greys)
        ax.axis('off')
    plt.show()

def Autoencoder_Initialize_Weights(shape):
    """ Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections.
   Note that W should be set to a matrix of size(L_out,  L_in+1) as
   Handle the "bias" terms separately! All biases should be initialized as 0
   """
    # Initialize W :use samples from a normal distribution with mean 0 and standard deviation 0.1
    w = np.random.normal(0,0.1,shape )
    return w

def Autoencoder_cross_entropy_loss(output_p, X, reg_lambda, nn_weight_list):
    m = X.shape[0]  # Number of samples
    J=0
    for i in range(m):
        J = J - X[i]*np.log(output_p[i,:]) - (1-X[i])*np.log(1-output_p[i,:])
    J = np.sum(J) /m
    w_sum = np.array(map(lambda w: sum(sum(w * w)), nn_weight_list))
    #print "J_orig", J, "reg loss", (0.5 * reg_lambda / m) * (sum(w_sum))
    J = J +  (0.5 * reg_lambda / m) * (sum(w_sum))
    return J

def Autoencoder_forward_prop(layer_sizes, nn_weight_list, X, activ_func):
    # ------Feed-forward the neural network and return the activations in each layer-----
    m = X.shape[0]
    activations = []
    W, bias1, bias2 = nn_weight_list
    for i in range(len(layer_sizes)-1):
        if i == 0:
            # Input Layer Sigmoid Activation
            theta = np.hstack((W.T,bias1)).T
            temp = np.hstack((X, np.ones((m, 1))))          # (3000 785) x (100, 785)
            a = np.matmul(temp, theta)
            z = activ_func(a)
            activations.append(z)
        else:
            # Output Layer Sigmoid Activation
            theta = np.hstack((W, bias2)).T 
            temp = np.hstack((activations[-1], np.ones((m, 1))))
            a = np.matmul(temp, theta)
            z = activ_func(a)
            activations.append(z)
    return activations

def Autoencoder_nnCostFunction(nn_weight_list, layer_sizes, X, reg_lambda):
    """Implements the cost function and gradient for a single hidden layer autoencoder
   The returned parameter grad should be a "unrolled" vector of the partial derivatives of the autoencoder."""
    m = X.shape[0] # Number of samples

    #------------------Feedforward the neural network------------------------------------
    activations = Autoencoder_forward_prop(layer_sizes, nn_weight_list, X, activ_func)
    output_p = activations[-1]

    #----------------Compute the cost function J of the feed forward:---------------------
    J = Autoencoder_cross_entropy_loss(output_p, X, reg_lambda, nn_weight_list)
    #print("Cost is: ",J)

    return J, activations

def Autoencoder_backprop(X, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func):
    m = X.shape[0]
    W, bias1, bias2 = nn_weight_list
    num_labels = layer_sizes[-1]
    #Compute the gradients for each layer:
    delta=[] #reversed order gradients
    theta_grad=[]
    n_hidden = len(layer_sizes)-2
    for i in reversed(range(len(layer_sizes)-1)):
        if i != 0: #hidden --> Output Layer
            output_p = activations[i]                  #3000x10
            delta3  = -(X - output_p)           # del {L}/del (pre-act L+1)"""\
            delta.append(delta3)
            
            h_K = activations[i-1]
            h_K = np.hstack((h_K, np.ones((m, 1))))
            #print W.shape, delta3.T.shape, h_K.shape
            Theta2_grad= np.matmul(delta3.T, h_K)      #10x10 "dont divide by m"
            # Adding regularization to the gradients
            Theta2_grad[:,0:-1] = Theta2_grad[:,0:-1] + reg_lambda *W
            theta_grad.append(Theta2_grad)

        else :
            a_K = activations[i]                        #3000 x 100  activation at layer i
            theta_Kplus = W.T                           #10 x 100
            delta_plus = delta[-1]                      #3000x10
            #print('Multiply', delta_plus.shape, theta_Kplus.shape, sigmoidGradient(a_K).shape)
            delta2 = np.matmul(delta_plus,theta_Kplus.T)* activ_Grad_func(a_K) #a_K*(1-a_K)
            delta.append(delta2)

            a_K = np.hstack((X, np.ones((m, 1))))
            #print W.shape, delta_plus.shape, a_K.shape
            Theta1_grad= np.matmul(delta2.T, a_K)
            # Adding regularization to the gradients
            Theta1_grad[:,0:-1] = Theta1_grad[:,0:-1] + reg_lambda *W.T
            theta_grad.append(Theta1_grad)
    return theta_grad

def Autoencoder_Train_network(epochmax, reg_lambda, LearningRate, minibatchsize, nn_weight_list, 
                              layer_sizes, momentum,  activ_func, activ_Grad_func, X_train, X_val):
    
    
    [input_layer_size, hidden_layer_size, output_layer_size] = layer_sizes
    #----------------Train the neural net:---------------------
    print("Training Neural net....")
    print('epochmax:{:3.0f}'.format(epochmax),' L2 Regularization: {:1.3f}'.format(reg_lambda),
      ' Learning rate: {:1.2f}'.format(LearningRate), 'Momentum : {:1.3f}'.format(momentum),' Layer Sizes',layer_sizes)

    train_cost=[]
    val_cost=[]
    for time in range(epochmax):
        #Shuffle the data at beginning of each epoch to learn different weights
        np.random.shuffle(X_train)
        np.random.shuffle(X_val)
        N = X_train.shape[0]
        ite = int(N/minibatchsize)+1
        theta_grad_prev=0
        for k in range(ite):
            if k==0:
                start=0
                end_i = minibatchsize
            elif k!= ite -1:                        #drop the last  minibatch
                start = end_i
                end_i = start + minibatchsize

            x = X_train[start:end_i:].reshape((end_i-start,X_train.shape[1])) #Needs to be of shape (3000x784 --> 1x784)
            #-----------------------Training: Compute Forward propagation:-----------------------------
            activations = Autoencoder_forward_prop(layer_sizes, nn_weight_list, x, activ_func)
            #----------------Training: Compute gradients for the back propagation:---------------------
            theta_grad = Autoencoder_backprop(x, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func)
            theta_grad_prev = theta_grad
            #--------------------------------Update the weights-----------------------------------------
            #Taking sum of the two gradients for Autoencoder
            W, bias1, bias2 = nn_weight_list
            
            theta_grad_W = theta_grad[0][:,0:-1] + theta_grad[1][:,0:-1].T
            theta_grad_prev_W = theta_grad_prev[0][:,0:-1] + theta_grad_prev[1][:,0:-1].T
            
            theta_grad_b1 = theta_grad[1][:,-1].reshape((hidden_layer_size,1))
            theta_grad_prev_b1 = theta_grad_prev[1][:,-1].reshape((hidden_layer_size,1))
            
            theta_grad_b2 = theta_grad[0][:,-1].reshape((output_layer_size,1))
            theta_grad_prev_b2 = theta_grad_prev[0][:,-1].reshape((output_layer_size,1))
            
            W = W - LearningRate*theta_grad_W - LearningRate*momentum*theta_grad_prev_W
            bias1 = bias1 -  LearningRate*theta_grad_b1 - LearningRate*momentum*theta_grad_prev_b1
            bias2 = bias2 -  LearningRate*theta_grad_b2 - LearningRate*momentum*theta_grad_prev_b2

            nn_weight_list = [W, bias1, bias2]
            
        #------------------------------------------Validation------------------------------------------
        #Getting Train & Val Results
        J_train, activations = Autoencoder_nnCostFunction(nn_weight_list, layer_sizes, X_train, reg_lambda)
        train_cost.append(J_train)

        J_val, _ = Autoencoder_nnCostFunction(nn_weight_list, layer_sizes, X_val, reg_lambda)
        val_cost.append(J_val)

        if time%5 ==0:
            print 'Time', time
            print "Cross Entropy","\t Training: ", J_train ,"\t Validation: ", J_val
        
        #Add Learning Rate Decay:
        if time>2 and abs(train_cost[-1]-train_cost[-2]) < 0.1:
            LearningRate = LearningRate*0.5
            

    """Plot the average training cross-entropy error (sum of the cross-entropy error terms over the training dataset
    divided by the total number of training example) on the y-axis vs. the epoch number (x-axis). On the same figure,
    plot the average validation cross-entropy error function."""
    return  train_cost, val_cost, nn_weight_list

def Denoise_X(X, p=0.1):
    mask = np.random.binomial(n=1,p=p,size=X.shape)
    X_new = ((1-mask)*X)
    return X_new

def Train_network_denoise(epochmax, reg_lambda, LearningRate, minibatchsize, nn_weight_list, layer_sizes, momentum,  
                          activ_func, activ_Grad_func, X_train, X_val, p =0.1):
    
    [input_layer_size, hidden_layer_size, output_layer_size] = layer_sizes
    #----------------Train the neural net:---------------------
    print("Training Neural net....")
    print('epochmax:{:3.0f}'.format(epochmax),' L2 Regularization: {:1.3f}'.format(reg_lambda),
      ' Learning rate: {:1.2f}'.format(LearningRate), 'Momentum : {:1.3f}'.format(momentum),' Layer Sizes',layer_sizes)

    train_cost=[]
    val_cost=[]
    for time in range(epochmax):
        #Shuffle the data at beginning of each epoch to learn different weights
        np.random.shuffle(X_train)
        np.random.shuffle(X_val)
        N = X_train.shape[0]
        ite = int(N/minibatchsize)+1
        theta_grad_prev=0
        for k in range(ite):
            if k==0:
                start=0
                end_i = minibatchsize
            elif k!= ite -1:                        #drop the last  minibatch
                start = end_i
                end_i = start + minibatchsize

            x = X_train[start:end_i:].reshape((end_i-start,X_train.shape[1])) #Needs to be of shape (3000x784 --> 1x784)
            x = Denoise_X(x, p=0.1)
            #-----------------------Training: Compute Forward propagation:-----------------------------
            activations = Autoencoder_forward_prop(layer_sizes, nn_weight_list, x, activ_func)
            #----------------Training: Compute gradients for the back propagation:---------------------
            theta_grad = Autoencoder_backprop(x, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func)
            theta_grad_prev = theta_grad
            #--------------------------------Update the weights-----------------------------------------
            #Taking sum of the two gradients for Autoencoder
            W, bias1, bias2 = nn_weight_list
            
            theta_grad_W = theta_grad[0][:,0:-1] + theta_grad[1][:,0:-1].T
            theta_grad_prev_W = theta_grad_prev[0][:,0:-1] + theta_grad_prev[1][:,0:-1].T
            
            theta_grad_b1 = theta_grad[1][:,-1].reshape((hidden_layer_size,1))
            theta_grad_prev_b1 = theta_grad_prev[1][:,-1].reshape((hidden_layer_size,1))
            
            theta_grad_b2 = theta_grad[0][:,-1].reshape((output_layer_size,1))
            theta_grad_prev_b2 = theta_grad_prev[0][:,-1].reshape((output_layer_size,1))
            
            W = W - LearningRate*theta_grad_W - LearningRate*momentum*theta_grad_prev_W
            bias1 = bias1 -  LearningRate*theta_grad_b1 - LearningRate*momentum*theta_grad_prev_b1
            bias2 = bias2 -  LearningRate*theta_grad_b2 - LearningRate*momentum*theta_grad_prev_b2

            nn_weight_list = [W, bias1, bias2]
            
        #------------------------------------------Validation------------------------------------------
        #Getting Train & Val Results
        J_train, activations = Autoencoder_nnCostFunction(nn_weight_list, layer_sizes, X_train, reg_lambda)
        train_cost.append(J_train)

        J_val, _ = Autoencoder_nnCostFunction(nn_weight_list, layer_sizes, X_val, reg_lambda)
        val_cost.append(J_val)

        if time%10 ==0:
            print 'Time', time
            print "Cross Entropy","\t Training: ", J_train ,"\t Validation: ", J_val
        
        #Add Learning Rate Decay:
        if time>2 and abs(train_cost[-1]-train_cost[-2]) < 0.1:
            LearningRate = LearningRate*0.5
            

    """Plot the average training cross-entropy error (sum of the cross-entropy error terms over the training dataset
    divided by the total number of training example) on the y-axis vs. the epoch number (x-axis). On the same figure,
    plot the average validation cross-entropy error function."""
    return  train_cost, val_cost, nn_weight_list