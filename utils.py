import numpy as np, csv, matplotlib.pyplot as plt, random

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

def RBM_Initialize_Weights(shape):
    """ Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections.
   Note that W should be set to a matrix of size(L_out,  L_in+1) as
   Handle the "bias" terms separately! All biases should be initialized as 0
   """
    # Initialize W :use samples from a normal distribution with mean 0 and standard deviation 0.1
    w = np.random.normal(0,0.1,shape )
    return w


def RBM_cross_entropy_loss(x, x_hat):
    J = sum(- x*np.log(x_hat) - (1-x)*np.log(1-x_hat))
    return J

def P_h_given_x(W, b, x):
    z= b + np.matmul(x, W)
    p = sigmoid(z)
    return p

def P_x_given_h(W, c, h ):
    z= c + np.matmul(W, h)
    p = sigmoid(z)
    return p

def Contrastive_Divergence(x, W, b, c, k=1):
    i=0
    h_hat = P_h_given_x(W, b, x)
    while i<k:
        x_hat = P_x_given_h(W, c, h_hat )
        h_hat = P_h_given_x(W, b, x_hat)
        i=i+1
    return x_hat, h_hat

def sampling_from_RBM(W, b, c):
    #X_start = np.random.normal(0, 0.1, (100,784) )# np.random.rand(100,784)
    X_hats=[]

    for i in range(100):
        random.seed(i*i)
        x = np.random.rand(784 )
        x_hat, h_hat = Contrastive_Divergence(x, W, b, c, k=10*(i+1))
        X_hats.append(x_hat)
    return X_hats


# Utility Functions for Neural Network 
def ReLu(x):
    return np.maximum(0,x)

def ReLuGradient(x):
    g = x>0
    return g.astype(float)

def Tanh(x):
    return np.tanh(x)

def TanhGradient(x):
    return (1 - (x ** 2))

def sigmoid(z):
    """Computes the sigmoid function activation for a numpy array input z"""
    g = 1.0 / (1.0 + np.exp(-z))
    # Need to bound g so that log g doesnt reach infinity
    esp = .000000000001  # Lower bound of z
    g[g < esp] = esp
    g[g > (1 - esp)] = 1 - esp
    return g

def sigmoidGradient(x):
    return x*(1-x)

def softmax(x):
        softmax_result = None
        if x.ndim == 1:
            z = x - np.max(x)
            softmax_result = np.exp(z) / np.sum(np.exp(z))
            return softmax_result
        else:
            softmax_result = []
            for row in x:
                z = row - np.max(row)
                row_softmax_result = np.exp(z) / np.sum(np.exp(z))
                softmax_result.append(row_softmax_result)
            return np.array(softmax_result)

def randInitializeWeights(L_in, L_out):
    """ Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections.
   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
   the column row of W handles the "bias" terms
   pre-activations = Sum_ij(Wij.xj) + Wi0, j = L_in, i = L_out
   All biases should be initialized as 0
   """

    # Initialize W randomly so that we break the symmetry while training the neural network.
    # Sample W from Uniform [-b,b] where b = epsilon_init =  np.sqrt(6.0/(L_in+L_out))
    #Note: The first row of W corresponds to the parameters for the bias units
    epsilon_init = np.sqrt(6.0/(L_in+L_out))
    #w = np.random.uniform(-epsilon_init, epsilon_init,(L_out, 1 + L_in))
    w = np.random.uniform(-epsilon_init, epsilon_init,(L_out,  L_in))
    bias = np.zeros((L_out,1))
    w = np.hstack((w,bias))
    return w

def Unroll_weights(nn_weight_list, layer_sizes):
    for l in range(len(layer_sizes)-1):
        W = nn_weight_list[l]
        l_in_units = layer_sizes[l]
        l_out_units= layer_sizes[l+1]
        if l==0: nnparams = W.reshape(l_out_units*(l_in_units+1))
        else: nnparams = np.hstack((nnparams,W.reshape(l_out_units*(l_in_units+1))))
    return nnparams


def Roll_weights(nnparams,layer_sizes):
    # Lets store the weight matrices in a list nn_weight_list
    nn_weight_list=[]
    for l in range(len(layer_sizes)-1):
        l_in_units = layer_sizes[l]
        l_out_units= layer_sizes[l+1]
        if l==0:
            start=0
            end_i = l_out_units*(l_in_units+1)
        else:
            start = end_i
            end_i = start + l_out_units*(l_in_units+1)
        #print start, end_i
        theta = nnparams[start:end_i].reshape((l_out_units,(l_in_units+1)))
        nn_weight_list.append(theta)
    return nn_weight_list

def Mean_classification_error(Y,output_p):
    pred = np.argmax(output_p,axis=1)
    mean_err = 1-np.mean(pred==Y)
    return  mean_err

def forward_prop(layer_sizes, nn_weight_list, X, y,activ_func):
    # ------Feed-forward the neural network and return the activations in each layer-----
    m = X.shape[0]
    activations = []
    for i in range(len(layer_sizes)-1):
        theta = nn_weight_list[i]
        if i == 0:
            # Input Layer Sigmoid Activation
            temp = np.hstack((X, np.ones((m, 1))))          # (3000 785) x (100, 785)
            a = np.matmul(temp, theta.T)
            z = activ_func(a)
            activations.append(z)
        elif i == len(layer_sizes)-2:
            # Output Layer Softmax activation
            temp = np.hstack((activations[-1], np.ones((m, 1))))
            a = np.matmul(temp, theta.T)
            output_p = softmax(a)
            activations.append(output_p)
        else:
            temp = np.hstack((activations[-1], np.ones((m, 1))))
            a = np.matmul(temp, theta.T)
            z = activ_func(a)
            activations.append(z)
    return activations

def backprop(X, Y, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func):
    m = X.shape[0]
    num_labels = layer_sizes[-1]
    #Compute the gradients for each layer:
    "Indicator Function"
    indicator = np.zeros((m,num_labels))
    indicator[np.arange(m), Y] = 1
    delta=[] #reversed order gradients
    theta_grad=[]
    n_hidden = len(layer_sizes)-2
    for i in reversed(range(len(layer_sizes)-1)):
        if i == len(layer_sizes)-2: #hidden --> Output Layer
            output_p = activations[i]                  #3000x10
            delta3  = -(indicator -output_p)           # del {L}/del (pre-act L+1)"""\
            delta.append(delta3)

            h_K = activations[i-1 ]
            h_K = np.hstack((h_K, np.ones((m, 1))))
            #print W.shape, delta3.T.shape, h_K.shape
            Theta2_grad= np.matmul(delta3.T, h_K)      #10x10 "dont divide by m"
            # Adding regularization to the gradients
            W = nn_weight_list[i]
            Theta2_grad[:,0:-1] = Theta2_grad[:,0:-1] + reg_lambda *W[:,0:-1]
            theta_grad.append(Theta2_grad)

        elif i==0:
            a_K = activations[i]                        #3000 x 100  activation at layer i
            theta_Kplus = nn_weight_list[i+1][:,0:-1]   #10 x 100
            delta_plus = delta[-1]                      #3000x10
            #print('Multiply', delta_plus.shape, theta_Kplus.shape, sigmoidGradient(a_K).shape)
            delta2 = np.matmul(delta_plus,theta_Kplus)* activ_Grad_func(a_K) #a_K*(1-a_K)
            delta.append(delta2)

            a_K = np.hstack((X, np.ones((m, 1))))
            #print W.shape, delta_plus.shape, a_K.shape
            Theta1_grad= np.matmul(delta2.T, a_K)
            # Adding regularization to the gradients
            W = nn_weight_list[i]
            Theta1_grad[:,0:-1] = Theta1_grad[:,0:-1] + reg_lambda *W[:,0:-1]
            theta_grad.append(Theta1_grad)

        else : #Hidden Layers
            a_K = activations[i]                        #3000 x 100  activation at layer i
            theta_Kplus = nn_weight_list[i+1][:,0:-1]   #10 x 100
            delta_plus = delta[-1]                      #3000x10
            #print('Multiply', delta_plus.shape, theta_Kplus.shape, sigmoidGradient(a_K).shape)
            delta2 = np.matmul(delta_plus,theta_Kplus)*activ_Grad_func(a_K) #a_K*(1-a_K)
            delta.append(delta2)

            h_K = activations[ i - 1 ]
            h_K = np.hstack((h_K, np.ones((m, 1))))
            #print W.shape, delta_plus.shape, h_K.shape
            Theta2_grad= np.matmul(delta2.T, h_K)
            # Adding regularization to the gradients
            W = nn_weight_list[i]
            Theta2_grad[:,0:-1] = Theta2_grad[:,0:-1] + reg_lambda *W[:,0:-1]
            theta_grad.append(Theta2_grad)
            # del {L}/del (pre-act K)
    return theta_grad

def Train_network(epochmax, reg_lambda, LearningRate, nnparams, layer_sizes, minibatchsize, momentum, activ_func, activ_Grad_func, X_train, Y_train, X_val, Y_val):
    num_labels = layer_sizes[-1]
    #----------------Train the neural net:---------------------
    print("Training Neural net....")
    print('epochmax:{:3.0f}'.format(epochmax),' L2 Regularization: {:1.3f}'.format(reg_lambda),
      ' Learning rate: {:1.2f}'.format(LearningRate), 'Momentum : {:1.3f}'.format(momentum),' Layer Sizes',layer_sizes)

    train_cost=[]
    val_cost=[]
    err_tr = []
    err_val = []
    nn_weight_list =  Roll_weights(nnparams,layer_sizes)
    #Chain X & Y to shuffle together:
    temp = np.hstack((X_train,Y_train.reshape((Y_train.shape[0],1))))
    for t in range(epochmax):
        #Shuffle the training data at beginning of each epoch to learn different weights
        np.random.shuffle(temp)
        x_train, y_train = temp[:,0:-1], temp[:,-1].astype(int)
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
            x = x_train[start:end_i:].reshape((end_i-start,x_train.shape[1])) #Needs to be of shape (3000x784 --> 1x784)
            y= y_train[start:end_i]
            #Forward Prop:
            activations = forward_prop(layer_sizes, nn_weight_list, x, y, activ_func)
            #Back Prop gradients:
            #print t,k
            theta_grad = backprop(x, y, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func)
            theta_grad_prev = theta_grad
            #Update the weights:
            n_hidden = len(layer_sizes)-2
            for i in range(len(layer_sizes)-1):
                nn_weight_list[i] = nn_weight_list[i] - LearningRate*theta_grad[n_hidden - i]-LearningRate*momentum*theta_grad_prev[n_hidden - i]
        #Getting Train & Val Results
        activations = forward_prop(layer_sizes, nn_weight_list, X_train, Y_train, activ_func)
        output_p = activations[-1]
        J_train= cross_entropy_loss(num_labels, output_p, Y_train, reg_lambda, nn_weight_list)
        train_cost.append(J_train)
        mean_err = Mean_classification_error(Y_train,output_p)
        err_tr.append(mean_err)

        activation_val = forward_prop(layer_sizes, nn_weight_list, X_val, Y_val, activ_func)
        output_p = activation_val[-1]
        J_val =  cross_entropy_loss(num_labels, output_p, Y_val, reg_lambda, nn_weight_list)
        val_cost.append(J_val)
        mean_err2 = Mean_classification_error(Y_val,output_p)
        err_val.append(mean_err2)

        if t%10 ==0:
            print 'Time', t
            print "Cross Entropy","\t Training: ", J_train ,"\t Validation: ", J_val
            print "Mean  Error :","\t Training: ", mean_err ,"\t Validation:", mean_err2
        #Add Learning Rate Decay:
        if t>2 and abs(train_cost[-1]-train_cost[-2]) < 0.01 :
            LearningRate = LearningRate*0.5

    return train_cost, val_cost, err_tr, err_val, nn_weight_list

def cross_entropy_loss(num_labels, output_p, y_true, reg_lambda, nn_weight_list):
    m = y_true.shape[0]  # Number of samples
    """For the purpose of training a neural network, we need to recode the labels
    as vectors containing only values 0 or 1: eye_labels gives a one-hot encoded
    result corresponding to training label """
    J=0
    eye_labels = np.eye(num_labels)  # Diagonal matrix
    for i in range(m):
        a = eye_labels[:,y_true[i]]
        J = J - a*np.log(output_p[i,:]) - (1-a)*np.log(1-output_p[i,:])
    J = (sum(J)) /m
    w_sum = np.array(map(lambda x: sum(sum(x * x)), nn_weight_list))
    J = J +  (0.5 * reg_lambda / m) * (sum(w_sum))
    return J
