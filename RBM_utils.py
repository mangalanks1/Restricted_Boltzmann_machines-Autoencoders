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
    inference = np.random.binomial(n=1, p = p, size =p.shape)
    return p.T, inference

def P_x_given_h(W, c, h ):
    z= c + np.matmul(W, h)
    p = sigmoid(z)
    inference = np.random.binomial(n=1, p = p, size =p.shape)
    return p.T, inference

def gibbs_hvh(x, W, b, c, CD_K=1):
    ''' This function implements one step of Gibbs sampling,
        starting from the hidden state: For Contrastive Divergence'''
    h0_mean, h0_sample = P_h_given_x(W, b, x)
    k=0
    while k < CD_K:
        v1_mean, v1_sample = P_x_given_h(W, c, h0_sample )
        h1_mean, h1_sample = P_h_given_x(W, b, v1_sample)
        k = k + 1
    return [v1_mean, v1_sample, h1_mean, h1_sample]

def gibbs_vhv(x, W, b, c, K=1):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state: For Inference later/ Persistent CD-k'''
        v_sample = x
        k=0
        while k < K:
            h1_mean, h1_sample = P_h_given_x(W, b, v_sample) 
            v1_mean, v_sample = P_x_given_h(W, c, h1_sample)
            k = k + 1
        return [h1_mean, h1_sample, v1_mean, v_sample]
    
def free_energy(x, h):
        ''' Function to compute the free energy '''
        wx_b = b + np.matmul(x, W) 
        vbias_term = np.matmul(x.reshape(1,784),c.reshape(784,1))
        hidden_term = sum(np.log(1 + np.exp(wx_b)))
        e = -hidden_term - vbias_term
        e = e.reshape(1)
        return e

def Persistent_Contrastive_Divergence(x, W, b, c, CD_K=1):
    k=0
    v_sample = x
    h1_mean, h1_sample = P_h_given_x(W, b, v_sample)
    v1_mean, v1_sample = P_x_given_h(W, c, h1_sample ) #Start chain from HERE !
    while k < CD_K:
        h1_mean, h1_sample = P_h_given_x(W, b, v1_sample)
        v1_mean, v1_sample = P_x_given_h(W, c, h1_sample )
        k = k + 1
    return [v1_mean, v1_sample, h1_mean, h1_sample]

def sampling_from_RBM(W, b, c, K=1000):
    #X_start = np.random.normal(0, 0.1, (100,784) )# np.random.rand(100,784)
    X_hats=[]
    X_probs=[]
    for i in range(100):
        np.random.seed(i*10)
        #x = np.random.rand(784)
        x = np.random.normal(0,1,784 )
        [h1_mean, h1_sample, x, v1_sample] = gibbs_vhv(x, W, b, c, K=1000) 
        #[x, v1_sample, h1_mean, h1_sample] = gibbs_hvh(x, W, b, c, CD_K = 10)
        X_hats.append(v1_sample)
        X_probs.append(x)
    return X_hats, X_probs