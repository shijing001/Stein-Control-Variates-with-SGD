# load modules
import torch
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy import stats
#%matplotlib inline
from scipy import special
import time
from torch.autograd import Variable
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
import pytorch_warmup as warmup
from scipy.spatial.distance import cdist

class NormalMeasure(object):
    def __init__(self, sigma):
        """
        initialization
        :params[in], sigma, the standard deviation of Gaussian distr.
        """
        self.sigma = sigma


    def score_func(self, x):
        """
        evaluate the score function, i.e., gradient of log density
        
        :params[in], x, a one-dimensional tensor 
        
        :params[out], score, a 1-d tensor
        """
        score = -1*x/(self.sigma**2)
        return score
    
    def sample(self, n, d, random_seed=0):
        """
        draw a sample from the multivariate Gaussian distr.
        
        :params[in], n, an integer, sample size
        :params[in], d, an integer, dimension
        :params[in], random_seed, int, random seed
        
        :params[out], samp, a n-by-d matrix tensor, of which each row represents a sample
        """
        torch.manual_seed(random_seed)
        samp = self.sigma*torch.randn(n, d)
        return samp

    
## class: Oates 2017 kernel
class OatesKernel(object):
    
    def __init__(self, input_data, score_vals, kernel_alpha1, kernel_alpha2, train_pct=.95):
        """
        functions related to kernel in Oates 2017 paper
        
        :params[in]: input_data, a 2-d tensor, each row represents a training example
        :params[in]: score_vals, a 2-d tensor, each row represents a vector of a training
            example
        :params[in]: kernel_alpha1, first parameter in kernel
        :params[in]: kernel_alpha2, 2nd parameter in kernel
        :params[in]: kernel_lam, regularization parameter in kernel
        :params[in]: normal_std, std dev. in normal dist.      
        :params[in]: train_pct, percent of data used for training 
        """
        self.X = input_data  # the input data for GP
        ## the dimension of input data -- dim. of random variables
        self.dim = input_data.size()[1] 
        ## score matrix
        self.score_vals = score_vals
        ## the number of samples
        self.sample_size = input_data.size()[0] 
        ## standard deviation in the base kernel
        self.kernel_std = kernel_alpha2 
        ## percent of data for training
        self.train_pct = train_pct
        ## first parameter in kernel function
        self.kernel_alpha1 = kernel_alpha1
        ## compute the association matrix
        #self.kernel_matrix = self.comp_kernel_matrix()
        ## training sample size
        self.train_size = int(self.sample_size*train_pct)
        ## regularized kernel matrix for training data
        #self.kernel_train, self.reg_lam = self.tikhonov(self.kernel_matrix[:self.train_size,\
        #                                                                   :self.train_size])
        
    ## base kernel functions
    def base_kernel(self, x, y):
        """
        evaluate the kernel function and associated derivatives
        at input vectors x and y
        
        :params[in]: x, a 1-d tensor
        :params[in]: y, same as x
        
        :params[out]:  a tuple of related tensors
        """
        ## difference between two tensors
        x_minus_y = x-y
        ## quadratic terms for x
        quad_x = (1. + self.kernel_alpha1*x.pow(2).sum()).pow(-1)
        ## quadratic terms for y
        quad_y = (1. + self.kernel_alpha1*y.pow(2).sum()).pow(-1)
        ## evaluation of kernel function
        ker_eval = quad_x*quad_y*torch.exp(-x_minus_y.pow(2).sum()/2\
                                                  /(self.kernel_std**2))
        ## partial derivative of kernel w.r.t. x
        ker_x = ker_eval*(-2*self.kernel_alpha1*x*quad_x - x_minus_y/self.kernel_std**2)
        ## partial derivative of kernel w.r.t. y
        ker_y = ker_eval*(-2*self.kernel_alpha1*y*quad_y + x_minus_y/self.kernel_std**2)
        ## second derivative w.r.t. x and y // laplace
        ker_xy = ker_eval*(4*(self.kernel_alpha1**2)*quad_x*quad_y*torch.matmul(x, y) +\
                           2*self.kernel_alpha1/(self.kernel_std**2)*quad_y*torch.matmul((x-y), y)-\
                           2*self.kernel_alpha1/(self.kernel_std**2)*quad_x*torch.matmul((x-y), x)-\
                           1/(self.kernel_std**4)*torch.matmul((x-y), (x-y)) + self.dim/(self.kernel_std**2))
        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)
    
    ## score function for sample distribution
    def score_vec(self, x):
        """
        compute the score function for an example
        
        :params[out]: score, the score vector, which represents
            a gradient of the log likelihood evaluated at an input example
        """
        score = -1*x/(self.normal_std**2)
        return score

    ## compute the kernel value
    def kernel(self, x_ind, y_ind):
        """
        evaluate the kernel value given the row indices of
         two training examples
        
        :params[in]: x_ind, int, an index
        :params[in]: y_ind, int, an index
        
        :params[out]: res, a real number
        """
        x, y = self.X[x_ind], self.X[y_ind]
        ker_eval, ker_x, ker_y, ker_xy = self.base_kernel(x, y)
        score_x,score_y = self.score_vals[x_ind],self.score_vals[y_ind]
        ## evalute the kernel
        res = ker_xy+torch.matmul(ker_x, score_y)+torch.matmul(ker_y, score_x)+\
              ker_eval*torch.matmul(score_x, score_y)
        return res.item()
        
    ## compute kernel matrix
    def comp_kernel_matrix(self, nrows=None, ncols=None):
        """
        :params[in]: nrows, ncols, number of rows and columns of association matrix
                     defaults to compute the association of all samples
        
        :params[out]: matrix of association between first nrows and ncols
        """
        if nrows==None:
            nrows=self.sample_size
        if ncols==None:
            ncols=self.sample_size
        matrix = torch.zeros(nrows, ncols) 
        ## association matrix of kernel function
        for i in range(nrows):
            for j in range(ncols):
                matrix[i,j]= self.kernel(i, j)
        ## return a kernel matrix
        return matrix
    
    ## use regularization to make kernel matrix invertible
    def tikhonov(self, matrix, lam=10**(-10)):
        """
        :params[in]: matrix, the square matrix to check 
        :params[in]: lam, lambda value to add
        
        :params[out]: reg_matrix, the regularized matrix
        """
        bad_cond = 1./LA.cond(matrix) < 10**(-15) # machine precision
        while bad_cond:  # bad condition
            matrix = matrix + lam*torch.eye(self.train_size)
            bad_cond = 1./LA.cond(matrix) < 10**(-15)
            lam = lam*10
        ## return regularized matrix and lam
        return matrix, lam
    
## class to compute the gram matrix
class OatesGram(object):
    """
    This is the class to compute Gram matrix of Kernel in Oates paper
    
    """
    def __init__(self, X, score_mat, alpha1, alpha2):
        """
        initialization of class
        :params[in]: X, 2-d np array, each row represents an example
        :params[in]: score_mat, 2-d np array, each row represents score vector of an example
        :params[in]: alpha1, alpha2, positive real, hyper-parameters in base kernel
        """
        sq_euclid = np.einsum('ij->i', np.power(X, 2)) ## squared euclidean dist.
        self.scale_vec = 1./(sq_euclid*alpha1 + 1.)     ## scale vectors of X
        self.scale_mat = np.einsum('i,j->ij', self.scale_vec, self.scale_vec)   ## scale matrix of base kernel
        exp_mat = np.exp(-cdist(X, X, 'sqeuclidean')/np.square(alpha2)/2.) ## exponential term
        self.base_ker_mat = self.scale_mat*exp_mat  ## element wise product, base kernel evaluation
        self.X,self.score_mat,self.alpha1,self.alpha2 = X, score_mat, alpha1, alpha2 
        self.x_minus_y = X[:, np.newaxis] - X   ## x-y in 3-d array, where (i,j)-th element (a vector) is i-th row - j-th

    def grad_x_score_y(self):
        """
        compute gradient of kernel function wrt x, inner product with score matrix
        
        """
        ## scaled i-th row of X inner product j-th row of score matrix
        scale_x_score_y = np.einsum('i,ik,jk->ij', self.scale_vec, self.X, self.score_mat)  
        ## (i,j)-th vector inner product j-th row of score
        x_y_score_y = np.einsum('ijk,jk->ij', self.x_minus_y, self.score_mat) 
        res = -self.base_ker_mat*(2.*self.alpha1*scale_x_score_y+x_y_score_y/np.square(self.alpha2))
        return res

    def grad_y_score_x(self):
        """
        compute gradient of kernel function wrt y, inner product with score matrix
        
        """
        ## scaled i-th row of X inner product j-th row of score matrix
        scale_y_score_x = np.einsum('ik,jk,j->ij', self.score_mat, self.X, self.scale_vec)  
        ## (i,j)-th vector inner product j-th row of score
        x_y_score_x = np.einsum('ijk,ik->ij', self.x_minus_y, self.score_mat) 
        res = -self.base_ker_mat*(2.*self.alpha1*scale_y_score_x-x_y_score_x/np.square(self.alpha2))
        return res
    
    def ker_score_xy(self):
        """
        compute kernel function times inner product of score vectors
        
        """
        ## inner product of i-th and j-th row of score matrix
        score_xy = np.einsum('ik,jk->ij', self.score_mat, self.score_mat)  
        ## elem ent wise matrix multiplication
        ker_score = self.base_ker_mat*score_xy
        return ker_score
    
    def ker_grad_xy(self):
        """
        compute gradient of kernel function wrt x,y
        
        """
        ## scaled inner product of i-th and j-th row of input data matrix
        scale_xy = np.einsum('i,ik,jk,j->ij', self.scale_vec,self.X, self.X,self.scale_vec)  
        ## scaled x inner prod x-y
        scale_x_x_y = np.einsum('i,ik,ijk->ij', self.scale_vec,self.X, self.x_minus_y) 
        ## scaled y inner prod x-y
        scale_y_x_y = np.einsum('j,ijk,jk->ij', self.scale_vec, self.x_minus_y,self.X) 
        ## x-y squared euclidean
        sq_x_y = np.einsum('ijk,ijk->ij', self.x_minus_y,self.x_minus_y)
        ## dimension of input matrix
        dim = self.X.shape[1]
        ## the second derivative
        res = self.base_ker_mat*(4*np.square(self.alpha1)*scale_xy -2*self.alpha1/np.square(self.alpha2)*
                                 scale_x_x_y + 2*self.alpha1/np.square(self.alpha2)*scale_y_x_y - 
                                 sq_x_y/np.power(self.alpha2, 4) + dim/np.square(self.alpha2))
        return res    

    def gram_matrix(self):
        """
        evaluate the gram matrix of a kernel
        """
        res = self.grad_x_score_y() + self.grad_y_score_x() +self.ker_score_xy()+self.ker_grad_xy()
        return res

## class to compute the gram matrix of Gaussian kernel -- squared exponentiated kernel
class GaussianGram(object):
    """
    This is the class to compute Gram matrix of the standard Gaussian Kernel
    
    """
    def __init__(self, X, score_mat, lengthscale, sigma):
        """
        initialization of class
        :params[in]: X, 2-d np array, each row represents an example
        :params[in]: score_mat, 2-d np array, each row represents score vector of an example
        :params[in]: lenthscale, sigma, positive real, hyper-parameters in base kernel
        """
        self.X,self.score_mat,self.lenthscale,self.sigma = X, score_mat, lengthscale, sigma 
        
    def kernel_gaussian_gram_and_derivatives(self):
        """
        :params[in], data, 2D input, number of samples by dimension of input
        :params[in]: l, real, lengthscale
        """
        data = np.atleast_2d(self.X)
        n_data,d_data = data.shape
        r = cdist(XA=data, XB=data, metric='sqeuclidean')
        l = self.lenthscale
        K = np.exp(-r/(2.*(l**2)))*(self.sigma**2)
        #K_extended = K[np.newaxis,:,:]
        y_minus_x = (data[np.newaxis,:]-data[:,np.newaxis]) ## 3-dimensional with (i,j) element j-th row - i-th
        #diff_extended = np.swapaxes(diff_extended,0,2)
        #diff_diffT = np.einsum('inm,jnm->ijnm',diff_extended,diff_extended)
        gradx_K = np.einsum('ij,ijk->ijk', K, y_minus_x)/(l**2) ## 3-dim array
        #gradx_K = -diff_extended*K_extended/(l**2)
        grady_K = -gradx_K
        tmp2 = d_data/(l**2) - r/(l**4)  ## n_data by n_data matrix
        gradxgrady_K = np.einsum('ij,ij->ij', K, tmp2)
        return (K, gradx_K, grady_K, gradxgrady_K)

    def gram_matrix(self):
        """
        evaluate the gram matrix of a kernel
        """
        ## components to compute gram
        K, gradx_K, grady_K, gradxgrady_K = self.kernel_gaussian_gram_and_derivatives()
        part1 = np.einsum('ijk,jk->ij', gradx_K, self.score_mat)
        part2 = np.einsum('ijk,jk->ij', grady_K, self.score_mat)
        part3 = np.einsum('ij,ik,jk->ij', K, self.score_mat, self.score_mat)
        res = part1 + part2 + part3 + gradxgrady_K
        return res

    
class GaussianControlVariate(torch.nn.Module):
    """
    initialization: K(x,y) = lambda^2*exp(-|x-y|_{2}^2/2*l^2)
     suppose u(x) = interpolant of Gaussian kernels
     use median heuristic function to choose the length scale
     This class only allows stochastic gradient descent training
    """
    def __init__(self, input_data, Y, gram_matrix, gram_time, sigma_sq=.0, train_pct=.95):
        """
        :params[in]: input_data, a 2-d tensor, the input data for Gaussian process, 
            each row represents an input example
        :params[in]: Y, a 1-d tensor, each element represents a response for a training
            example
        :params[in]: gram_matrix, gram_matrix of whole dataset
        :params[in]: gram_time, time to compute whole gram_matrix
        :params[in]: sigma_sq, positive real number in kernel
        :params[in]: train_pct, default .95, the percent of data used as training set
        """
        # initialize torch.nn.Module
        super().__init__() 
        # gram matrix of train data set, i.e., association matrix between training examples
        self.gram_matrix = torch.FloatTensor(gram_matrix) + sigma_sq
        self.gram_time = gram_time     ## time to compute the whole gram matrix
        self.train_time = gram_time*train_pct*train_pct  ## time to compute the gram_matrix for training data
        # sample size
        self.sample_size = self.gram_matrix.size(0)
        self.train_pct = train_pct
        self.X = input_data
        self.sigma_sq = sigma_sq     ## constant to adjust the Gram matrix
        self.train_size = int(self.sample_size*train_pct)
        # the matrix to forward, each row represents the association between an example
        # with examples in the training set
        self.forward_matrix = self.gram_matrix[:, 0:(self.train_size)] 
        # weights of kernel functions
        self.w = torch.nn.Parameter(torch.zeros(self.train_size), requires_grad=True) 
        # intercept initial value to the mean of response
        self.b = torch.nn.Parameter(Y.mean(), requires_grad=True) 

    ## use regularization to make kernel matrix invertible
    def tikhonov(self, matrix, lam=10**(-10)):
        """
        :params[in]: matrix, the square matrix to check 
        :params[in]: lam, lambda value to add
        
        :params[out]: reg_matrix, the regularized matrix
        """
        bad_cond = 1./LA.cond(matrix) < 10**(-15) # machine precision
        while bad_cond:  # bad condition
            matrix = matrix + lam*torch.eye(self.train_size)
            bad_cond = 1./LA.cond(matrix) < 10**(-15)
            lam = lam*10
        ## return regularized matrix and lam
        return matrix, lam
        
    ## forward with stein's first order operator
    def forward(self, x, index):
        """
        forward function
        :params[in]: x, the input tensor, one row of input data
        :params[in]: index, an integer, index of traning example to retrieve derivatives
        :params[in]: forward_matrix, 2-d tensor, association matrix of all samples with 
                     training samples, number of rows and columns are sample size and training size
        
        :params[out]: y_pred, the predicted value
        """
        # predicted y
        y_pred = torch.matmul(self.w, self.forward_matrix[index]) + self.b
        return y_pred
    
    ## mini-batch stochastic gradient
    def minibatch(self, x_batch, indices):
        """
        evaluate the forward functions on a batch of examples
        
        :params[in]: x_batch, a 2-d tensor of which each row is a training example
        :params[in]: indices, a list of indices
        :params[in]: model, an integer, indicates which model to use, 1 is for first order operator,
                     2 for the second order operator
        :params[in]: forward_matrix, association matrix of all samples with training samples
        
        :params[out]: y_pred, a 1-d tensor of predicted results
        """
        ## second order operator
        res = [self.forward(x, ind) for x,ind in zip(x_batch, indices)]
        ## stack a list of scalar tensors into a 1-d tensor
        y_pred = torch.stack(res)
        return y_pred
   
    ## split an iterable of items into batches
    def chunks(self, ls, batch_size):
        """
        Yield successive n-sized chunks from l.
        
        :params[in]: ls, an iterable of items
        :params[in]: batch_size, an integer, batch size
        
        returns a generator
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]
    
    ## train and compute control variates with Adam for Gaussian Kernel
    def train(self, Y, integral, lr, gamma, clip_value=5.0e+4, \
                    valid_perc=.05, batch_size=4, N_epochs=[10,20,30,40,50], step_size=10, weight_decay=0.0):
        """
        Train the control variates model and estimate the integral, save the error at different
        training epochs
        
        :params[in]: Y, 1-d tensor, response for inputs 
        :params[in]: gamma, the decreasing multiplier in linear learning rate scheduler
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the mini-batch size
        :params[in]: N_epochs, list of numbers of iterations to monitor
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: threshold, early stopping threshold, the level below which to stop
        :params[in]: weight_decay, the penalty coefficients on L2 norm

        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        start_time = time.time()                 # starting time 
        ## initialize an optimizer
        ##optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        ## warm up scheduler
        ##warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        ## number of training examples, validation examples-> n_train:(n_train+n_valid)  
        n_valid = int(self.sample_size*valid_perc)      
        ## errors and computing times to record
        cv_errs,comp_times = [],[]
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        ## epoch_optimal, err_optimal = 0,-100.0 # optimal epochs and negative error
        train_indexes = list(range(self.train_size))    # indices of training examples
        ## train the model over many epochs
        for i in range(max(N_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err, abs_test_err= [], []   # initialize absolute error of train/valid sets       
            for batch in batches:                 # mini-batch training
                #if (len(batch)<.25*batch_size):  # a batch has too few training examples
                #    continue
                x, y = self.X[batch], Y[batch]    # select a batch
                ## initialize gradients
                optimizer.zero_grad()
                ## evaluate a batch of training examples
                y_pred = self.minibatch(x, batch)
                ## pdb.set_trace()  ## check size of y_pred and y
                ## loss function
                loss = (y_pred - y).pow(2).mean()+ weight_decay*self.w.pow(2).sum()         
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                 # update once 
            ## after each epoch of training, compute the absolute error on the training set
            abs_train_err = [(self.forward(self.X[ind0], ind0) - Y[ind0]).abs().item() \
                             for ind0 in range(0,  self.train_size)]
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(self.X[ind0], ind0) - Y[ind0]).abs().item() \
                        for ind0 in range(self.train_size, self.sample_size)]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    #optimal_par = self.w.clone().detach()
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                if lst_train_err > np.mean(abs_train_err):
                    lst_train_err = np.mean(abs_train_err)
                    ## estimate of integral
                    #cv_est = self.b.item() + self.w.data.sum().item()*self.sigma_sq                   
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                      'best train loss:',lst_train_err)
            ## schedulers
            scheduler.step()                     # schedule learning rate
            #warmup_scheduler.dampen()
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimate errors at some epochs
            if (i+1) in N_epochs:
                if self.train_pct < 1.0:         # use part of data to train the model
                    optimal_par = self.w.clone().detach()
                    err0, err1 = self.comp_errors(Y, integral, optimal_par)
                    comp_time = time.time()-start_time+self.train_time # record the computational time
                    comp_times.append(comp_time)
                    cv_errs.append(err1)                    
                else:                            # use full data to train the model    
                    ## evaluate control variates
                    cv_est = self.b.item() + self.w.data.sum().item()*self.sigma_sq   
                    err1 = cv_est - integral
                    comp_time = time.time() - start_time + self.train_time    # computational time
                    comp_times.append(comp_time)
                    cv_errs.append(err1)                    
        ## after training, record mean absolute errors on the training and validation set
        ## train_err, test_err = np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        return (cv_errs, comp_times)        

    ## train and compute control variates by minimizing the sample variance
    def train_var(self, Y, integral, lr, gamma, clip_value=5.0e+4, \
                    valid_perc=.05, batch_size=4, N_epochs=[10,20,30,40,50], step_size=10, weight_decay=0.0):
        """
        Train the control variates model and estimate the integral, save the error at different
        training epochs
        
        :params[in]: Y, 1-d tensor, response for inputs 
        :params[in]: gamma, the decreasing multiplier in linear learning rate scheduler
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the mini-batch size
        :params[in]: N_epochs, list of numbers of iterations to monitor
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: threshold, early stopping threshold, the level below which to stop
        :params[in]: weight_decay, the penalty coefficients on L2 norm

        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        self.b.requires_grad=False               # set gradient of constant self.b to zero
        start_time = time.time()                 # starting time 
        ## compute forward_matrix of associations
        #forward_matrix = self.comp_kernel_matrix(self.sample_size, self.train_size)
        ## initialize an optimizer
        ##optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## warm up scheduler
        ##warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        ## number of training examples, validation examples-> n_train:(n_train+n_valid)  
        n_valid = int(self.sample_size*valid_perc)      
        ## errors and computing times to record
        cv_errs,comp_times = [],[]
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        ## epoch_optimal, err_optimal = 0,-100.0 # optimal epochs and negative error
        train_indexes = list(range(self.train_size))    # indices of training examples
        ## train the model over many epochs
        for i in range(max(N_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err, abs_test_err= [], []   # initialize absolute error of train/valid sets       
            for batch in batches:                 # mini-batch training
                if (len(batch)<2):    # a batch has too few (<2) training examples
                    continue
                x, y = self.X[batch], Y[batch]    # select a batch
                ## initialize gradients
                optimizer.zero_grad()
                ## evaluate a batch of training examples
                y_pred = self.minibatch(x, batch)
                ## pdb.set_trace() 
                ## loss function
                loss = (y_pred - y).var()+ weight_decay*self.w.pow(2).sum()         
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                 # update once 
            ## after each epoch of training, compute the absolute error on the training set
            train_err = [(Y[ind0]-self.forward(self.X[ind0], ind0)).item() \
                             for ind0 in range(0,  self.train_size)]
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(self.X[ind0], ind0) - Y[ind0]).abs().item() \
                        for ind0 in range(self.train_size, self.sample_size)]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    #optimal_par = self.w.clone().detach()
                print(i,'-th Epoch, training loss: ', np.abs(train_err).mean(), \
                       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                if lst_train_err > np.abs(train_err).mean():
                    lst_train_err = np.abs(train_err).mean()
                    ## estimate of integral
                    #cv_est = self.b.item() + self.w.data.sum().item()*self.sigma_sq                    
                print(i,'-th Epoch, training loss: ', np.abs(train_err).mean(), \
                      'best train loss:',lst_train_err)
            ## schedulers
            scheduler.step()                     # schedule learning rate
            #warmup_scheduler.dampen()
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimate errors at some epochs
            if (i+1) in N_epochs:
                if self.train_pct < 1.0:         # use part of data to train the model
                    optimal_par = self.w.clone().detach()
                    err0, err1 = self.comp_errors(Y, integral, optimal_par)
                    comp_time = time.time()-start_time +self.train_time # record the computational time
                    comp_times.append(comp_time)
                    cv_errs.append(err1)                    
                else:                            # use full data to train the model    
                    ## evaluate control variates
                    cv_est = self.b.item() + self.w.data.sum().item()*self.sigma_sq
                    err1 = cv_est - integral
                    comp_time = time.time() - start_time +self.train_time    # computational time
                    comp_times.append(comp_time)
                    cv_errs.append(err1)                    
        ## after training, record mean absolute errors on the training and validation set
        ## train_err, test_err = np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        return (cv_errs, comp_times)        

    ## compute errors after training period
    def comp_errors(self, Y, integral, beta_hat):
        """
        After model training is finished,
        Compute the estimates of the integral and errors from Monte Carlo and control variates

        :params[in]: Y, a 1-d tensor, output
        :params[in]: integral, real, the real value of integral
        :params[in]: beta_hat, 1-d tensor, coefficient of linear interpolant
        :params[in]: forward_matrix, association matrix of all samples with training samples

        :params[out]: err0, real, the Monte Carlo error
        :params[out]: err1, real, the control variates error only using test set, no training example             
        """  
        err0 = Y.mean().item() - integral                     # Monte Carlo error 
        ## find the predicted values of the control variates    
        cv_eval = []
        for i1 in range(self.train_size, self.sample_size):
            temp = self.forward_matrix[i1]
            cv_eval.append(torch.matmul(temp, beta_hat).item())
        cv_estimates = Y[self.train_size:self.sample_size].numpy()-np.array(cv_eval)+(beta_hat.sum()*self.sigma_sq).item()
        ## compute estimates from the control variates    
        cv_mean = np.mean(cv_estimates)
        ## full_est = np.mean(cv_estimates)                   # estimates using whole sample
        ## compute errors     
        err1 = cv_mean - integral                             # error using test data set
        ##err2 = full_est - integral                          # error using the whole data set
        return (err0, err1) 
    
        
    ## classical way to find kernel control variates
    def classical_way(self, Y, integral):
        """
        classical approach to estimate parameters -- using the method in oates' paper
        
        :params[in]: Y, the evaluation of integrand at X
        :params[in]: integral, real, the real value of integral
        :params[in]: train_perc, the percent of samples used in training period
        
        :params[out]: err0, monte carlo error 
        :params[out]: err1, error from control variates 
        :params[out]: err2, error from CV with full dataset
        :params[out]: comp_time, time to compute
        """
        start_time = time.time()                              # start to time
        ## compute the association matrix -- take account of the time 
        kernel_matrix = self.gram_matrix - self.sigma_sq
        ## regularized kernel matrix for training data
        kernel_train, reg_lam = self.tikhonov(kernel_matrix[:self.train_size,\
                                                            :self.train_size])
        ## inverse matrix for training sample
        inv_train = kernel_train.inverse()
        Y_train = Y[:self.train_size]                         # train data set
        Y_test = Y[self.train_size:]                          # test data response
        test_size = Y_test.size()[0]
        ## estimate of test responses
        kernel10 = kernel_matrix[self.train_size:, :self.train_size]
        ## first component in nominator in Page 703 of Oates paper
        comp1 = torch.matmul(torch.ones(1,self.train_size),\
                             torch.matmul(inv_train, Y_train)).item()
        ## denominator
        comp2 = torch.matmul(torch.ones(1,self.train_size),torch.matmul(inv_train,\
                                            torch.ones(self.train_size, 1))).item()
        ## constant in f(x)=c+beta*kernel
        const = comp1/(1.+comp2)
        if self.train_pct ==1.0:
            ## estimate of integral
            cv_est = const
            ## evaluate control variates
            err0 = Y.mean().item() - integral                   # Monte Carlo error 
            err1 = cv_est - integral                            # error from control variates
            comp_time = time.time()-start_time+self.gram_time   # computational time
            return (err0, err1, comp_time)
        ## frequent component
        comp3 = torch.matmul(kernel10, inv_train)
        ## estimate of beta
        beta_hat = torch.matmul(inv_train, (Y_train - const*torch.ones(self.train_size)))
        ## estimate of test data
        test_hat = torch.matmul(comp3, Y_train)+(torch.ones(test_size)-\
                        torch.matmul(comp3, torch.ones(self.train_size)))*const
        ## estimate of integral
        cv_est = (Y_test-test_hat).mean()+const
        ## evaluate control variates
        err0 = Y.mean().item() - integral                     # Monte Carlo error 
        err1 = cv_est.item() - integral
        comp_time = time.time()-start_time+self.gram_time     # computational time
        #pdb.set_trace()
        return (err0, err1, comp_time)

