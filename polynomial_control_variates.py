# load modules
import torch
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy import stats
#%matplotlib inline
from scipy import special
import time,pickle
from torch.autograd import Variable
import pdb
from numpy import linalg as LA

# prepare the simulation: define the polynomial integrand
# compute the integration and the gradient
class Prepare_Poly(object):
    def __init__(self, Alpha, sigma):
        """
        initialization
        :params[in], Alpha, two-dimensional torch array, i.e., matrix
        :params[in], sigma, the standard deviation of Gaussian distr.
        """
        self.Alpha = Alpha  # matrix of constants in the integrand
        self.dim, self.order = Alpha.size() # dimension of x, and the highest order of polynomial
        self.sigma = sigma
        
    def vect_power_matrix(self, x):
        """
        for each input 1-d tensor x, find the matrix of its powers
        
        :params[in], x, a d-dimensional 1-d tensor
        
        :params[out], res, a 2-d tensor, each column is x^i
        """
        powers = [x.pow(i) for i in range(self.order)]
        res = torch.stack(powers, dim=1) # dim * order 2-d tensor
        return res
        
    def integrand(self, x):
        """
        evaluate the integrand function
        
        :params[in], x, a one-dimensional tensor 
        
        :params[out], res, a real number
        """
        pow_mat = self.vect_power_matrix(x) # power matrix
        hadam = self.Alpha * pow_mat  # hadamard product
        res = torch.sum(hadam, dim=1).sum().item() # evaluation
        return res
    
    def integral(self):
        """
        compute the real value of the integral
        
        :params[out], res, a real number
        """
         # values of right part of Eq. 10
        vals = [(self.sigma**i)*special.factorial2(i-1).item() if i%2==0 \
                else 0 for i in range(0, self.order)]
        prod_mat_vec = torch.matmul(self.Alpha, torch.tensor(vals))
        res = prod_mat_vec.sum().item()
        return res
    
    def score_func(self, x):
        """
        evaluate the score function, i.e., gradient of log density
        
        :params[in], x, a one-dimensional tensor 
        
        :params[out], score, a 1-d tensor
        """
        score = -1*x/(self.sigma**2)
        return score
    
    def sample(self, n, random_seed=0):
        """
        draw a sample from the multivariate Gaussian distr.
        
        :params[in], n, an integer, sample size
        
        :params[out], samp, a n-by-d matrix tensor, of which each row represents a sample
        """
        torch.manual_seed(random_seed)
        samp = self.sigma*torch.randn(n, self.dim)
        return samp


class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't 
    improve after a given patience.
    """
    def __init__(self, patience=5):
        """
        initialize this class
        
        :params[in]: patience, int, How long to wait after last time 
            validation loss improved. Default: 5
            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        """
        this method makes the instances of this class a function
        
        :params[in]: val_loss, real, the validation loss from some epoch
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score: # validation loss increases
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:                         # validation loss decreases
            self.best_score = score
            self.counter = 0



## class: Polynomial Control Variates-- quadratic polynomial (of order 2) by
## using Stein's second order operator with quadratic polynomial control variates
## polynomial family: P(x)=1/2*x^{T}wx + x^{T}a 
class SteinSecondOrderQuadPolyCV(torch.nn.Module): # inherit torch.nn.Module
    
    def __init__(self, D_in, score_matrix):
        """
        initialization
        
        :params[in]: D_in, dimension of input to the integrand
        :params[in]: score_matrix, 2 dimensional tensor, n by d, the score matrix
        """
        super().__init__() # initialize the superclass
        # initialize second order matrix
        self.w = torch.nn.Parameter(torch.zeros(D_in, D_in), requires_grad=True) 
        # initialize first order array of parameters
        self.a = torch.nn.Parameter(torch.zeros(D_in), requires_grad=True)
        # intercept initial value = 0
        self.b = torch.nn.Parameter(torch.tensor(.0), requires_grad=True) 
        self.score_matrix = score_matrix # score matrix of prob. measure
        
    def poly_function(self, x):
        """
        conpute the gradient and second order derivatives of polynomial functions
        :params[in]: x, an input example, a 1-d tensor
        
        :params[out]: grad, a one-d tensor
        :params[out]: sec, a scalar, laplace operator on polynomial
        """
        grad = torch.matmul(self.w, x) + self.a  # 1st order gradient
        sec = torch.trace(self.w)                # 2nd order laplace operator
        return grad, sec
    
        
    ## forward
    def forward(self, x, ind):
        """
        forward function
        :params[in]: x, 1-d input tensor
        :params[in]: ind, int, row index of score matrix
        
        :params[out]: y_pred, the predicted value
        """
        # predicted y
        score = self.score_matrix[ind]
        grad, sec = self.poly_function(x)
        y_pred = torch.matmul(grad, score) + sec + self.b
        return y_pred
    
    ## mini-batch stochastic gradient
    def minibatch(self, x_batch, indices):
        """
        evaluate the forward functions on a batch of examples
        
        :params[in]: x_batch, a 2-d tensor of which each row is a training example
        :params[in]: indices, list of index, row indexes of score matrix
        
        :params[out]: y_pred, a 1-d tensor of predicted results
        """
        # first order operator
        res = [self.forward(x, ind) for x,ind in zip(x_batch, indices)]
        y_pred = torch.stack(res)
        return y_pred
   
    ## split an iterable of items into batches
    def chunks(self, ls, batch_size):
        """
        Yield successive n-sized chunks from ls, an iterable.
        
        :params[in]: ls, an iterable of items
        :params[in]: batch_size, an integer, batch size
        
        returns a generator
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]
    
    
    ## train and compute control variates with Adam
    def train(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, \
              step_size=10, weight_decay = 0.0, batch_size = 1, result_epochs=[10,20,30,40], norm_init_std=1.e-4):
        """
        Train the control variates model and estimate the integral
        
        :params[in]: X, 2-d tensor, input, each row represent an example
        :params[in]: Y, 1-d tensor, output 
        :params[in]: l_penalty, the penalty parameter
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the batch size, for now it is fixed at 1
        :params[in]: N_epochs, number of iterations
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: valid_perc, a real between 0 and 1, the percent of sample for validation
        :params[in]: gamma, real, learning rate decay rate
        :params[in]: lr, real number, initial learning rate\
        :params[in]: result_epochs, list of integers, epochs at which evaluates estimates
        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        #ES = EarlyStopping(5)                   # early stopping
        self.b.data = Y.mean()
        for each_par in self.parameters():   ## initialize this neural nets
            if each_par is self.b:
                continue
            torch.nn.init.normal_(each_par, mean=0, std=norm_init_std)
        ## use adam optimizer
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ## SGD optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        valid_perc=1. - train_perc  ## validation percent
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        #epoch_optimal, err_optimal = 0, -100.0 # optimal epochs and negative error
        train_indexes = np.arange(n_train)      # indices of training examples
        start_time = time.time()                # starting time 
        ## quantities  to record
        mc_errs, cv_errs, comp_times = [],[],[]
        ## training the model many epochs
        for i in range(max(result_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err,abs_test_err = [],[]   # initialize absolute error of train/valid set       
            for batch in batches:                # mini-batch training
                if (len(batch)<.25*batch_size):  # a batch has too few training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                #pdb.set_trace()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, batch)
                l1norm = self.w.norm(p=1)+self.a.norm(p=1)
                # produce output 
                loss = (y_pred - y).pow(2).mean()+weight_decay*l1norm     
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            abs_train_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                             range(0,  n_train)]         
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    #optimal_par = self.w.clone().detach(),self.a.clone().detach()
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                if lst_train_err > np.mean(abs_train_err):
                    lst_train_err = np.mean(abs_train_err)
                    ## estimate of integral
                    #cv_est = self.b.item()                    
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                      'best train loss:',lst_train_err)
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## compute the estimates during training period
            if (i+1) in result_epochs:
                if train_perc<1.0:   ## part data for training
                    optimal_par = self.w.clone().detach(),self.a.clone().detach()
                    err0, err1 = self.comp_errors(X, Y, integral, n_train, nsamples, optimal_par)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:     ## all data for training
                    err0 = Y.mean().item() - integral
                    err1 = self.b.item() - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        ## for debugging purpose, return the evaluated control variates
        #cv_evaluations = [self.forward(X[ind0], ind0).item() for ind0 in range(nsamples)]
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        

    ## train and compute control variates by minimizing variance of residuals
    def train_var(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, valid_perc=.1,\
              step_size=10, weight_decay = 0.0, batch_size = 1, result_epochs=[10,20,30,40]):
        """
        Train the control variates model and estimate the integral// set constant part to zero, 
        without updating permanently
        
        :params[in]: X, 2-d tensor, input, each row represent an example
        :params[in]: Y, 1-d tensor, output 
        :params[in]: l_penalty, the penalty parameter
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the batch size, for now it is fixed at 1
        :params[in]: N_epochs, number of iterations
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: valid_perc, a real between 0 and 1, the percent of sample for validation
        :params[in]: gamma, real, learning rate decay rate
        :params[in]: lr, real number, initial learning rate\
        :params[in]: result_epochs, list of integers, epochs at which evaluates estimates
        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        self.b.requires_grad=False             # set gradient of constant self.b to zero
        ## use adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ## SGD optimizer
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        #epoch_optimal, err_optimal = 0, -100.0 # optimal epochs and negative error
        train_indexes = np.arange(n_train)      # indices of training examples
        dict_iters = {}                         # dictionary of iterations
        start_time = time.time()                # starting time 
        ## quantities  to record
        mc_errs, cv_errs, comp_times = [],[],[]
        ## training the model many epochs
        for i in range(max(result_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err,abs_test_err = [],[]   # initialize absolute error of train/valid set       
            for batch in batches:                # mini-batch training
                if (len(batch)<2):               # each batch has >=2 training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                #pdb.set_trace()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, batch)
                l1norm = self.w.norm(p=1)+self.a.norm(p=1)
                # produce output 
                loss = (y_pred - y).var()+weight_decay*l1norm     
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            abs_train_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                             range(0,  n_train)]         
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    #optimal_par = self.w.clone().detach(),self.a.clone().detach()
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                #print('Use all data for training')
                if lst_train_err > np.mean(abs_train_err):
                    lst_train_err = np.mean(abs_train_err)
                    ## estimate of integral by average error
                    #cv_est = np.mean([(Y[ind0]-self.forward(X[ind0], ind0)).item() for ind0 in \
                    #        range(n_train)])                    
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #      'best train loss:',lst_train_err)
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimates of parameters after each epoch
            #dict_iters[i] = [list(self.parameters())[1].clone().detach(),\
            #                 list(self.parameters())[2].clone().detach()]
            ## compute the estimates during training period
            if (i+1) in result_epochs:
                if train_perc<1.0:   ## part data for training
                    optimal_par = self.w.clone().detach(),self.a.clone().detach()
                    err0, err1 = self.comp_errors(X, Y, integral, n_train, nsamples, optimal_par)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:     ## all data for training
                    err0 = Y.mean().item() - integral
                    cv_est = np.mean([(Y[ind0]-self.forward(X[ind0], ind0)).item() for ind0 in \
                            range(n_train)])    
                    err1 = cv_est - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        ## for debugging purpose, return the evaluated control variates
        #cv_evaluations = [self.forward(X[ind0], ind0).item() for ind0 in range(nsamples)]
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        

    ## compute the estimates of integrals after training period 
    def comp_errors(self, X, Y, integral, n_train, nsamples, parameters):
        """
        After model training is finished,
        Compute the estimates of the integral and errors from Monte Carlo and control variates

        :params[in]: X, a 2-d input data
        :params[in]: Y, a 1-d tensor, output
        :params[in]: integral, real, the real value of integral
        :params[in]: n_train, int, the training sample size
        :params[in]: nsamples, int, the whole sample size
        :params[in]: parameters, tuple, weight matrix and vector trained by train set

        :params[out]: err0, real, the Monte Carlo error
        :params[out]: err1, real, the control variates error only using test set, 
                       no training example        
        :params[out]: err2, real, the control variates error using total sample       
        """  
        err0 = Y.mean().item() - integral                     # Monte Carlo error 
        ## find the predicted values from the control variates with  poly_value(x, ind, parameters)  
        cv_eval = [(self.forward(X[i1], i1)-self.b).item() for i1 in range(nsamples)]
        #cv_train_mean = np.mean(cv_eval[:n_train])            # mean of control variates on train set  
        cv_estimates = Y.numpy() - np.array(cv_eval)
        #cv_estimates1 = Y.numpy() - cv_eval + cv_train_mean  # another way
        ## compute estimates from the control variates    
        cv_mean = np.mean(cv_estimates[n_train:nsamples])     # estimate do not use training sample
        #full_est = np.mean(cv_estimates)                      # estimates using whole sample
        #cv_mean1 = np.mean(cv_estimates1[n_train:nsamples])   # estimate of using training mean
        ## compute errors     
        err1 = cv_mean - integral                             # error using test data set
        return (err0, err1)
    
    ## use regularization to make kernel matrix invertible
    def tikhonov(self, matrix, lam=10**(-10)):
        """
        :params[in]: matrix, the square matrix to check 
        :params[in]: lam, lambda value to add
        
        :params[out]: reg_matrix, the regularized matrix
        """
        bad_cond = 1./LA.cond(matrix) < 10**(-15) # machine precision
        while bad_cond:  # bad condition
            matrix = matrix + lam*torch.eye(matrix.shape[0])
            bad_cond = 1./LA.cond(matrix) < 10**(-15)
            lam = lam*10
        ## return regularized matrix and lam
        return matrix

    
    ## Least square finding coefficients of linear control variates
    def linear_coef(self, X_train, Y_train):
        """
        compute the coefficients in linear control variates without intercept
        
        :params[in]: X_train, a 2-d tensor, input training data of explanatory variables,
            each row indicates a training example, each column indicates a variable
        :params[in]: Y, a 1-d tensor, and its length equal to number of rows in X_train
        
        :params[out]: coef, real, the coefficients of linear regression
        """
        x_dim = X_train.size()[1]                             # input dimension 
        ## attach response y as the last variable
        score_resp = torch.cat((X_train, Y_train.unsqueeze(dim=1)), dim=1)
        ## covariance matrix between design matrix and response  
        cov_score_y = torch.tensor(np.cov(score_resp, rowvar=False))
        cov_score, score_y = cov_score_y[:x_dim,:x_dim],cov_score_y[:x_dim,x_dim]
        coef = torch.matmul(cov_score.inverse(), score_y)     # coefficients
        return coef.float()
    
    ## prepare explanatory variables for each training example Quadratic CV
    def prepare_feature(self, i, x):
        """
        prepare feature vectors for each training example prior to
            computing coefficients in quadratic control variates
            
        :params[in]: x, a 1-d tensor, training data that represents an observation
        :params[in]: i, an integer, row index of the score matrix
        
        :params[out]: x_features, a 1-d tensor, explanatory variables in feature space
        """
        ## score at x, i.e., gradient of log density
        score_x = self.score_matrix[i]
        ## data for diagonal entries
        diag_par = 1.+x*score_x
        ## compute matrix multiplication
        x_prod_score = torch.matmul(x.unsqueeze(dim=1), score_x.unsqueeze(dim=0))
        ## length of x
        len_x = len(x)
        ## lower indices no diagonal
        low_inds = np.tril_indices(len_x, -1)
        ## list of elements
        elements = [(x_prod_score[i,j]+x_prod_score[j,i]).item() for \
                    i,j in zip(low_inds[0], low_inds[1])]
        ## concatenate these tensors
        x_features = torch.cat((score_x, diag_par, torch.tensor(elements)))
        return x_features
    
    ## classical way to find QUADRATIC control variates
    def classical_way(self, X, Y, integral, train_perc=.3):
        """
        classical approach to estimate parameters
        
        :params[in]: X, input design matrix, a 2-d torch array
        :params[in]: Y, the evaluation of integrand at X
        :params[in]: integral, real, the real value of integral
        :params[in]: train_perc, the percent of samples used in training period
        
        :params[out]: err0, monte carlo error 
        :params[out]: err1, error from control variates 
        :params[out]: err2, error from CV with full dataset
        :params[out]: comp_time, time to compute
        :params[out]: coef, coefficients of linear regression
        """
        start_time = time.time()                              # starting time 
        ## features from input data X, each row is a feature vector for an input
        X_feature = torch.stack([self.prepare_feature(i, x) for i,x in enumerate(X)], dim=0)
        nsamples,n_train = len(Y),int(len(Y)*train_perc)      # number of samples
        X_train,Y_train = X_feature[:n_train],Y[:n_train]     # train data set
        ## call function to compute coefficients
        coef = self.linear_coef(X_train, Y_train)
        ## evaluate control variates
        cv_eval = torch.matmul(X_feature, coef)
        cv_estimates = Y.numpy() - cv_eval.numpy()
        ## estimates of f hat
        cv_hat = cv_eval+Y_train.mean()
        ## compute estimates from the control variates 
        if train_perc<1.0:                                    # part of data for training    
            cv_mean = np.mean(cv_estimates[n_train:nsamples])     # estimate do not use training sample
            ## compute errors     
            err0 = Y.mean().item() - integral                     # monte carlo error
            err1 = cv_mean - integral                             # error using test data set
            #err2 = full_est - integral                           # error using whole data 
            comp_time = time.time() - start_time                  # computational time
        else:   ## all data used in training
            cv_mean = np.mean(cv_estimates)
            ## compute errors     
            err0 = Y.mean().item() - integral                     # monte carlo error
            err1 = cv_mean - integral                             # error using test data set
            #err2 = full_est - integral                           # error using whole data 
            comp_time = time.time() - start_time                  # computational time
        return (err0, err1, comp_time, cv_hat, coef)

## class: Polynomial Control Variates-- quadratic polynomial (of order 1) by
## using Stein's second order operator with linear control variates
## polynomial family: P(x)= x^{T}a 
class SteinFirstOrderQuadPolyCV(torch.nn.Module): # inherit torch.nn.Module
    
    def __init__(self, D_in, score_matrix):
        """
        initialization
        
        :params[in]: D_in, dimension of input to the integrand
        :params[in]: sigma, standard deviation of normal distr.
        """
        super().__init__() # initialize the superclass
        # initialize first order array of parameters
        self.a = torch.nn.Parameter(torch.zeros(D_in), requires_grad=True)
        # intercept initial value = 0
        self.b = torch.nn.Parameter(torch.tensor(.0), requires_grad=True) 
        self.score_matrix = score_matrix # score matrix of prob. measure
        
    def poly_function(self, x):
        """
        conpute the gradient and second order derivatives of polynomial functions
        :params[in]: x, an input example, a 1-d tensor
        
        :params[out]: grad, a one-d tensor
        :params[out]: sec, a scalar, laplace operator on polynomial
        """
        grad = self.a  # 1st order gradient
        sec = 0.                # 2nd order laplace operator
        return grad, sec
    
    def poly_value(self, x, ind, parameters):
        """
        evaluate the value of polynomial given x and parameters
        :params[in]: x, a training example
        :params[in]: ind, index of a training example
        :params[in]: parameters, tuple, weight matrix and vector
        
        :params[out]: val, tensor
        """
        a0 = parameters
        score = self.score_matrix[ind]
        val = torch.matmul(a0, score)
        return val.item()
        
    ## forward
    def forward(self, x, ind):
        """
        forward function
        :params[in]: x, 1-d input tensor
        :params[in]: ind, int, row index of score matrix
        
        :params[out]: y_pred, the predicted value
        """
        # predicted y
        score = self.score_matrix[ind]
        grad, sec = self.poly_function(x)
        y_pred = torch.matmul(grad, score) + sec + self.b
        return y_pred
    
    ## mini-batch stochastic gradient
    def minibatch(self, x_batch, indices):
        """
        evaluate the forward functions on a batch of examples
        
        :params[in]: x_batch, a 2-d tensor of which each row is a training example
        :params[in]: indices, list of index, row indexes of score matrix
        
        :params[out]: y_pred, a 1-d tensor of predicted results
        """
        # first order operator
        res = [self.forward(x, ind) for x,ind in zip(x_batch, indices)]
        y_pred = torch.stack(res)
        return y_pred
   
    ## split an iterable of items into batches
    def chunks(self, ls, batch_size):
        """
        Yield successive n-sized chunks from ls, an iterable.
        
        :params[in]: ls, an iterable of items
        :params[in]: batch_size, an integer, batch size
        
        returns a generator
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]

    ## use regularization to make kernel matrix invertible
    def tikhonov(self, matrix, lam=10**(-10)):
        """
        :params[in]: matrix, the square matrix to check 
        :params[in]: lam, lambda value to add
        
        :params[out]: reg_matrix, the regularized matrix
        """
        bad_cond = 1./LA.cond(matrix) < 10**(-15) # machine precision
        while bad_cond:  # bad condition
            matrix = matrix + lam*torch.eye(matrix.shape[0])
            bad_cond = 1./LA.cond(matrix) < 10**(-15)
            lam = lam*10
        ## return regularized matrix and lam
        return matrix
    
    ## train and compute control variates with Adam
    def train(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, valid_perc=.1,\
              step_size=10, weight_decay = 0.0, batch_size = 1, result_epochs=[10,20,30,40]):
        """
        Train the control variates model and estimate the integral
        
        :params[in]: X, 2-d tensor, input, each row represent an example
        :params[in]: Y, 1-d tensor, output 
        :params[in]: l_penalty, the penalty parameter
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the batch size, for now it is fixed at 1
        :params[in]: N_epochs, number of iterations
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: valid_perc, a real between 0 and 1, the percent of sample for validation
        :params[in]: gamma, real, learning rate decay rate
        :params[in]: lr, real number, initial learning rate\
        :params[in]: result_epochs, list of integers, epochs at which evaluates estimates
        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        #ES = EarlyStopping(5)                   # early stopping
        ## use adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ## SGD optimizer
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        #epoch_optimal, err_optimal = 0, -100.0 # optimal epochs and negative error
        train_indexes = np.arange(n_train)      # indices of training examples
        dict_iters = {}                         # dictionary of iterations
        start_time = time.time()                # starting time 
        ## quantities  to record
        mc_errs, cv_errs, comp_times = [],[],[]
        ## training the model many epochs
        for i in range(max(result_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err,abs_test_err = [],[]   # initialize absolute error of train/valid set       
            for batch in batches:                # mini-batch training
                if (len(batch)<.25*batch_size):  # a batch has too few training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                #pdb.set_trace()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, batch)
                l1norm = self.a.norm(p=1)
                # produce output 
                loss = (y_pred - y).pow(2).mean() + weight_decay*l1norm     
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            abs_train_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                             range(0,  n_train)]         
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    optimal_par = self.a.clone().detach()
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                if lst_train_err > np.mean(abs_train_err):
                    lst_train_err = np.mean(abs_train_err)
                    ## estimate of integral
                    cv_est = self.b.item()                    
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #      'best train loss:',lst_train_err)
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimates of parameters after each epoch
            #dict_iters[i] = [list(self.parameters())[1].clone().detach(),\
            #                 list(self.parameters())[2].clone().detach()]
            ## compute the estimates during training period
            if (i+1) in result_epochs:
                if train_perc<1.0:   ## part data for training
                    err0, err1 = self.comp_errors(X, Y, integral, n_train, nsamples, optimal_par)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:     ## all data for training
                    err0 = Y.mean().item() - integral
                    err1 = self.b.item() - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        ## for debugging purpose, return the evaluated control variates
        #cv_evaluations = [self.forward(X[ind0], ind0).item() for ind0 in range(nsamples)]
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        

    ## train and compute control variates by minimizing variance of residuals
    def train_var(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, valid_perc=.1,\
              step_size=10, weight_decay = 0.0, batch_size = 1, result_epochs=[10,20,30,40]):
        """
        Train the control variates model and estimate the integral// set constant part to zero, 
        without updating permanently
        
        :params[in]: X, 2-d tensor, input, each row represent an example
        :params[in]: Y, 1-d tensor, output 
        :params[in]: l_penalty, the penalty parameter
        :params[in]: integral, the real value of integral
        :params[in]: batch_size, the batch size, for now it is fixed at 1
        :params[in]: N_epochs, number of iterations
        :params[in]: train_perc, a real between 0 and 1, the percent of sample for training
        :params[in]: valid_perc, a real between 0 and 1, the percent of sample for validation
        :params[in]: gamma, real, learning rate decay rate
        :params[in]: lr, real number, initial learning rate\
        :params[in]: result_epochs, list of integers, epochs at which evaluates estimates
        
        :params[out]: err0, the error between average of Y and true integral
        :params[out]: err1, the error of control variates
        """
        self.b.requires_grad=False             # set gradient of constant self.b to zero
        ## use adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ## SGD optimizer
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
        ## least test error, and optimal parameters
        lst_test_err, lst_train_err, optimal_par = np.inf, np.inf, None 
        #epoch_optimal, err_optimal = 0, -100.0 # optimal epochs and negative error
        train_indexes = np.arange(n_train)      # indices of training examples
        dict_iters = {}                         # dictionary of iterations
        start_time = time.time()                # starting time 
        ## quantities  to record
        mc_errs, cv_errs, comp_times = [],[],[]
        ## training the model many epochs
        for i in range(max(result_epochs)):
            ## split indices into batches
            batches = self.chunks(train_indexes, batch_size) 
            abs_train_err,abs_test_err = [],[]   # initialize absolute error of train/valid set       
            for batch in batches:                # mini-batch training
                if (len(batch)<2):               # each batch has >=2 training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                #pdb.set_trace()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, batch)
                l1norm = self.a.norm(p=1)
                # produce output 
                loss = (y_pred - y).var()+weight_decay*l1norm     
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            abs_train_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                             range(0,  n_train)]         
            ## compute the absolute errors on the validation set
            if n_valid > 0:
                abs_test_err = [(self.forward(X[ind0], ind0) - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]   
                if lst_test_err > np.mean(abs_test_err):   ## update optimal parameters till now
                    lst_test_err = np.mean(abs_test_err)
                    optimal_par = self.a.clone().detach()
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #       'current validation loss: ', np.mean(abs_test_err), 'best val loss:',lst_test_err)
            else:
                #print('Use all data for training')
                if lst_train_err > np.mean(abs_train_err):
                    lst_train_err = np.mean(abs_train_err)
                    ## estimate of integral by average error
                    cv_est = np.mean([(Y[ind0]-self.forward(X[ind0], ind0)).item() for ind0 in \
                            range(n_train)])                    
                #print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                #      'best train loss:',lst_train_err)
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimates of parameters after each epoch
            #dict_iters[i] = [list(self.parameters())[1].clone().detach(),\
            #                 list(self.parameters())[2].clone().detach()]
            ## compute the estimates during training period
            if (i+1) in result_epochs:
                if train_perc<1.0:   ## part data for training
                    err0, err1 = self.comp_errors(X, Y, integral, n_train, nsamples, optimal_par)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:     ## all data for training
                    err0 = Y.mean().item() - integral
                    err1 = cv_est - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        ## for debugging purpose, return the evaluated control variates
        #cv_evaluations = [self.forward(X[ind0], ind0).item() for ind0 in range(nsamples)]
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        

    ## compute the estimates of integrals after training period 
    def comp_errors(self, X, Y, integral, n_train, nsamples, parameters):
        """
        After model training is finished,
        Compute the estimates of the integral and errors from Monte Carlo and control variates

        :params[in]: X, a 2-d input data
        :params[in]: Y, a 1-d tensor, output
        :params[in]: integral, real, the real value of integral
        :params[in]: n_train, int, the training sample size
        :params[in]: nsamples, int, the whole sample size
        :params[in]: parameters, tuple, weight matrix and vector trained by train set

        :params[out]: err0, real, the Monte Carlo error
        :params[out]: err1, real, the control variates error only using test set, 
                       no training example        
        :params[out]: err2, real, the control variates error using total sample       
        """  
        err0 = Y.mean().item() - integral                     # Monte Carlo error 
        ## find the predicted values from the control variates with  poly_value(x, ind, parameters)  
        cv_eval = [self.poly_value(X[i1], i1) for i1 in range(nsamples)]
        #cv_train_mean = np.mean(cv_eval[:n_train])            # mean of control variates on train set  
        cv_estimates = Y.numpy() - np.array(cv_eval)
        #cv_estimates1 = Y.numpy() - cv_eval + cv_train_mean  # another way
        ## compute estimates from the control variates    
        cv_mean = np.mean(cv_estimates[n_train:nsamples])     # estimate do not use training sample
        ## compute errors     
        err1 = cv_mean - integral                             # error using test data set
        return (err0, err1)
    
    ## Least square finding coefficients of linear control variates
    def linear_coef(self, X_train, Y_train):
        """
        compute the coefficients in linear control variates without intercept
        
        :params[in]: X_train, a 2-d tensor, input training data of explanatory variables,
            each row indicates a training example, each column indicates a variable
        :params[in]: Y, a 1-d tensor, and its length equal to number of rows in X_train
        
        :params[out]: coef, real, the coefficients of linear regression
        """
        x_dim = X_train.size()[1]                             # input dimension 
        ## attach response y as the last variable
        score_resp = torch.cat((X_train, Y_train.unsqueeze(dim=1)), dim=1)
        ## covariance matrix between design matrix and response  
        cov_score_y = torch.tensor(np.cov(score_resp, rowvar=False))
        cov_score, score_y = cov_score_y[:x_dim,:x_dim],cov_score_y[:x_dim,x_dim]
        coef = torch.matmul(cov_score.inverse(), score_y)     # coefficients
        return coef.float()
    
    ## prepare explanatory variables for each training example Quadratic CV
    def prepare_feature(self, i, x):
        """
        prepare feature vectors for each training example prior to
            computing coefficients in quadratic control variates
            
        :params[in]: x, a 1-d tensor, training data that represents an observation
        :params[in]: i, an integer, row index of the score matrix
        
        :params[out]: x_features, a 1-d tensor, explanatory variables in feature space
        """
        ## score at x, i.e., gradient of log density
        score_x = self.score_matrix[i]
        return score_x
    
    ## classical way to find QUADRATIC control variates
    def classical_way(self, X, Y, integral, train_perc=.3):
        """
        classical approach to estimate parameters
        
        :params[in]: X, input design matrix, a 2-d torch array
        :params[in]: Y, the evaluation of integrand at X
        :params[in]: integral, real, the real value of integral
        :params[in]: train_perc, the percent of samples used in training period
        
        :params[out]: err0, monte carlo error 
        :params[out]: err1, error from control variates 
        :params[out]: err2, error from CV with full dataset
        :params[out]: comp_time, time to compute
        :params[out]: coef, coefficients of linear regression
        """
        start_time = time.time()                              # starting time 
        ## features from input data X, each row is a feature vector for an input
        X_feature = torch.stack([self.prepare_feature(i, x) for i,x in enumerate(X)], dim=0)
        nsamples,n_train = len(Y),int(len(Y)*train_perc)      # number of samples
        X_train,Y_train = X_feature[:n_train],Y[:n_train]     # train data set
        ## call function to compute coefficients
        coef = self.linear_coef(X_train, Y_train)
        ## evaluate control variates
        cv_eval = torch.matmul(X_feature, coef)
        cv_estimates = Y.numpy() - cv_eval.numpy()
        ## estimates of f hat
        cv_hat = cv_eval+Y_train.mean()
        ## compute estimates from the control variates 
        if train_perc<1.0:                                    # part of data for training    
            cv_mean = np.mean(cv_estimates[n_train:nsamples])     # estimate do not use training sample
            ## compute errors     
            err0 = Y.mean().item() - integral                     # monte carlo error
            err1 = cv_mean - integral                             # error using test data set
            #err2 = full_est - integral                           # error using whole data 
            comp_time = time.time() - start_time                  # computational time
        else:   ## all data used in training
            cv_mean = np.mean(cv_estimates)
            ## compute errors     
            err0 = Y.mean().item() - integral                     # monte carlo error
            err1 = cv_mean - integral                             # error using test data set
            #err2 = full_est - integral                           # error using whole data 
            comp_time = time.time() - start_time                  # computational time
        return (err0, err1, comp_time, cv_hat, coef)

