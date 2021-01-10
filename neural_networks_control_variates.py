# load modules
import torch
import time, pickle
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy import stats
#%matplotlib inline
from scipy import special
from torch.autograd import Variable

## class: Neural nets Control Variates-- use neural nets in contrtol variates
## using Stein's first order operator with quadratic polynomial control variates
class SteinFirstOrderNeuralNets(nn.Module): # inherit torch.nn.Module
    
    def __init__(self, D_in, h_dims, score_matrix, init_val, drop_prob=.5):
        """
        initialization
        
        :params[in]: D_in, int, dimension of input to the integrand
        :params[in]: h_dims, list, dimension of hidden layers        
        :params[in]: score_matrix, 2-d tensor, each row represents a score vector for a training example
        :params[in]: init_val, a scalar tensor, initial value for constant
        """
        super().__init__()     # initialize the superclass
        self.dims = h_dims     # hidden dimensions
        self.dims.append(1)    # output dimension
        self.dims.insert(0, D_in) # input dimension
        # fully connected layers
        self.layers = nn.ModuleList([nn.Linear(self.dims[i-1], self.dims[i]) for \
                                  i in range(1, len(self.dims))])
        # intercept constant initial value = 0
        self.c = torch.nn.Parameter(init_val, requires_grad=True) 
        self.score_matrix = score_matrix # score matrix of prob. measure
        self.drop = nn.Dropout(p=drop_prob)
        
    def net_utils(self, x):
        """
        conpute the gradient and second order derivatives of neural networks
        :params[in]: x, an input example, a 1-d tensor
        
        :params[out]: grads, a one-d tensor
        :params[out]: y, evaluate the output of neural net
        """
        y=F.relu(self.drop(self.layers[0](x)))
        for it in range(1, len(self.layers)-1):
            #if it%2 ==1:
            #    y=y+F.relu(self.drop(self.layers[it](y)))   # iteratively pass all layers
            #else:
            y=F.relu(self.drop(self.layers[it](y)))
        y = self.layers[-1](y)                        # last layer -- linear       
        ## find dy/dx
        grads = grad(y, x, create_graph=True)[0]
        #sec = torch.trace(self.w)                # 2nd order laplace operator
        return y,grads
        
    ## forward
    def forward(self, x, ind):
        """
        forward function
        :params[in]: x, 1-d input tensor
        :params[in]: ind, index of score vector
        
        :params[out]: y_pred, the predicted value
        """
        # predicted y
        score = self.score_matrix[ind]
        eva_net, grads = self.net_utils(x)
        y_pred = grads.sum() + eva_net*score.sum()
        return y_pred
    
    ## mini-batch stochastic gradient
    def minibatch(self, x_batch, indices):
        """
        evaluate the forward functions on a batch of examples
        
        :params[in]: x_batch, a 2-d tensor of which each row is a training example
        :params[in]: indices, list of indices to retrieve score vectors
        
        :params[out]: y_pred, a 1-d tensor of predicted results
        """
        # first order operator
        res = [self.forward(x,ind) for x,ind in zip(x_batch, indices)]
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
    def trainer(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, valid_perc=.1,\
              step_size=10, weight_decay = 0.0, batch_size = 32, result_epochs=[10,20,30,40],norm_init_std=1.e-3):
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
        ##ES = EarlyStopping(5)                   # early stopping
        for each_par in self.parameters():   ## initialize this neural nets
            if each_par is self.c:
                continue
            torch.nn.init.normal_(each_par, mean=0, std=norm_init_std)
        ## use adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        ## SGD optimizer
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
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
            self.train(True)                     # training mode
            for batch in batches:                # mini-batch training
                if (len(batch)<.25*batch_size):  # a batch has too few training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, list(batch)).squeeze()
                #l1norm = self.w.norm(p=1)
                # produce output 
                loss = (y - y_pred - self.c).pow(2).mean() + weight_decay*y_pred.pow(2).mean()
                #pdb.set_trace()
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            self.eval()                     # training mode
            abs_train_err = [(self.forward(X[ind0], ind0).detach() +self.c.detach() - Y[ind0]).abs().item() for ind0 in \
                             range(0,  n_train)]
            ## compute the absolute errors on the validation set
            if train_perc<1.0:
                abs_test_err = [(self.forward(X[ind0], ind0).detach()+self.c.detach() - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]  
                ## indicator of how close the expectation of control variates to zero
                indicator = [(self.forward(X[ind0], ind0).detach()).item() for ind0 in \
                            range(n_train, (n_train+n_valid))]
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                    'validation loss: ',np.mean(abs_test_err), 'indicator:',np.abs(np.mean(indicator)))
            else:
                ## indicator of how close the expectation of control variates to zero
                indicator = [(self.forward(X[ind0], ind0).detach()).item() for ind0 in \
                            range(0, n_train)]
                print(i,'-th Epoch, training loss: ', np.mean(abs_train_err), \
                      'indicator:',np.abs(np.mean(indicator)))                
            #ES(np.mean(abs_test_err))             # early stopping function
            #if ES.early_stop:                     # early stopping rule satisfied
            #    break
            ## check best performance 
            #if err_optimal < -1*np.mean(abs_test_err):      
            #    err_optimal, epoch_optimal = -1*np.mean(abs_test_err), i         
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimates of parameters after each epoch
            #dict_iters[i] = [list(self.parameters())[1].clone().detach(),\
            #                 list(self.parameters())[2].clone().detach()]
            ## compute the estimates during training period                
            if (i+1) in result_epochs:
                if train_perc<1.0:
                    err0, err1 = self.comp_errors(X, Y, integral, n_train)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:
                    err0 = Y.mean().item() - integral
                    err1 = self.c.item() - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        
    
    ## train and compute control variates by minimizing variance
    def trainer_var(self, X, Y, integral, lr, gamma, clip_value=10.0, train_perc=.3, valid_perc=.1,\
              step_size=10, weight_decay = 0.0, batch_size = 32, result_epochs=[10,20,30,40]):
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
        self.c.requires_grad=False             # set gradient of constant self.c to zero
        ##ES = EarlyStopping(5)                   # early stopping
        ## use adam optimizer
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        ## SGD optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
        ## learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)     
        ## loop over batches 
        nsamples = len(Y)                       # number of samples
        ## training examples -> 0: n_train//validation -> n_train:(n_train+n_valid)
        n_train,n_valid = int(nsamples*train_perc),int(nsamples*valid_perc)    
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
            self.train(True)                     # training mode
            for batch in batches:                # mini-batch training
                if (len(batch)<2):    # a batch has less than 2 training examples
                    continue
                x, y = X[batch], Y[batch]        # select a batch of training examples
                ## initialize gradients to zero
                optimizer.zero_grad()
                ## evaluate over a batch 
                y_pred = self.minibatch(x, list(batch)).squeeze()
                #l1norm = self.w.norm(p=1)
                # produce output 
                loss = (y_pred+self.c - y).var() + weight_decay*y_pred.pow(2).mean()
                #pdb.set_trace()
                ## optimize parameters    
                loss.backward()    
                ## gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()                  # update once 
            # after each epoch of training, compute the absolute error on the training set
            self.eval()                     # training mode
            train_err = [(Y[ind0]-self.forward(X[ind0], ind0).detach()-self.c.detach()).item() for ind0 in \
                             range(0,  n_train)]
            ## compute the absolute errors on the validation set
            if train_perc<1.0:
                abs_test_err = [(self.forward(X[ind0], ind0).detach()+self.c.detach() - Y[ind0]).abs().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]  
                ## indicator of how close the expectation of control variates to zero
                indicator = [self.forward(X[ind0], ind0).detach().item() for ind0 in \
                            range(n_train, (n_train+n_valid))]
                print(i,'-th Epoch, training loss: ', np.abs(train_err).mean(), \
                    'validation loss: ',np.mean(abs_test_err), 'indicator:',np.abs(np.mean(indicator)))
            else:
                ## indicator of how close the expectation of control variates to zero
                indicator = [self.forward(X[ind0], ind0).detach().item() for ind0 in \
                            range(0, n_train)]
                print(i,'-th Epoch, training loss: ', np.abs(train_err).mean(), \
                      'indicator:',np.abs(np.mean(indicator)))                
            #ES(np.mean(abs_test_err))             # early stopping function
            #if ES.early_stop:                     # early stopping rule satisfied
            #    break
            ## check best performance 
            #if err_optimal < -1*np.mean(abs_test_err):      
            #    err_optimal, epoch_optimal = -1*np.mean(abs_test_err), i         
            scheduler.step()                     # schedule learning rate
            np.random.shuffle(train_indexes)     # shuffle indices of examples for next epoch
            ## record estimates of parameters after each epoch
            #dict_iters[i] = [list(self.parameters())[1].clone().detach(),\
            #                 list(self.parameters())[2].clone().detach()]
            ## compute the estimates during training period                
            if (i+1) in result_epochs:
                if train_perc<1.0:
                    err0, err1 = self.comp_errors(X, Y, integral, n_train)
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)
                else:
                    err0 = Y.mean().item() - integral
                    err1 = np.mean(train_err) - integral
                    mc_errs.append(err0)
                    cv_errs.append(err1)
                    comp_time = time.time()-start_time # record the computational time
                    comp_times.append(comp_time)                    
        ## after training, record mean absolute errors on the training and validation set
        #train_err, test_err=np.mean(abs_train_err), np.mean(abs_test_err) 
        ## after model training, compute the errors from control variates and monte carlo
        return (mc_errs, cv_errs, comp_times) # train_err, test_err, err_optimal)        
    
    ## compute the estimates of integrals after training period 
    def comp_errors(self, X, Y, integral, n_train):
        """
        After model training is finished,
        Compute the estimates of the integral and errors from Monte Carlo and control variates

        :params[in]: X, a 2-d input data
        :params[in]: Y, a 1-d tensor, output
        :params[in]: integral, real, the real value of integral
        :params[in]: n_train, int, the training sample size
        #:params[in]: nsamples, int, the whole sample size

        :params[out]: err0, real, the Monte Carlo error
        :params[out]: err1, real, the control variates error only using test set, 
                       no training example        
        :params[out]: err2, real, the control variates error using total sample       
        """  
        err0 = Y.mean().item() - integral                     # Monte Carlo error
        nsamples = X.size()[0]                                # total sample size
        ## find the predicted values from the control variates    
        cv_eval = [self.forward(X[i1], i1).item() for i1 in range(nsamples)]
        cv_train_mean = np.mean(cv_eval[:n_train])            # mean of control variates on train set  
        cv_estimates = Y.numpy() - np.array(cv_eval)
        #cv_estimates1 = Y.numpy() - cv_eval + cv_train_mean   # another way
        ## compute estimates from the control variates    
        cv_mean = np.mean(cv_estimates[n_train:nsamples])     # estimate do not use training sample
        full_est = np.mean(cv_estimates)                      # estimates using whole sample
        #cv_mean1 = np.mean(cv_estimates1[n_train:nsamples])   # estimate of using training mean
        ## compute errors     
        err1 = cv_mean - integral                             # error using test data set
        err2 = full_est - integral                            # error using the whole data set
        #err3 = cv_mean1 - integral                            # error using test data with training mean
        return (err0, err1)
    
if __name__ == "__main__":
    ## experiments with polynomial covariates
    start_time = time.time()
    vary_dims = [5]         # varying dimensions
    num_ex,dim = 10, vary_dims[0]     # number of experiments
    half_res,full_res=[],[]
    for ex in range(num_ex):          # loop over dimensions
        tmp_half_res,tmp_full_res={},{}
        tmp_half_res['id']=tmp_full_res['id'] = ex # initialize the half/full training
        t1 = [[1., -1.] for i in range(dim)]
        Alpha, sigma = torch.tensor(t1), 1. # setup
        prob1 = Prepare_Poly(Alpha, sigma)  # problem 1
        D_in = Alpha.size()[0]
        # simulate data
        size = 1000   # sample size
        N_epochs = [10, 30, 50] # number of epochs
        train_pct, valid_pct = 0.9, .1
        lr, gamma = 1.e-2, .5
        tmp_half_res['sample_size']=tmp_full_res['sample_size']=size
        ## monte carlo method
        mc_time0=time.time()
        X = prob1.sample(size) # draw sample
        X.requires_grad=True
        Y = torch.tensor([prob1.integrand(X[i]) for i in range(size)]) # f(x[i])
        integral = prob1.integral() # true value of integral
        mc_est=Y.mean().item()
        mc_time=time.time()-mc_time0
        tmp_half_res['mc_time']=tmp_full_res['mc_time']=mc_time
        ## control variates with neural nets 
        cv1 = SteinFirstOrderNeuralNets(D_in)  
        ## split the data into two parts
        cv_res1=cv1.train(X, Y, integral, lr, gamma, clip_value=5000.0, train_perc=.9, valid_perc=0.1,\
                  weight_decay = 10**(-4), batch_size = 16, result_epochs=[10,20,30,40])
        tmp_half_res['mc_err']=mc_est - integral
        tmp_half_res['cv_err']=cv_res1[1]
        tmp_half_res['cv_time']=cv_res1[2]
        half_res.append(tmp_half_res)
        ## use the full data to estimate the intercept
        cv_res2=cv1.train(X, Y, integral, lr, gamma, clip_value=5000.0, train_perc=1., valid_perc=0.0,\
                  weight_decay = 10**(-4), batch_size = 16, result_epochs=[10,20,30,40])
        tmp_full_res['mc_err']=mc_est - integral
        tmp_full_res['cv_err']=cv_res2[1]
        tmp_full_res['cv_time']=cv_res2[2]
        full_res.append(tmp_full_res)
        ## final result
        res={'full_res':full_res, 'half_res':half_res}
    