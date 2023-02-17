import numpy as np
import pylab as pl
import os,sys
import pandas as pd
import glob

import argparse
import configparser as ConfigParser
import ast

from scipy.optimize import minimize
import emcee
import corner
import dynesty
from dynesty import plotting as dyplot
        
        
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

class QUmodel():

    def __init__(self, pol_frac=False, catdata=None):
        
        self.nparms = 3
        self.pol_frac = pol_frac
        self.cat_data = catdata
            
        self.labels=["P0", "phi0", "chi0"]


    def log_like(self, theta):

        q_mod, u_mod = self.model(theta, self.data.l2)
        sigma2 = self.data.noise ** 2

        llike = -0.5 * np.sum((self.data.stokesQn - q_mod) ** 2 / sigma2) - 0.5 * np.sum((self.data.stokesUn - u_mod) ** 2 / sigma2)
        
        return llike


    def log_prior(self, theta):

        p0, phi0, chi0 = theta

        if 0. <= p0 < self.p0_max and -1000. < phi0 < 1000. and self.chi0_min <= chi0 < self.chi0_max:
            return 0.0
        return -np.inf


    def log_prob(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_like(theta)
        
        
    def ml_fit(self):
    
        if not hasattr(self, 'pinit'):
            self.initialisation()
        
        nll = lambda *args: -1.*self.log_like(*args)
        
        soln = minimize(nll, self.pinit, bounds=self.bnds)
        self.parms_ml = soln.x
                
        print("------")
        print("ML estimates:")
        for i in range(len(self.pinit)):
            print("{} = {:.3f}".format(self.labels[i].strip('$\\').replace('_',''), self.parms_ml[i]))
        print("------")
    
        return
        
        
    def mcmc_fit(self, plot_corner):

        self.plot_corner = plot_corner
        
        if not hasattr(self, 'pos'):
            self.initialisation()
            
        nwalkers, ndim = self.pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)
        sampler.run_mcmc(self.pos, 10000, progress=True)

        try:
            tau = sampler.get_autocorr_time()
            self.mcmc_samples = sampler.get_chain(discard=int(5*np.mean(tau)), thin=15, flat=True)
        except:
            print("Autocorrelation time too long - results questionable")
            self.mcmc_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

        if self.plot_corner:
            if self.parms_ml is not None:
                fig = corner.corner(self.mcmc_samples, labels=self.labels, truths=self.parms_ml, show_titles=True)
            else:
                fig = corner.corner(self.mcmc_samples, labels=self.labels, show_titles=True)
            srcid = 'ID'+str(self.srcid)
            pl.suptitle(srcid)
            pl.savefig(srcid+"_corner_mcmc.png", dpi=150)
            pl.close()

        print("------")
        print("MCMC estimates:")
    
        p_exp = np.zeros((ndim,3))
        for i in range(ndim):
            mcmc = np.percentile(self.mcmc_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            p_exp[i,0] = mcmc[1]
            p_exp[i,1] = q[0]
            p_exp[i,2] = q[1]
            
            print("{} = {:.3f} (-{:.3f}, +{:.3f})".format(self.labels[i].strip('$\\').replace('_',''), p_exp[i,0], p_exp[i,1], p_exp[i,2]))
        
        print("------")

        self.parms_mcmc = p_exp
        
        return
    
    
    def nested_fit(self, plot_corner):

        self.plot_corner = plot_corner
        
        if not hasattr(self, 'pinit'):
            self.initialisation()
            
        ndim = self.pinit.shape[0]
                
        sampler = dynesty.DynamicNestedSampler(self.log_prob, self.prior_transform, ndim)
        sampler.run_nested()
        dres = sampler.results
        
        if self.plot_corner:
            if self.parms_ml is not None:
                fig, axes = dyplot.cornerplot(dres, show_titles=True, truths=self.parms_ml, title_kwargs={'y': 1.04}, labels=self.labels, fig=pl.subplots(ndim, ndim, figsize=(15, 15)), title_quantiles=(0.16, 0.5, 0.84))
            else:
                fig, axes = dyplot.cornerplot(dres, show_titles=True, title_kwargs={'y': 1.04}, labels=self.labels,
                              fig=pl.subplots(ndim, ndim, figsize=(15, 15)))
                              
            srcid = 'ID'+str(self.srcid)
            pl.suptitle(srcid)
            pl.savefig(srcid+"_corner_nested.png", dpi=150)
            pl.close()

        self.nest_weights = dres.importance_weights()
        self.nest_samples = dres['samples']
        self.evid_nest = np.array([dres['logz'][-1],dres['logzerr'][-1]])
                
        print("------")
        print("Nested Sampler estimates:")
        
        p_exp = np.zeros((ndim,3))
        for i in range(ndim):
            nest = dynesty.utils.quantile(self.nest_samples[:,i],[0.16, 0.5, 0.84], weights=self.nest_weights)
            q = np.diff(nest)
            p_exp[i,0] = nest[1]
            p_exp[i,1] = q[0]
            p_exp[i,2] = q[1]
            
            print("{} = {:.3f} (-{:.3f}, +{:.3f})".format(self.labels[i].strip('$\\').replace('_',''), p_exp[i,0], p_exp[i,1], p_exp[i,2]))
        
        print(" ")
        print(r"log Z = {:.3f} +/- {:.3f}".format(self.evid_nest[0], self.evid_nest[1]))
        print("------")
        
        self.parms_nest = p_exp
        
        # 3 sigma bounds:
        nsig = 3
        a = []
        for i in range (0,ndim):
            a.append((p_exp[i,0] - nsig*p_exp[i,1], p_exp[i,0] + nsig*p_exp[i,2]))
        a = tuple(a)
        self.bnds = a
        
        sampler = dynesty.DynamicNestedSampler(self.log_prob, self.prior_transform, ndim)
        sampler.run_nested()
        dres = sampler.results
        self.evid_nest = np.array([dres['logz'][-1],dres['logzerr'][-1]])
        print(" ")
        print(r"log Z (3 sigma) = {:.3f} +/- {:.3f}".format(self.evid_nest[0], self.evid_nest[1]))
        print("------")
        
        # 10 sigma bounds:
        nsig = 10
        a = []
        for i in range (0,ndim):
            a.append((p_exp[i,0] - nsig*p_exp[i,1], p_exp[i,0] + nsig*p_exp[i,2]))
        a = tuple(a)
        self.bnds = a
        
        sampler = dynesty.DynamicNestedSampler(self.log_prob, self.prior_transform, ndim)
        sampler.run_nested()
        dres = sampler.results
        self.evid_nest = np.array([dres['logz'][-1],dres['logzerr'][-1]])
        print(" ")
        print(r"log Z (10 sigma) = {:.3f} +/- {:.3f}".format(self.evid_nest[0], self.evid_nest[1]))
        print("------")
        
        return
    
    
    def eval_model(self, parms):

        n = 2.*len(self.data.stokesQn)
        k = len(parms)
        
        if len(parms.shape)>1: parms = parms[:,0]
        
        llike = self.log_like(parms)
        self.chi2_red = -2.*llike/(n-k)
        self.chi2_red_err = np.sqrt(2./(n-k))
        
        self.bic = -2.*llike + k*np.log(n)
        self.aic = 2.*k - 2.*llike
        self.aicc= self.aic + 2*k*(k+1)/(n-k-1)
        
        print("------")
        print("Metrics:")
        print("Chi2_r : {:.2f}+/-{:.2f}".format(self.chi2_red, self.chi2_red_err))
        print("BIC    : {:.2f}".format(self.bic))
        print("AIC    : {:.2f}".format(self.aic))
        print("AICc   : {:.2f}".format(self.aicc))
        print("------")
        
        return


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

            
class QUSimple(QUmodel):

    def __init__(self, pol_frac=False, catdata=None):
    
        self.nparms = 3
        self.pol_frac = pol_frac
        self.cat_data = catdata
        self.data = None

        self.labels = [r"$P_0$", r"$\phi_0$", r"$\chi_0$"]
        

    def model(self, theta, x):
    
        p0, phi0, chi0 = theta

        q_mod = p0*np.cos(2*(phi0*x + chi0))
        u_mod = p0*np.sin(2*(phi0*x + chi0))

        return q_mod, u_mod


    def initialisation(self):

        idx = np.argwhere(self.cat_data[:,0]==self.srcid)[0][0]
        
        self.nu_ref = self.cat_data[idx,17]*1e6
        
        if not self.pol_frac:
            self.p0_init  = self.cat_data[idx,9]
            self.phi_init = self.cat_data[idx,15]
        else:
            self.p0_init  = self.cat_data[idx,11]
            self.phi_init = self.cat_data[idx,15]

        self.chi0_min = -0.5*np.pi
        self.chi0_max = 0.5*np.pi
        
        self.pinit = np.array([self.p0_init, self.phi_init, 0.]) # update this line for the model
                
        if self.pol_frac:
            self.p0_max = 100.
        else:
            # set adaptive upper bound on P0 prior:
            self.p0_max = 1000.
            if self.pinit[0]>1000.:
                self.p0_max = 1.2*pinit[0]
        
        print("------")
        print("Initialisation:")
        for i in range(0, self.nparms):
            print("{0} = {1:.3f}".format(self.labels[i].strip('$\\').replace('_',''), self.pinit[i]))
        print("------")
        
        self.pos = self.pinit + 1e-4 * np.random.randn(32, self.nparms)
        self.bnds = ((0, self.p0_max), (-1000., 1000.), (self.chi0_min, self.chi0_max))
        self.df_init = pd.DataFrame(self.pinit, self.labels)
        
        return

    def prior_transform(self, u):
    
        """Only used for the nested sampler"""

        x = np.array(u)  # copy u
        
        # uniform prior:
        for i in range(0,len(x)):
            w = self.bnds[i][1] - self.bnds[i][0]
            x[i] = u[i]*w
            x[i]+= self.bnds[i][0]

        return x


    def log_prior(self, theta):

        p0, phi0, chi0 = theta

        if 0. <= p0 < self.p0_max and -1000. < phi0 < 1000. and self.chi0_min <= chi0 < self.chi0_max:
            return 0.0
        return -np.inf
        
       
         
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

            
class QUSimpleExternal(QUmodel):

    def __init__(self, pol_frac, catdata):
    
        self.nparms = 4
        self.pol_frac = pol_frac
        self.cat_data = catdata

        self.labels = [r"$P_0$", r"$\phi_0$", r"$\chi_0$", r"$\sigma_{RM}$"]
        

    def model(self, theta, x):
    
        p0, phi0, chi0, sigma_RM = theta

        screen = np.exp(-2.*sigma_RM**2*x**2)

        q_mod = p0*np.cos(2*(phi0*x + chi0))*screen
        u_mod = p0*np.sin(2*(phi0*x + chi0))*screen

        return q_mod, u_mod
        
 
    def initialisation(self):

        idx = np.argwhere(self.cat_data[:,0]==self.srcid)[0][0]
        
        if not self.pol_frac:
            self.p0_init  = self.cat_data[idx,9]
            self.phi_init = self.cat_data[idx,15]
        else:
            self.p0_init  = self.cat_data[idx,11]
            self.phi_init = self.cat_data[idx,15]

        self.chi0_min = -0.5*np.pi
        self.chi0_max = 0.5*np.pi
        
        self.pinit = np.array([self.p0_init, self.phi_init, 0., 0.1])
                
        if self.pol_frac:
            self.p0_max = 100.
        else:
            # set adaptive upper bound on P0 prior:
            self.p0_max = 1000.
            if self.pinit[0]>1000.:
                self.p0_max = 1.2*pinit[0]
        
        print("------")
        print("Initialisation:")
        for i in range(0, self.nparms):
            print("{0} = {1:.3f}".format(self.labels[i].strip('$\\').replace('_',''), self.pinit[i]))
        print("------")

        self.pos = self.pinit + 1e-4 * np.random.randn(32, self.nparms)
        self.pos[:,-1] = np.abs(self.pos[:,-1]) # sigmaRM >= 0.
        self.bnds = ((0, self.p0_max), (-1000., 1000.), (self.chi0_min,self.chi0_max),(0, 100.))
        
        return

    def prior_transform(self, u):
    
        """Only used for the nested sampler"""

        x = np.array(u)  # copy u
        
        # uniform prior:
        for i in range(0,len(x)):
            w = self.bnds[i][1] - self.bnds[i][0]
            x[i] = u[i]*w
            x[i]+= self.bnds[i][0]

        return x


    def log_prior(self, theta):

        p0, phi0, chi0, sigmaRM = theta

        if 0. <= p0 < self.p0_max and -1000. < phi0 < 1000. and self.chi0_min <= chi0 < self.chi0_max and 0. <= sigmaRM < 100.:
            return 0.0
        return -np.inf
 
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
            
class QUSimpleDouble(QUmodel):

    def __init__(self, pol_frac, catdata):
    
        self.nparms = 6
        self.pol_frac = pol_frac
        self.cat_data = catdata

        self.labels = [r"$P_0$", r"$\phi_0$", r"$\chi_0$", r"$f_{\rm p}$", r"$\phi_1$", r"$\chi_1$"]
        

    def model(self, theta, x):
    
        p0, phi0, chi0, f, phi1, chi1 = theta

        q_mod = p0*np.cos(2*(phi0*x + chi0)) + f*p0*np.cos(2*(phi1*x + chi1))
        u_mod = p0*np.sin(2*(phi0*x + chi0)) + f*p0*np.sin(2*(phi1*x + chi1))

        return q_mod, u_mod
        
 
    def initialisation(self):

        idx = np.argwhere(self.cat_data[:,0]==self.srcid)[0][0]
        
        if not self.pol_frac:
            self.p0_init  = self.cat_data[idx,9]
            self.phi_init = self.cat_data[idx,15]
        else:
            self.p0_init  = self.cat_data[idx,11]
            self.phi_init = self.cat_data[idx,15]

        self.chi0_min = -0.5*np.pi
        self.chi0_max = 0.5*np.pi
        
        self.pinit = np.array([self.p0_init, self.phi_init, 0., 0.5, self.phi_init, 0.])
                
        if self.pol_frac:
            self.p0_max = 100.
        else:
            # set adaptive upper bound on P0 prior:
            self.p0_max = 1000.
            if self.pinit[0]>1000.:
                self.p0_max = 1.2*pinit[0]
        
        print("------")
        print("Initialisation:")
        for i in range(0, self.nparms):
            print("{0} = {1:.3f}".format(self.labels[i].strip('$\\').replace('_',''), self.pinit[i]))
        print("------")

        self.pos = self.pinit + 1e-4 * np.random.randn(32, self.nparms)
        self.bnds = ((0, self.p0_max), (-1000., 1000.), (self.chi0_min,self.chi0_max),(0, 1.), (-1000., 1000.), (self.chi0_min,self.chi0_max))
        
        return

    def prior_transform(self, u):
    
        """Only used for the nested sampler"""

        x = np.array(u)  # copy u
        
        # uniform prior:
        for i in range(0,len(x)):
            w = self.bnds[i][1] - self.bnds[i][0]
            x[i] = u[i]*w
            x[i]+= self.bnds[i][0]

        return x


    def log_prior(self, theta):

        p0, phi0, chi0, f, phi1, chi1 = theta

        if 0. <= p0 < self.p0_max and -1000. < phi0 < 1000. and self.chi0_min <= chi0 < self.chi0_max and 0. <= f < 1. and -1000. < phi1 < 1000. and self.chi0_min <= chi1 < self.chi0_max:
            return 0.0
        return -np.inf
 
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
