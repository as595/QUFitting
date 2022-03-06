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

from faraday_stuff import *

pl.rcParams['figure.figsize'] = [20, 5]
pl.rcParams['figure.dpi'] = 300

# ----------------------------------------------------------

def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')

    args, __ = parser.parse_known_args()

    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):

    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(filename)

    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():

        if section not in taskvals:
            taskvals[section] = dict()

        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

# -----------------------------------------------------------


class QUfit():

    def __init__(self, cfg_file):

        self.config = cfg_file
        self.ml_done = False
        self.mcmc_done = False

    def read_cfg(self):

        #vars = parse_args()
        config_dict, config = parse_config(self.config)

        self.fit_type  = config_dict['data']['type']
        self.data_path = config_dict['data']['path']
        if self.fit_type=='single':
            self.data_file = config_dict['data']['file']
            
        self.bkg_corr  = config_dict['data']['bkg_correction']
        self.pol_frac  = config_dict['data']['pol_frac']

        self.rm_path   = config_dict['rmspec']['path']

        self.fit_ml    = config_dict['fitting']['ml']
        self.fit_mcmc  = config_dict['fitting']['mcmc']

        self.plot_path = config_dict['plots']['path']
        if not os.path.exists(self.plot_path):
            print("Path to plotting outputs does not exist - please correct path")
            quit()

        self.plot_raw  = config_dict['plots']['rawdata']
        self.plot_fd   = config_dict['plots']['fdspec']
        self.plot_ml   = config_dict['plots']['mlfit']
        self.plot_corner = config_dict['plots']['corner']
        self.plot_mcmc = config_dict['plots']['mcmcfit']

        self.outpath   = config_dict['output']['path']
        self.exists    = config_dict['output']['exists']
        self.outfile   = config_dict['output']['outfile']
        self.output    = config_dict['output']['write_output']

        return


    def stokesI(self, theta, x):
    
        i0, alpha = theta
        
        i_mod = i0*(x/1.4e9)**-1.*alpha # frequency in hz
    
        return i_mod


    def stokesQU(self, theta, x):

        p0, phi0, chi0 = theta

        q_mod = p0*np.cos(2*(phi0*x + chi0))
        u_mod = p0*np.sin(2*(phi0*x + chi0))

        return q_mod, u_mod


    def plot_rawdata(self):

        srcid = self.data_file.split('_')[0]

        ax = pl.subplot(111)

        ax.errorbar(self.l2[::-1],self.stokesQn[::-1], yerr=self.noise[::-1], fmt='.', c='black', capthick=0, label="Stokes Q")
        ax.errorbar(self.l2[::-1],self.stokesUn[::-1], yerr=self.noise[::-1], fmt='.', c='grey', capthick=0, label="Stokes U")

        if self.plot_ml and self.ml_done:
            lstar = np.linspace(np.min(self.l2), np.max(self.l2), 512)
            q_fit, u_fit = self.stokesQU(self.parms_ml, lstar)

            ax.plot(lstar, q_fit, linestyle='-', color = 'c', lw=1.0)
            ax.plot(lstar, u_fit, linestyle='-', color = 'c', lw=1.0)
            pl.title(r"{} Maximum likelihood fit: $P_0$={:.3f}, $\phi_0$={:.3f}, $\chi_0$={:.3f}".format(srcid, self.parms_ml[0],self.parms_ml[1],self.parms_ml[2]))


        if self.plot_mcmc and self.mcmc_done:
            inds = np.random.randint(len(self.flat_samples), size=100)
            lstar = np.linspace(np.min(self.l2), np.max(self.l2), 512)
            for ind in inds:
                sample = self.flat_samples[ind]
                q_fit, u_fit = self.stokesQU(sample, lstar)

                ax.plot(lstar, q_fit, "C1", alpha=0.1)
                ax.plot(lstar, u_fit, "C1", alpha=0.1)

            mu_q, mu_u = self.stokesQU(self.parms_exp[:,0], lstar)
            ax.plot(lstar, mu_q, linestyle='-', color = 'c', lw=1.0)
            ax.plot(lstar, mu_u, linestyle='-', color = 'c', lw=1.0)
            pl.title(r"{} MCMC fit: $P_0={:.3f}_{{-{:.3f}}}^{{{:.3f}}}$, $\phi_0={:.3f}_{{-{:.3f}}}^{{{:.3f}}}$, $\chi_0={:.3f}_{{-{:.3f}}}^{{{:.3f}}}$".format(srcid, self.parms_exp[0,0],self.parms_exp[0,1],self.parms_exp[0,2],self.parms_exp[1,0],self.parms_exp[1,1],self.parms_exp[1,2],self.parms_exp[2,0],self.parms_exp[2,1],self.parms_exp[2,2]))

        ax.set_ylabel("Polarisation")
        ax.set_xlabel(r"$\lambda^2$")

        ax.legend()

        if self.plot_ml and self.ml_done:
            pl.savefig(self.plot_path+srcid+"_mlfit.png")
        elif self.plot_mcmc and self.mcmc_done:
            pl.savefig(self.plot_path+srcid+"_mcmcfit.png")
        else:
            pl.savefig(self.plot_path+srcid+"_raw.png")

        pl.close()

        return


    def plot_fdspec(self):

        pl.rcParams['figure.figsize'] = [20, 5]
        pl.rcParams['figure.dpi'] = 300

        srcid = self.data_file.split('_')[0]
        
        ax1 = pl.subplot(121)
        ax1.plot(self.phi,np.real(self.rmtf),ls='--',c='c',label="Real")
        ax1.plot(self.phi,np.imag(self.rmtf),ls=':',c='c',label="Imag")
        ax1.plot(self.phi,np.abs(self.rmtf),ls='-',c='grey',label="Abs")
        ax1.set_xlim(-2000,2000)
        ax1.set_xlabel(r"Faraday Depth [rad m$^{-2}$]", fontsize=12)
        ax1.set_title("RMTF")

        ax3 = pl.subplot(122)
        ax3.plot(self.phi,np.real(self.fspec),ls='--',c='c',label="Real")
        ax3.plot(self.phi,np.imag(self.fspec),ls=':',c='c',label="Imag")
        ax3.plot(self.phi,np.abs(self.fspec),ls='-',c='grey',label="Abs")
        ax3.set_xlim(-2000,2000)
        ax3.set_xlabel(r"Faraday Depth [rad m$^{-2}$]", fontsize=12)
        ax3.set_title("FD Spectrum")

        pl.suptitle(srcid)
        pl.savefig(self.plot_path+srcid+"_fdspec.png")
        pl.close()

        return


    def read_data(self):

        # freq       I        Q        U        V        N
        data = np.loadtxt(self.data_path+self.data_file)
        self.nu   = data[:,0]*1e6
        self.stokesI  = data[:,1]  # stokes I in file is Russ' model, not the raw data
        self.stokesQn = data[:,2]
        self.stokesUn = data[:,3]
        self.stokesVn = data[:,4]
        Qbkg = data[:,5]
        Ubkg = data[:,6]
        Vbkg = data[:,7]
        self.noise    = data[:,8]

        if self.bkg_corr:
            self.stokesQn -= Qbkg
            self.stokesUn -= Ubkg
            
        if self.pol_frac:
            self.stokesQn *= 100./self.stokesI
            self.stokesUn *= 100./self.stokesI
            self.noise    *= 100./self.stokesI
        
        const_c = 3e8

        # make data in lambda^2:
        self.l2 = (const_c/self.nu)**2

        if self.plot_raw:
            self.plot_rawdata()

        return

    def make_fdspec(self):

        fspec = []; rmtf = []

        self.w = np.ones(len(self.stokesQn))

        self.phi = np.linspace(-2000,2000,10000)

        for i in range(0,len(self.phi)):
            fspec.append(calc_f(self.phi[i],self.l2[::-1],self.stokesQn[::-1],self.stokesUn[::-1],self.w))
            rmtf.append(calc_r(self.phi[i],self.l2[::-1],self.w))

        self.fspec = np.array(fspec)
        self.rmtf = np.array(rmtf)

        if self.plot_fd:
            self.plot_fdspec()

        self.phi_init = self.phi[np.argmax(np.abs(self.fspec))]
        self.p0_init  = np.max(np.abs(self.fspec))
        
        return
        

    def get_initialisation(self):

        srcid = int(self.data_file.split('_')[0][2:])
        idx = np.argwhere(self.new_data[:,0]==srcid)[0][0]
        
        if not self.pol_frac:
            self.p0_init  = self.new_data[idx,9]
            self.phi_init = self.new_data[idx,15]
        else:
            self.p0_init  = self.new_data[idx,11]
            self.phi_init = self.new_data[idx,15]

            
        print("------")
        print("Initialisation:")
        print("P0 = {0:.3f}".format(self.p0_init))
        print("phi0 = {0:.3f}".format(self.phi_init))
        print("chi0 = {0:.3f}".format(0.0))
        print("------")

        return


    def log_like(self, theta):

        q_mod, u_mod = self.stokesQU(theta, self.l2)
        sigma2 = self.noise ** 2

        llike = -0.5 * np.sum((self.stokesQn - q_mod) ** 2 / sigma2) - 0.5 * np.sum((self.stokesUn - u_mod) ** 2 / sigma2)
        
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
    
        self.get_initialisation()

        np.random.seed(42)
        nll = lambda *args: -1.*self.log_like(*args)
        initial = np.array([self.p0_init, self.phi_init, 0.]) + 0.1 * np.random.randn(3)
        
        bnds = ((0, None), (None, None), (-0.5*np.pi, 0.5*np.pi))
        soln = minimize(nll, initial, bounds=bnds)
        self.parms_ml = soln.x
        self.p0_ml, self.phi_ml, self.chi_ml = soln.x

        print("------")
        print("Maximum likelihood estimates:")
        print("P0 = {0:.3f}".format(self.p0_ml))
        print("phi0 = {0:.3f}".format(self.phi_ml))
        print("chi0 = {0:.3f}".format(self.chi_ml))
        print("------")

        self.ml_done = True
        if self.plot_ml:
            self.plot_rawdata()
        self.ml_done = False

        return


    def mcmc_fit(self):

        self.chi0_min = -np.pi
        self.chi0_max = np.pi
        
        if not self.fit_ml:
            self.get_initialisation()
            pinit = [self.p0_init, self.phi_init, 0.]
        else:
            pinit = self.parms_ml
                
        if self.pol_frac:
            self.p0_max = 100.
        else:
            # set adaptive upper bound on P0 prior:
            self.p0_max = 1000.
            if pinit[0]>1000.:
                self.p0_max = 1.2*pinit[0]
        
        pos = pinit + 1e-4 * np.random.randn(32, 3)
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)
        sampler.run_mcmc(pos, 10000, progress=True)

        try:
            tau = sampler.get_autocorr_time()
            self.flat_samples = sampler.get_chain(discard=int(5*np.mean(tau)), thin=15, flat=True)
        except:
            print("Autocorrelation time too long - results questionable")
            self.flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

        if self.plot_corner:
            labels = [r"$P_0$", r"$\phi_0$", r"$\chi_0$"]
            fig = corner.corner(self.flat_samples, labels=labels, show_titles=True)
            srcid = self.data_file.split('_')[0]
            pl.suptitle(srcid)
            pl.savefig(self.plot_path+srcid+"_corner.png", dpi=150)
            pl.close()

        p_exp = np.zeros((ndim,3))
        for i in range(ndim):
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            p_exp[i,0] = mcmc[1]
            p_exp[i,1] = q[0]
            p_exp[i,2] = q[1]

        self.parms_exp = p_exp

        print("------")
        print("MCMC estimates:")
        print("P0 = {:.3f} (-{:.3f}, +{:.3f})".format(self.parms_exp[0,0],self.parms_exp[0,1],self.parms_exp[0,2]))
        print("phi0 = {:.3f} (-{:.3f}, +{:.3f})".format(self.parms_exp[1,0],self.parms_exp[1,1],self.parms_exp[1,2]))
        print("chi0 = {:.3f} (-{:.3f}, +{:.3f})".format(self.parms_exp[2,0],self.parms_exp[2,1],self.parms_exp[2,2]))
        print("------")

        self.mcmc_done = True
        if self.plot_mcmc:
            self.plot_rawdata()
                    
        self.mcmc_done = False

        return


    def create_output(self):

        f = open(self.outpath+self.exists)
        cat_cols = f.readline().rstrip("\n")
        cat_unts = f.readline().rstrip("\n")
        cat_data = np.loadtxt(self.outpath+self.exists)
        
        if self.fit_ml and self.fit_mcmc:
            new_cols = cat_cols+"{:^10} {:^10} {:^10} {:^10}".format("RM_ml", "RM_map", "-ve_err_map", "+ve_err_map")
            new_unts = cat_unts+"{:^10} {:^10} {:^10} {:^10}".format("(rad/m^2)", "(rad/m^2)", "(rad/m^2)", "(rad/m^2)")
            self.new_data = np.c_[cat_data, -99*np.ones(cat_data.shape[0]), -99*np.ones(cat_data.shape[0]), -99*np.ones(cat_data.shape[0]), -99*np.ones(cat_data.shape[0])]

        elif self.fit_mcmc and not self.fit_ml:
            new_cols = cat_cols+"{:^10} {:^10} {:^10}".format("RM_map", "-ve_err_map", "+ve_err_map")
            new_unts = cat_unts+"{:^10} {:^10} {:^10}".format("(rad/m^2)", "(rad/m^2)", "(rad/m^2)")
            self.new_data = np.c_[cat_data, -99*np.ones(cat_data.shape[0]), -99*np.ones(cat_data.shape[0]), -99*np.ones(cat_data.shape[0])]


        new_cols = new_cols.lstrip('#')
        new_unts = new_unts.lstrip('#')

        self.header = new_cols+"\n"+new_unts

        return


    def update_output(self):

        srcid = int(self.data_file.split('_')[0][2:])
        if self.fit_ml:
            rm_a  = self.parms_ml[1]
        else:
            rm_a = 0.0
            
        rm_b  = self.parms_exp[1,0]
        err1_b= self.parms_exp[1,1]
        err2_b= self.parms_exp[1,2]

        try:
            idx = np.argwhere(self.new_data[:,0]==srcid)[0][0]
            if self.fit_ml and self.fit_mcmc:
                self.new_data[idx,18] = rm_a
                self.new_data[idx,19] = rm_b
                self.new_data[idx,20] = err1_b
                self.new_data[idx,21] = err2_b
            elif self.fit_mcmc and not self.fit_ml:
                self.new_data[idx,18] = rm_b
                self.new_data[idx,19] = err1_b
                self.new_data[idx,20] = err2_b

        except:
            print("Source not in catalogue - no update")

        return


    def write_output(self):

        if self.fit_ml and self.fit_mcmc:
            fmt = '%6.0f %9.5f %5.2f %9.5f %5.2f %8.1f %9.2f %8.2f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.2f %8.2f %8.2f %8.2f'
            np.savetxt(self.outpath+self.outfile, self.new_data, fmt=fmt, header=self.header)
        elif self.fit_mcmc and not self.fit_ml:
            fmt = '%6.0f %9.5f %5.2f %9.5f %5.2f %8.1f %9.2f %8.2f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.2f %8.2f %8.2f'
            np.savetxt(self.outpath+self.outfile, self.new_data, fmt=fmt, header=self.header)


        return


    def run_file(self):

        srcid = self.data_file.split('_')[0]
        print("------")
        print("Fitting {}".format(srcid))
        print("------")

        self.read_data()
        if self.fit_ml and not self.fit_mcmc:
            #self.make_fdspec()
            self.ml_fit()
        if self.fit_mcmc and not self.fit_ml:
            print("------")
            print("MCMC initialisation from main peak in FD spectrum")
            print("------")
            #self.make_fdspec()
            self.mcmc_fit()
        if self.fit_mcmc and self.fit_ml:
            print("------")
            print("MCMC initialisation from ML optimisation")
            print("------")
            #self.make_fdspec()
            self.ml_fit()
            self.mcmc_fit()

        return


    def run_qu(self):

        self.read_cfg()
        self.create_output()

        if self.fit_type=='all':

            self.file_list = glob.glob(self.data_path+"*polspec.txt")

            for file in self.file_list:
                self.data_file = file.split('/')[-1]
                srcid = int(self.data_file.split('_')[0][2:])
                try:
                    idx = np.argwhere(self.new_data[:,0]==srcid)[0][0]
                    self.run_file()
                    self.update_output()
                    
                except:
                    print("{}: source not in catalogue - no fit".format(srcid))

                
        else:
            self.run_file()
            self.update_output()

        if self.output:
            self.write_output()

        return

    def check_cat(self):
    
        self.read_cfg()
        
        if self.fit_type=='all':

            self.file_list = glob.glob(self.data_path+"*polspec.txt")
            self.spec_list = glob.glob(self.rm_path+"*rmspectrum.png")

            for file in self.file_list:
                self.data_file = file.split('/')[-1]
                srcid = int(self.data_file.split('_')[0][2:])
                try:
                    idx = np.argwhere(self.new_data[:,0]==srcid)[0][0]
                    print("{}: source in pol catalogue".format(srcid))
                    
                    rmplots = [s for s in self.spec_list if idx in s]
                    if len(rmplots)==0:
                        print("{}: source has no rmsynth plot".format(srcid))
            
                    
                except:
                    print("{}: source not in pol catalogue".format(srcid))
                    rmplots = [s for s in self.spec_list if idx in s]
                    if len(rmplots)==0:
                        print("{}: source has no rmsynth plot".format(srcid))
            
    
        return
        
                
    def log_like_i(self, theta):

        i_mod = self.stokesI(theta, self.nu)
        sigma2 = self.noise ** 2

        llike = -0.5 * np.sum((self.stokesIn - i_mod) ** 2 / sigma2) - 0.5 * np.sum((self.stokesIn - i_mod) ** 2 / sigma2)
        
        return llike


    def fit_spectrum(self):
    
        self.i0_init = np.mean(self.stokesIn)
        self.alpha_init = 0.7
        
        np.random.seed(42)
        nll = lambda *args: -1.*self.log_like_i(*args)
        initial = np.array([self.i0_init, self.alpha_init]) + 0.1 * np.random.randn(2)
        
        bnds = ((0, None), (None, None))
        soln = minimize(nll, initial, bounds=bnds)
        self.parms_ml = soln.x
        self.i0_ml, self.alpha_ml = soln.x

        return


if __name__ == "__main__":

    vars = parse_args()
    cfg_file = vars['config']
    
    qu = QUfit(cfg_file)
    qu.run_qu()


    
