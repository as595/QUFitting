import numpy as np
import pylab as pl
import os,sys
import glob
import csv
import itertools
from tqdm import tqdm

from scipy.optimize import minimize

from faraday_stuff import *
from utils import *
from dataio import *
from qu_models import *

pl.rcParams['figure.figsize'] = [20, 5]
pl.rcParams['figure.dpi'] = 300


class QUfit():

    def __init__(self, cfg_file):
    
        self.cfg = QUcfg()              # init config class
        self.cfg.read_cfg(cfg_file)     # read the config file
        self.data = QUdata(self.cfg)    # init data class
        self.data.read_data()           # read the QU data
        self.data.read_cat()            # read catalogue file
        self._results_col = []          # empty results list headers
        self._results = []              # empty results array

        self.verbose=True

    def select_model(self, pinit=None):
    
        if self.cfg.modeltype=='SFT':           # SFT: Simple Faraday Thin
            self.model = QUSimple(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)
        elif self.cfg.modeltype=='SFTexternal': # SFT with External depol screen
            self.model = QUSimpleExternal(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)
        elif self.cfg.modeltype=='SFTdouble':   # SFT x 2
            self.model = QUSimpleDouble(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)

        self.pinit=pinit
        
        return


    def run_file(self):

        srcid = self.cfg.data_file.split('_')[0]
        if self.verbose:
            print("------")
            print("Fitting {}".format(srcid))
            print("Model {}".format(self.cfg.modeltype))
            print("------")

        self.select_model()
        self.model.srcid = int(srcid[2:])
        self.model.data = self.data
        self.model.verbose = self.verbose
            
        self._results_col.append(['SRCID'])
        self._results.append([self.model.srcid])
            
        if self.cfg.fit_ml :
            self.model.ml_fit()
            self.parms_ml = self.model.parms_ml
            # create output row:
            labels=[]
            for label in self.model.labels: labels.append(label+"_ML")
            self._results_col.append(labels)
            _ml_results = self.parms_ml
            self._results.append(list(_ml_results))

        if self.cfg.fit_mcmc :
            self.model.mcmc_fit(plot_corner=True)
            self.parms_mcmc = self.model.parms_mcmc
            # create output row:
            labels=[]
            for label in self.model.labels: labels.append([label+"_MCMC", label+"_lo", label+"_hi"])
            labels = list(itertools.chain.from_iterable(labels))
            self._results_col.append(labels)
            _mcmc_results = self.parms_mcmc.flatten()
            self._results.append(list(_mcmc_results))

        if self.cfg.fit_nested :
            self.model.nested_fit(plot_corner=True)
            self.parms_nest = self.model.parms_nest
            self._results_col.append(self.model.labels)
            # create output row:
            labels=[]
            for label in self.model.labels: labels.append([label+"_NS", label+"_lo", label+"_hi"])
            labels.append(["lnZ", "e_lnZ"])
            labels = list(itertools.chain.from_iterable(labels))
            self._results_col.append(labels)
            _ns_results = np.concatenate((self.parms_nest.flatten(), self.model.evid_nest))
            self._results.append(list(_ns_results))
            
        return


    def run_qu(self):

        if self.cfg.fit_type=='all':

            self.file_list = glob.glob(self.cfg.data_path+"*polspec.txt")
            if os.path.exists(self.cfg.outfile): os.system("rm {} \n".format(self.cfg.outfile))
            self.verbose = False

            for file in tqdm(self.file_list):
                self._results_col = []          # empty results list headers
                self._results = []              # empty results array

                self.cfg.data_file = file.split('/')[-1]
                srcid = int(self.cfg.data_file.split('_')[0][2:])
                
                try:
                    idx = np.argwhere(self.data.cat_data[:,0]==srcid)[0][0]
                    self.data.read_data()           # update QU data
                    self.run_file()
                    if self.cfg.output:
                        if os.path.exists(self.cfg.outfile):
                            self.write_output(append=True)
                        else:
                            self.write_output(append=False)
                    
                except:
                    if self.verbose:
                        print("{}: source not in catalogue - no fit".format(srcid))

                
        else:
            self.run_file()
            if self.cfg.output: self.write_output(append=False)
            
        return
        
    def write_output(self, append=False):

        if append: 
            with open(self.cfg.outfile, 'a', newline="") as f_out:
                        writer = csv.writer(f_out, delimiter=',')
                        _results = list(itertools.chain.from_iterable(self._results))
                        writer.writerow(_results)
        else: 
            with open(self.cfg.outfile, 'w+', newline="") as f_out:
                writer = csv.writer(f_out, delimiter=',')
                _results_col = list(itertools.chain.from_iterable(self._results_col))
                _results = list(itertools.chain.from_iterable(self._results))
                writer.writerow(_results_col)
                writer.writerow(_results)
        
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
        


if __name__ == "__main__":

    vars = parse_args()
    cfg_file = vars['config']
    
    qu = QUfit(cfg_file)
    qu.run_qu()


    
