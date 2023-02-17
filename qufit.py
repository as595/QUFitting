import numpy as np
import pylab as pl
import os,sys
import glob

from scipy.optimize import minimize

from faraday_stuff import *
from utils import *
from dataio import *
from qu_models import *

pl.rcParams['figure.figsize'] = [20, 5]
pl.rcParams['figure.dpi'] = 300


class QUfit():

    def __init__(self, cfg):
    
        self.cfg = cfg             # get configuration
        self.data = QUdata(cfg)    # init data class
        self.data.read_cat()       # read catalogue file 
        self.data.read_data()      # read the QU data
        

    def select_model(self):
    
        if self.cfg.modeltype=='SFT':           # SFT: Simple Faraday Thin
            self.model = QUSimple(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)
        elif self.cfg.modeltype=='SFTexternal': # SFT with External depol screen
            self.model = QUSimpleExternal(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)
        elif self.cfg.modeltype=='SFTdouble':   # SFT x 2
            self.model = QUSimpleDouble(pol_frac=self.cfg.pol_frac, catdata=self.data.cat_data)


        return

    def run_file(self):

        srcid = self.cfg.data_file.split('_')[0]
        print("------")
        print("Fitting {}".format(srcid))
        print("Model {}".format(self.cfg.modeltype))
        print("------")

        self.select_model()
        self.model.srcid = int(srcid[2:])
        self.model.data = self.data
            
        if self.cfg.fit_ml :
            self.model.ml_fit()
            self.parms_ml = self.model.parms_ml
            
        if self.cfg.fit_mcmc :
            self.model.mcmc_fit(plot_corner=True)
            self.parms_mcmc = self.model.parms_mcmc
            
        if self.cfg.fit_nested :
            self.model.nested_fit(plot_corner=True)
            self.parms_nest = self.model.parms_nest
            
        return


    def run_qu(self):

        if self.cfg.fit_type=='all':

            if self.data.catfile:
                self.file_list = glob.glob(self.cfg.data_path+"*polspec_external.txt")
            else:
                self.file_list = glob.glob(self.cfg.data_path+"*polspec.txt")

            for file in self.file_list:
                self.data_file = file.split('/')[-1]
                srcid = int(self.data_file.split('_')[0][2:])
                try:
                    idx = np.argwhere(self.data.cat_data[:,0]==srcid)[0][0]
                    self.run_file()
#                    self.update_output()
                    
                except:
                    print("{}: source not in catalogue - no fit".format(srcid))

                
        else:
            self.run_file()
 
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


    
