import numpy as np
import pylab as pl
import os,sys
import pandas as pd
import glob

import argparse
import configparser as ConfigParser
import ast


class QUcfg():

    def __init__(self):
        return

    # ----------------------------------------------------------

    def parse_args(self):
        """
            Parse the command line arguments
            """
        parser = argparse.ArgumentParser()
        parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
        parser.add_argument('-S','--srcid', default=None, required=False, help='Source ID')
        
        args, __ = parser.parse_known_args()

        return vars(args)

    # -----------------------------------------------------------

    def parse_config(self, filename):

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

    def read_cfg(self, cfg_file):

        #vars = parse_args()
        config_dict, config = self.parse_config(cfg_file)

        self.fit_type  = config_dict['data']['type']
        self.data_path = config_dict['data']['path']
        self.data_file = config_dict['data']['file']
        self.catpath   = config_dict['data']['catpath']
        self.catfile   = config_dict['data']['catfile']

        self.bkg_corr  = config_dict['data']['bkg_correction']
        self.pol_frac  = config_dict['data']['pol_frac']

        if "rmspec" in config_dict:
            self.rm_path   = config_dict['rmspec']['path']

        if "fitting" in config_dict:
            self.modeltype = config_dict['fitting']['modeltype']
            self.fit_ml    = config_dict['fitting']['ml']
            self.fit_mcmc  = config_dict['fitting']['mcmc']
            self.fit_nested= config_dict['fitting']['nested']

        if "plots" in config_dict:
            self.plot_path = config_dict['plots']['path']
            if not os.path.exists(self.plot_path):
                print("Path to plotting outputs does not exist - please correct/create path")
                quit()

        if "output" in config_dict:
            self.outfile   = config_dict['output']['outfile']
            self.output   = config_dict['output']['write_output']
            

        return

