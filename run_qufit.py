import numpy as np
import pylab as pl
import os,sys

from qufit import *

qu = QUfit('xmmlss12.cfg')
qu.run_qu()

qu = QUfit('xmmlss13.cfg')
qu.run_qu()

qu = QUfit('xmmlss14.cfg')
qu.run_qu()

qu = QUfit('cosmos.cfg')
qu.run_qu()
