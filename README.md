# QUFitting for MeerKAT MIGHTEE-POL

## TL;DR

To run as an executable from the command line:

```
python qufit.py --config xmmlss13.cfg
```

To call as a class within a Python script:

```python
from qufit import *

cfg_file = 'xmmlss13.cfg'
qu = QUfit(cfg_file)
qu.run_qu()
```

## Method

QU fitting can be done in two ways: (i) a maximum-likelihood (ML) value obtained through direct optimisation using a single Faraday-thin component model, and (ii) a maximum-a-posteriori (MAP) expectation value obtained using MCMC optimisation of a single Faraday-thin component model. In both cases fits can be made using the measured QU data *or* using the polarisation fraction data, and with or without background subtraction.

Fits are initialised using the P0 and phi0 values from the Faraday depth spectrum. chi0 is initialised to zero.

MCMC convergence is assessed using the auto-correlation length of the chains; a burn-in of 5 times the autocorrelation length is discarded and chains are thinned by a factor of 15 in order to calculate final posterior distributions. Posterior uncertainties at a level of 1-sigma are provided for MAP estimates.

Fits can be made to either a `single` source, or to `all` sources in the pol_detections catalogue.

## Details

QUFit uses a [configuration file](https://github.com/as595/QUFitting/blob/main/configs/xmmlss13.cfg) to define the required fitting. The configuration file defines where the spectral data are located and where the MIGHTEE-POL source catalogue is located.

QUFit expects spectral data in the following format:

```
#  freq       I        Q        U        V       Qbgnd    Ubgnd    Vbgnd    Noise
 887.05     49.1      -3.8     12.2     -1.1      10.6      8.3      1.1     35.8
 889.56     48.8     -14.5    -25.0     24.8      13.1      8.5      0.4     37.0
 892.08     48.6     -22.8      7.6    -13.6      14.1      5.5      2.3     33.6
 894.60     48.4     -11.6     62.0     29.2       8.2      9.2      2.2     34.4
 897.11     48.1      36.8     71.5      2.0      10.4      5.6      4.1     33.7
...
```



