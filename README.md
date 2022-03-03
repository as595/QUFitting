# QUFitting
QU Fitting for MeerKAT MIGHTEE-POL

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

## Details

QUFit uses a [configuration file]() to define the required fitting. The configuration file defines where the spectral data are located and where the MIGHTEE-POL source catalogue is located.

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



