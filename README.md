# SpectralFunction
project on spectral function reconstruction from correlation data.

Discuss the ill-posedness of the reconstruction problem by performing continuous eigenstate decomposition, (aka. generalized Fourier Transform).
Show Neural Network representation and MEM results in both generalized coordinate and momentum spaces.

Cite this work as,<br>
* L. Wang, S. Shi, and K. Zhou, <br>
*Reconstructing Spectral Functions via Automatic Differentiation*,<br>
ArXiv:2111.14760 [Hep-Lat, Physics:Hep-Ph] (2021). <br>
Link to HEPinsipre(https://inspirehep.net/literature/1978876).<br>
* S. Shi, L. Wang, and K. Zhou, <br>
*Rethinking the ill-posedness of the spectral function reconstruction - why is it fundamentally hard and how Artificial Neural Networks can help*, <br>
ArXiv:2201.02564 [Hep-Ph] (2022). <br>
Link to HEPinsipre(https://inspirehep.net/literature/2005569).

## Running the tests

The code consist of two parts, MEM result generator and NN result generator. In what follows, we show examples of how to repeat the plots in 2201.02564.

### 0. Input files

Although more files are provided in ```data/```, the following files are needed for running step **1** and **2**.

Needed for the reconstruction code:
```
data/fig3/True_D.txt, data/fig4/True_D.txt, data/fig5/True_D.txt
```
Needed for plotting: 
```
data/fig3/True_rho.txt
data/fig4/
  True_rho.txt, True_rho_D_s.txt
data/fig5/
  True_rho.txt, True_rho_D_s.txt
```

### 1. Generate NN results
For fig. 3 (The depth parameter is $d = l-1$, which can be chosen from 0 to 3.)
```python
python NN_comp.py --width 64 --depth 0
python NN_comp.py --width 64 --depth 1
python NN_comp.py --width 64 --depth 2
python NN_comp.py --width 64 --depth 3
```
For fig. 4
```python
python NNspectrum.py --noise 0 --l2 1E-2 --maxiter 5000
python P2Pspectrum.py --noise 0 --l2 1E-2
```
For fig. 5
```python
python NNspectrum.py --noise 1 --l2 1E-2 --maxiter 1000
python P2Pspectrum.py --noise 1 --l2 1E-2
```

* Note: The code requires Python >= 3.8 and PyTorch >= 1.2. You can configure on CPU machine and accelerate with a recent Nvidia GPU card. 

### 2. Generate MEM results and perform generalized Fourier Transformation on NN results

```python
python mem.py
```

### 3. Make the plots including analytical results.

Run the mathematica notebook ```plot.nb```.

* Note 1: output files from step **1** and **2** are also included in ```data/fig*/``` directories. One may make the plots directly without going through the data-generation steps.
* Note 2: In order to load the plotting scripts, ```plot.nb``` and ```mathematica_package/``` shall be put in the same directory.


## License

This project is licensed under the MIT License - see the LICENSE file for details
