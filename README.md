# SpectralFunction
project on spectral function reconstruction from correlation data.

Discuss the ill-posedness of the reconstruction problem by performing continuous eigenstate decomposition, (aka. generalized Fourier Transform); 
Show Neural Network representation and MEM results in both generalized coordinate and momentum spaces.

Cite this work as,

L. Wang, S. Shi, and K. Zhou, *Reconstructing Spectral Functions via Automatic Differentiation*, ArXiv:2111.14760 [Hep-Lat, Physics:Hep-Ph] (2021).

S. Shi, L. Wang, and K. Zhou, *Rethinking the ill-posedness of the spectral function reconstruction - why is it fundamentally hard and how Artificial Neural Networks can help*, ArXiv:2201.02564 [Hep-Ph] (2022).

## Getting Started

The code requires Python >= 3.8 and PyTorch >= 1.2. You can configure on CPU machine and accelerate with a recent Nvidia GPU card.

## Running the tests

The code consist of two parts, MEM result generator and NN result generator.

**0.** Input files: although more files are provided in ```data/```, the following files are needed for running step **1** and **2**.
```
data/fig3/
  True_D.txt, True_rho.txt
data/fig4/
  True_D.txt, True_rho.txt, True_rho_D_s.txt
data/fig5/
  True_D.txt, True_rho.txt, True_rho_D_s.txt
```

**1**. Generate NN results
```python

```

**2.** Generate MEM results and perform generalized Fourier Transformation on NN results
```python
python mem.py
```

**3.** Make the plots including analytical results, run the mathematica notebook ```plot.nb```.
* Note 1: output files from step **1** and **2** are also included in ```data/fig*/``` directories. One may make the plots directly without going through the data-generation steps.
* Note 2: In order to load the plotting scripts, ```plot.nb``` and ```mathematica_package/``` shall be put in the same directory.


## License

This project is licensed under the MIT License - see the LICENSE file for details
