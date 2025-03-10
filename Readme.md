# Causal Temporal Regime Structure Learning
This is the source code to reproduce the experiments for ["Causal Temporal Regime Structure Learning"](https://arxiv.org/abs/2311.01412) by Abdellah Rahmani, and Pascal Frossard, AISTATS 2025.

CASTOR is designed to learn causal relationships, determine the number of regimes, and uncover their arrangement from MTS with multiple regimes, each corresponding to an MTS block. Unlike other methods, CASTOR does not require prior knowledge of the number of regimes or their indices. It optimizes the data log-likelihood using an expectation-maximization algorithm (EM), alternating between assigning regime indices (expectation step) and inferring causal relationships in each regime (maximization step). We prove that in our framework, comprising multivariate time series with multiple regimes modeled by Gaussian structural equation models with equal error variances, both the regimes and their corresponding graphs are identifiable up to a permutation of the regime labels.

The main contributions of this paper can be summarized as follows:
* We formulate a new SEMs and CGMs for MTS composed of multiple regimes.
* We present, CASTOR, the first method designed to learn the number of regimes, their indices and their corresponding DAG from MTS with multiple regimes.
* We show that the exact maximization of the score function identifies the ground truth regimes and graphs up to a permutation in the case of Gaussian noise with equal variance.
* We show that CASTOR outperforms state-of-the-art methods in a wide variety of conditions, including linear and non-linear causal relationships and different number of nodes and regimes on both synthetic and real-world datasets.

 # Tutorial:
 To use our code we provided a notebook tutorial that shows the different step


 # References:
 If you find this code useful, please cite the following paper:
 ```bibtex
@inproceedings{rahmanicausal,
  title={Causal Temporal Regime Structure Learning},
  author={Rahmani, Abdellah and Frossard, Pascal},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}
```

