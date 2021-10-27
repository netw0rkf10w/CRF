# CRF - Conditional Random Fields
A library for dense conditional random fields (CRFs).

This is the official 
accompanying code for the paper **Regularized Frank-Wolfe for Dense CRFs: Generalizing Mean Field and Beyond** published at NeurIPS 2021 by Đ.Khuê Lê-Huu and Karteek Alahari. Please cite this paper if you use any part of this code, using the following BibTeX entry:

```
@inproceedings{lehuu2021regularizedFW,
  title={Regularized Frank-Wolfe for Dense CRFs: Generalizing Mean Field and Beyond},
  author={L\^e-Huu, \DJ.Khu\^e and Alahari, Karteek},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

Currently the code is messy and undocumented, and we apology for that. We will
make an effort to fix this soon. To facilitate the maintenance, the code and pre-trained models for the semantic segmentation task will be available in a separate repository.

## Installation

```bash
git clone https://github.com/netw0rkf10w/CRF.git
cd CRF
python setup.py install
```

## Usage

After having installed the package, you can create a CRF layer as follows:

```python
import CRF

params = CRF.FrankWolfeParams(scheme='fixed', # constant stepsize
            stepsize=1.0,
            regularizer='l2',
            lambda_=1.0, # regularization weight
            lambda_learnable=False,
            x0_weight=0.5, # useful for training, set to 0 if inference only
            x0_weight_learnable=False)

crf = CRF.DenseGaussianCRF(classes=21,
                alpha=160,
                beta=0.05,
                gamma=3.0,
                spatial_weight=1.0,
                bilateral_weight=1.0,
                compatibility=1.0,
                init='potts',
                solver='fw',
                iterations=5,
                params=params)
```

Detailed documentation on the available options will be added later.

Below is an example of how to use this layer in combination with a CNN. We can
define for example the following simple CNN-CRF module:

```python
import torch

class CNNCRF(torch.nn.Module):
    """
    Simple CNN-CRF model
    """
    def __init__(self, cnn, crf):
        super().__init__()
        self.cnn = cnn
        self.crf = crf

    def forward(self, x):
        """
        x is a batch of input images
        """
        logits = self.cnn(x)
        logits = self.crf(x, logits)
        return logits

# Create a CNN-CRF model from given `cnn` and `crf`
# This is a PyTorch module that can be used in a usual way
model = CNNCRF(cnn, crf)
```


## Acknowledgements

The CUDA implementation of the permutohedral lattice is due to https://github.com/MiguelMonteiro/permutohedral_lattice. An initial version of our permutohedral layer was based on https://github.com/Fettpet/pytorch-crfasrnn.