# pytorch_ssn
A pytorch version of SSN (Superpixel Sampling Networks)
The data preparation is same as https://github.com/NVlabs/ssn_superpixels.git.
To enforce connectivity in superpixels, the cython script takes from official code.

To simplify the implementation, each init superpixel has the same number of pixels during the training.
