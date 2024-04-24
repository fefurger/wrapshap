## How to install Wrapshap

First download this repository. Then navigate to the root of the wrapshap folder where `setup.py` should be.

In order to install the normal CPU implementation of Wrapshap, please just run:

```bash
pip install .
```

If you can afford it, the GPU implementation is much faster and comes with additional methods. To install it instead, please run:

```bash
pip install .[gpu]
```

You should now be able to import wrapshap utilities.

### For the CPU implementation:

```python
from wrapshap import Wrapshap
```
or for the already wrapped models:

```python
from wrapshap.already_wrapped import WrappedBinaryClassifierXGB, WrappedRegressorXGB
```

### For the GPU implementation:

```python
from wrapshap.gpu_implementation import WrapshapGPU
```
or for the already wrapped models:

```python
from wrapshap.gpu_implementation.already_wrapped_gpu import WrappedRegressorXGB_GPU, WrappedBinaryClassifierXGB_GPU
```

For more details about how to use the package, please refer to the example notebook: `example_notebooks/Wrapshap_example.ipynb`
