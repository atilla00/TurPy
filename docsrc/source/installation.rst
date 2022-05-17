Installation
===========================

``TurPy`` currently supports:

- Python >= 3.7.1
- Unix, Windows


Install with pip::

    pip install turpy-nlpkit
    
If you want to use transformer based models using CPU::

    pip install turpy-nlpkit[deep]

If you want to use transformer based models using GPU, it is preferred that you use conda to install cuda dependencies::

    conda install pytorch cudatoolkit=11.3 -c pytorch
    pip install turpy-nlpkit[deep]