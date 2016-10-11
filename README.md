# Preamble

This code is the one used to generate results presented in the paper 
*Efficient Temporal Kernels between Feature Sets for Time Series Classification*.
When using this code, please cite:
```
TODO
```

# Requirements
For this code to run properly, the following python packages should be installed:
```
numpy  
scipy  
sklearn
```

Also, if one wants to run experiments on the UCR dataset, she should download it from 
[here](http://www.cs.ucr.edu/~eamonn/time_series_data/) and paste it (preserving its subfolder structure) to `datasets/ucr`.
Then, using the software from <https://github.com/a-bailly/dbotsw>, she should generate time-sensitive features into the folder `datasets/ucr_t` with class information inside `datasets/ucr_classes`.

# Running
## Baseline (BoW)
To run the baseline on dataset `FISH` with $k=1024$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/
EXECUTABLE=${SOURCEDIR}/ucr_bow.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 1024
```

## SQFD-KM
To run the _SQFD-KM_ method on dataset `FISH` with $D=1024$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/
EXECUTABLE=${SOURCEDIR}/ucr_sqfd_km.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 1024
```

## SQFD-Nystroem
To run the _SQFD-Nystroem_ method on dataset `FISH` with $D=1024$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/
EXECUTABLE=${SOURCEDIR}/ucr_sqfd_nystroem.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 1024
```

## SQFD-Fourier
To run the _SQFD-Fourier_ method on dataset `FISH` with $D=1024$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/
EXECUTABLE=${SOURCEDIR}/ucr_sqfd_fourier.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 1024
```