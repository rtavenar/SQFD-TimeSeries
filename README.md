# Preamble

This code is the one used to generate results presented in the paper 
*Efficient Temporal Kernels between Feature Sets for Time Series Classification*.
When using this code, please cite:
```
@inproceedings{tavenard:halshs-01561461,
  TITLE = {{Efficient Temporal Kernels between Feature Sets for Time Series Classification}},
  AUTHOR = {Tavenard, Romain and Malinowski, Simon and Chapel, Laetitia and Bailly, Adeline and Sanchez, Heider and Bustos, Benjamin},
  URL = {https://halshs.archives-ouvertes.fr/halshs-01561461},
  BOOKTITLE = {{European Conference on Machine Learning and Principles and Practice of Knowledge Discovery}},
  ADDRESS = {Skopje, Macedonia},
  YEAR = {2017},
  MONTH = Sep,
  KEYWORDS = {Time series classification},
  PDF = {https://halshs.archives-ouvertes.fr/halshs-01561461/file/paper.pdf},
  HAL_ID = {halshs-01561461},
  HAL_VERSION = {v1},
}
```

# Supplementary material

Supplementary material for this paper is available [here](https://github.com/rtavenar/SQFD-TimeSeries/blob/master/ecml_sqfd_supp.pdf).

# Requirements
For this code to run properly, the following python packages should be installed:
```
numpy  
scipy  
sklearn
```

Also, if one wants to run experiments on the UCR dataset, she should download it from 
[here](http://www.cs.ucr.edu/~eamonn/time_series_data/).
Then, using the software from <https://github.com/a-bailly/dbotsw>, she should generate time-sensitive features (_cf._ `get_feature_vectors`) into the folder `datasets/ucr_t` with class information inside `datasets/ucr_classes`.

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
To run the _SQFD-KM_ method on dataset `FISH` with $k=256$, run:
```bash
SOURCEDIR=/path/to/the/base/dir/of/the/project/
WORKINGDIR=${SOURCEDIR}/
EXECUTABLE=${SOURCEDIR}/ucr_sqfd_km.py
export PYTHONPATH="${PYTHONPATH}:${SOURCEDIR}"
cd ${WORKINGDIR}
python ${EXECUTABLE} FISH 256
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
