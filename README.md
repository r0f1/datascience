# Data Science

## Machine Learning

### Core
[numpy](https://www.numpy.org/) | Fundamental package for scientific computing with Python.  
[pandas](https://pandas.pydata.org/) | Data structures and data analysis tools for Python.  
[scikit-learn](https://scikit-learn.org/stable/) | Core ML library  
[matplotlib](https://matplotlib.org/) | Plotting library.  
[seaborn](https://seaborn.pydata.org/) | Python data visualization library based on matplotlib  
[pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) | Descriptive statistics using `ProfileReport`.  
[sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) | Helpful `DataFrameMapper` class.  

### Exploration and Cleaning
[missingno](https://github.com/ResidentMario/missingno) | Missing data visualization. 
[fancyimpute](https://github.com/iskandr/fancyimpute) | Matrix completion and imputation algorithms.  
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | Resampling for imbalanced datasets.

### Feature Engineering and Selection
[categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding) | Categorical encoding of variables.  
[patsy](https://github.com/pydata/patsy/) | R-like syntax for statistical models.   
[featuretools](https://github.com/Featuretools/featuretools) | Automated feature engineering.   
[scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) | Relief-based feature selection algorithms.  

### Dimensionality Reduction
prince										Dimensionality Reduction, PCA, MCA, CA, FAMD
tsne https://lvdmaaten.github.io/tsne/
tsne https://github.com/DmitryUlyanov/Multicore-TSNE
Dimensionality Reduction, Multifactor Dimensionality Reduction (MDR) https://github.com/EpistasisLab/scikit-mdr


### Big Data
dask										Pandas for big data.
turicreate										Helpful SFrame class for out-of-memory dataframes.
[ray](https://github.com/ray-project/ray/) | Flexible, high-performance distributed execution framework.  
[ni](https://github.com/spencertipping/ni) | Command line tool for big data


### Frameworks
h2o										General purpose ML
turicreate									
dask-ml										ML with Dask

### Recommender Systems

### Visualization
physt										Better Histograms
yellowbrick										Wrapper for matplotlib for diagnosic ML plots
altair
folium										Plot geographical maps

### Decision Trees

catboost										Gradient Boosting implementation with a focus on categorical features
lightgbm										DecisionTrees
xgboost										DecisionTrees
forestci										Confidence intervals for RandomForests
h2o
scikit-garden										QuantileRegressionForests
# https://sites.google.com/view/lauraepp/parameters

### Text Processing

gensim										Topic Modelling
pyldavis										Visualization for Topic Modelling
spaCy										NLP
fasttext										NLP (word embeddings)


### Automated Machine Learning

auto_ml										Automated ML

### Evolutionary Algorithms

deap										Evolutionary Algorithms
evol										Evolutionary Algorithms


### Neural Networks

tensorflow
keras										Neural Networks

### Time Series

prophet										Facebook's time series prediction library
tsfresh										Time Series feature engineering
astroml										Lomb Scargle Periodogram
https://github.com/RJT1990/pyflux

#### Survival Analysis

lifelines										Cox PH Regression
# Predicting conversion rates, Survival analysis
# https://better.engineering/convoys/

### Bayes
PyMC3 https://docs.pymc.io/notebooks/getting_started

### Outlier Detection & Anomaly Detection

pyod										Outlier Detection / Anomaly Detection


### Stacking Models

mlxtend										EnsembleVoteClassifier
StackNet										Stacking ML models
vecstack										Stacking ML models

### Model Evaluation & Explaining Models

pandas_ml										ConfusionMatrix
eli5										Explain predictions of ML models
shap										Explain predictions of ML models

### Hyperparameter Tuning

skopt										BayesSearchCV for Hyperparameter Search
[tune](https://ray.readthedocs.io/en/latest/tune.html) | Scalable framework for hyperparameter search with a focus on deep learning and deep reinforcement learning.  


### Regression

# Multivariate Adaptive Regression Splines (MARS)
# https://uc-r.github.io/mars
# https://github.com/scikit-learn-contrib/py-earth

### Clustering

somoclu										Self-organizing Maps
kmodes										Variant of kmeans for categorical data
https://github.com/annoviko/pyclustering


### Reinforcement Learning

[RLLib](https://ray.readthedocs.io/en/latest/rllib.html) | Library for reinforcement learning  


### Other

xarray
daft - Render some probabilistic graphical models using matplotlib
https://stadiamaps.com/ - Maps, Geo
https://github.com/csurfer/blackcellmagic


https://github.com/spotify/chartify/
# Dill, Serialization, Alternative to pickle
# https://pypi.org/project/dill/
# Working with units https://github.com/yt-project/unyt
# Multiobjective Optimization
# https://platypus.readthedocs.io/en/latest/
https://github.com/modin-project/modin
https://mlflow.org/
# http://explained.ai/decision-tree-viz/index.html
# https://github.com/parrt/animl
https://panel.pyviz.org/index.html
geocoder # https://github.com/DenisCarriere/geocoder

# https://github.com/python-visualization/folium
# https://github.com/bokeh/datashader

# Spatial Data Analysis With Python - Dillon R Gardner, PhD
# https://www.youtube.com/watch?v=eHRggqAvczE
# https://github.com/dillongardner/PyDataSpatialAnalysis

# Low Level Geospatial Tools
# GEOS
# GDAL/OGR
# PROJ.4
# 
# Vector Data
# Shapely
# Fiona
# Pyproj
# 
# Raster Data
# Rasterio
# 
# Misc
# Geopandas
# Plotting: Descartes, Catropy

# https://github.com/njanakiev/osm-predict-economic-measurements/blob/master/osm-predict-economic-indicators.ipynb
# https://github.com/python-visualization/folium
# https://github.com/bokeh/datashader
# https://github.com/DenisCarriere/geocoder

# Surprise - https://github.com/NicolasHug/Surprise # https://www.youtube.com/watch?v=d7iIb_XVkZs

# Besserer Artikel: https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/
# https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

# Intro article: https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223
# Jupyter notebook from article: https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb


# https://github.com/benfred/implicit
# https://github.com/maciejkula/spotlight
# https://github.com/apple/turicreate/blob/master/README.md
# https://github.com/Mendeley/mrec
# https://github.com/ocelma/python-recsys
# https://muricoca.github.io/crab/
# https://github.com/lyst/lightfm
# https://www.kaggle.com/morrisb/how-to-recommend-anything-deep-recommender
# https://www.pinterest.de/dataliftoff/recommender-systems/

# https://www.pinterest.de/dataliftoff/recommender-systems/




# HDBSCAN
from hdbscan import HDBSCAN




# https://uc-r.github.io/mars
# https://www.depends-on-the-definition.com/multivariate-adaptive-regression-splines/


# https://scikit-garden.github.io/examples/QuantileRegressionForests/#example


# https://www.youtube.com/watch?v=m-tAASQA7XQ&t=18m57s
# Saved notebook from this talk is in folder bak
# https://github.com/scikit-multilearn/scikit-multilearn


# http://www.ritchieng.com/machinelearning-learning-curve/
# http://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html




# https://github.com/AustinRochford/PyCEbox

# https://github.com/SauceCat/PDPbox
# https://github.com/nyuvis/partial_dependence
# https://www.kaggle.com/dansbecker/partial-plots

# http://explained.ai/rf-importance/index.html


# https://christophm.github.io/interpretable-ml-book/agnostic.html


# Feature Importance for RandomForests using Permuation Importance
# https://github.com/parrt/random-forest-importances

# https://github.com/tmadl/sklearn-expertsys 
# https://github.com/tmadl/sklearn-interpretable-tree


# https://github.com/jphall663/interpretable_machine_learning_with_python
# https://github.com/jphall663/lime_xgboost
# https://github.com/jphall663/awesome-machine-learning-interpretability

# https://github.com/datascienceinc/Skater
# https://github.com/AustinRochford/PyCEbox
# https://github.com/SauceCat/PDPbox
# https://github.com/MI2DataLab/pyBreakDown
# https://github.com/marcotcr/lime
# https://github.com/Jianbo-Lab/L2X
# https://github.com/adebayoj/fairml
# https://github.com/TeamHG-Memex/eli5
# https://github.com/MarcelRobeer/ContrastiveExplanation
# https://github.com/marcotcr/anchor

# https://github.com/andosa/treeinterpreter

# https://github.com/sigopt/sigopt-sklearn

# https://github.com/godatadriven/evol
# https://www.youtube.com/watch?v=68ABAU_V8qI&t=11m49s




# Another Pipeline Example
# https://github.com/jem1031/pandas-pipelines-custom-transformers



# https://github.com/swager/grf


# https://github.com/lmcinnes/umap

from sklearn.manifold import TSNE
# https://github.com/DmitryUlyanov/Multicore-TSNE
# https://github.com/lvdmaaten/bhtsne



# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html


# https://github.com/sebp/scikit-survival
# R https://stats.stackexchange.com/questions/101353/cox-regression-with-time-varying-covariates
# https://lifelines.readthedocs.io/en/latest/Quickstart.html#survival-regression
# http://www.hammerlab.org/2017/06/26/introducing-survivalstan/
# https://www.youtube.com/watch?v=fli-yE5grtY
# https://statmd.wordpress.com/2015/05/02/survival-analysis-with-generalized-additive-models-part-iii-the-baseline-hazard/
# https://github.com/dswah/pyGAM
# RandomSurvivalForests? 
## R packages: randomForestSRC and ggRandomForests


# https://www.youtube.com/watch?v=E4NMZyfao2c&t=20m

# https://www.astroml.org/gatspy/
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html

# Has been generalized to Baysian Lomb Scargle
# (result of quick google search): https://github.com/mfouesneau/bgls



# Add predictions of models
# statsmodels decompose - https://gist.github.com/balzer82/5cec6ad7adc1b550e7ee
# facebook prophet - https://github.com/facebook/prophet
# SARIMAX - https://github.com/tgsmith61591/pyramid
# GARCH - https://pyflux.readthedocs.io/en/latest/garch.html
# shapelets - https://github.com/IBCNServices/GENDIS/blob/master/gendis/example.ipynb
# pastas - https://pastas.readthedocs.io/en/latest/examples.html
# tensorflow and lstms - https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/timeseries
#                        https://github.com/hzy46/TensorFlow-Time-Series-Examples


# Misc
# Preprocessing: Denoising, Compression, Resampling - https://github.com/MaxBenChrist/tspreprocess
# Extract features, tsfresh - https://github.com/blue-yonder/tsfresh
# Learning features, clustering of different time series - https://tslearn.readthedocs.io/en/latest/auto_examples/index.html

# Working with dates and time
# https://github.com/sdispater/pendulum
# https://github.com/crsmithdev/arrow

# Time Series Prediction
# https://github.com/RJT1990/pyflux

# Anomaly detection
# https://github.com/rob-med/awesome-TS-anomaly-detection
# https://github.com/twitter/AnomalyDetection






# Visualization library
# http://holoviews.org/

# Austrian Monuments
# https://github.com/njanakiev/austrian-monuments-visualization


# https://machinelearningmastery.com/feature-selection-machine-learning-python/
# https://github.com/EpistasisLab/scikit-rebate
# https://github.com/mutantturkey/PyFeast
# http://featureselection.asu.edu/ # https://github.com/jundongl/scikit-feature
# https://github.com/scikit-learn-contrib/boruta_py # Feature Importance
# https://stats.stackexchange.com/questions/264360/boruta-all-relevant-feature-selection-vs-random-forest-variables-of-importanc/264467)
# https://www.kaggle.com/tilii7/boruta-feature-elimination



# Web Scraping Library
# https://github.com/scrapy/scrapy


# https://github.com/BurntSushi/xsv         # Big data
# https://csvkit.readthedocs.io/en/1.0.3/   # Big data

https://github.com/scikit-learn-contrib/lightning
https://github.com/scikit-learn-contrib/stability-selection
https://github.com/scikit-learn-contrib/skope-rules
https://github.com/scikit-learn-contrib/boruta_py
https://github.com/scikit-learn-contrib/polylearn

# Articles

## Bayes

https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html
