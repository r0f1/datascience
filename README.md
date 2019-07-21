# Awesome Data Science with Python

> A curated list of awesome resources for practicing data science using Python, including not only libraries, but also links to tutorials, code snippets, blog posts and talks.  

#### Core
[pandas](https://pandas.pydata.org/) - Data structures built on top of [numpy](https://www.numpy.org/).  
[scikit-learn](https://scikit-learn.org/stable/) - Core ML library.  
[matplotlib](https://matplotlib.org/) - Plotting library.  
[seaborn](https://seaborn.pydata.org/) - Data visualization library based on matplotlib.  
[pandas_summary](https://github.com/mouradmourafiq/pandas-summary) - Basic statistics using `DataFrameSummary(df).summary()`.  
[pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) - Descriptive statistics using `ProfileReport`.  
[sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) - Helpful `DataFrameMapper` class.  
[janitor](https://pyjanitor.readthedocs.io/) - Clean messy column names.  
[missingno](https://github.com/ResidentMario/missingno) - Missing data visualization.  

#### Pandas and Jupyter
General tricks: [link](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)  
Python debugger (pdb) - [blog post](https://www.blog.pythonlibrary.org/2018/10/17/jupyter-notebook-debugging/), [video](https://www.youtube.com/watch?v=Z0ssNAbe81M&t=1h44m15s), [cheatsheet](https://nblock.org/2011/11/15/pdb-cheatsheet/)  
[cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) - Project template for data science projects.  
[nteract](https://nteract.io/) - Open Jupyter Notebooks with doubleclick.  
[modin](https://github.com/modin-project/modin) - Parallelization library for faster pandas `DataFrame`.  
[swifter](https://github.com/jmcarpenter2/swifter) - Apply any function to a pandas dataframe faster.  
[xarray](https://github.com/pydata/xarray/) - Extends pandas to n-dimensional arrays.  
[blackcellmagic](https://github.com/csurfer/blackcellmagic) - Code formatting for jupyter notebooks.  
[pivottablejs](https://github.com/nicolaskruchten/jupyter_pivottablejs) - Drag n drop Pivot Tables and Charts for jupyter notebooks.  
[qgrid](https://github.com/quantopian/qgrid) - Pandas `DataFrame` sorting.  
[ipysheet](https://github.com/QuantStack/ipysheet) - Jupyter spreadsheet widget.  
[nbdime](https://github.com/jupyter/nbdime) - Diff two notebook files, Alternative GitHub App: [ReviewNB](https://www.reviewnb.com/).  
[RISE](https://github.com/damianavila/RISE) - Turn Jupyter notebooks into presentations.  

#### Helpful
[pyprojroot](https://github.com/chendaniely/pyprojroot) - Helpful `here()` command from R.  
[intake](https://github.com/intake/intake) - Loading datasets made easier, [talk](https://www.youtube.com/watch?v=s7Ww5-vD2Os&t=33m40s). 

#### Extraction
[textract](https://github.com/deanmalmgren/textract) - Extract text from any document.  
[camelot](https://github.com/socialcopsdev/camelot) - Extract text from PDF.  

#### Big Data
[spark](https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html#work-with-dataframes) - `DataFrame` for big data, [cheatsheet](https://gist.github.com/crawles/b47e23da8218af0b9bd9d47f5242d189), [tutorial](https://github.com/ericxiao251/spark-syntax).  
[sparkit-learn](https://github.com/lensacom/sparkit-learn), [spark-deep-learning](https://github.com/databricks/spark-deep-learning) - ML frameworks for spark.  
[koalas](https://github.com/databricks/koalas) - Pandas API on Apache Spark.  
[dask](https://github.com/dask/dask), [dask-ml](http://ml.dask.org/) - Pandas `DataFrame` for big data and machine learning library, [resources](https://matthewrocklin.com/blog//work/2018/07/17/dask-dev), [talk1](https://www.youtube.com/watch?v=ccfsbuqsjgI), [talk2](https://www.youtube.com/watch?v=RA_2qdipVng), [notebooks](https://github.com/dask/dask-ec2/tree/master/notebooks), [videos](https://www.youtube.com/user/mdrocklin).  
[dask-gateway](https://github.com/jcrist/dask-gateway) - Managing dask clusters.  
[turicreate](https://github.com/apple/turicreate) - Helpful `SFrame` class for out-of-memory dataframes.  
[h2o](https://github.com/h2oai/h2o-3) - Helpful `H2OFrame` class for out-of-memory dataframes.  
[datatable](https://github.com/h2oai/datatable) - Data Table for big data support.  
[cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library.  
[ray](https://github.com/ray-project/ray/) - Flexible, high-performance distributed execution framework.  
[mars](https://github.com/mars-project/mars) - Tensor-based unified framework for large-scale data computation.  
[bottleneck](https://github.com/kwgoodman/bottleneck) - Fast NumPy array functions written in C.   
[bolz](https://github.com/Blosc/bcolz) - A columnar data container that can be compressed.  
[cupy](https://github.com/cupy/cupy) - NumPy-like API accelerated with CUDA.  
[vaex](https://github.com/vaexio/vaex) - Out-of-Core DataFrames.  
[petastorm](https://github.com/uber/petastorm) - Data access library for parquet files by Uber.  
[zappy](https://github.com/lasersonlab/zappy) - Distributed numpy arrays.  

##### Command line tools
[ni](https://github.com/spencertipping/ni) - Command line tool for big data.  
[xsv](https://github.com/BurntSushi/xsv) - Command line tool for indexing, slicing, analyzing, splitting and joining CSV files.  
[csvkit](https://csvkit.readthedocs.io/en/1.0.3/) - Another command line tool for CSV files.  
[csvsort](https://pypi.org/project/csvsort/) - Sort large csv files.  

#### Classical Statistics
[researchpy](https://github.com/researchpy/researchpy) - Helpful `summary_cont()` function for summary statistics (Table 1).  
[scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) - Statistical post-hoc tests for pairwise multiple comparisons.  
[Bland-Altman Plot](http://www.statsmodels.org/dev/generated/statsmodels.graphics.agreement.mean_diff_plot.html) - Plot for agreement between two methods of measurement.    

#### Tests
[Blog post](https://lindeloev.github.io/tests-as-linear/)  
[scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) - Statistical tests.
[ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html), Tutorials: [One-way](https://pythonfordatascience.org/anova-python/), [Two-way](https://pythonfordatascience.org/anova-2-way-n-way/), [Type 1,2,3 explained](https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/).  

##### Visualizations
[Null Hypothesis Significance Testing (NHST) and Sample Size Calculation](https://rpsychologist.com/d3/NHST/)  
[Correlation](https://rpsychologist.com/d3/correlation/)  
[Cohen's d](https://rpsychologist.com/d3/cohend/)  
[Confidence Interval](https://rpsychologist.com/d3/CI/)  
[Equivalence, non-inferiority and superiority testing](https://rpsychologist.com/d3/equivalence/)  
[Bayesian two-sample t test](https://rpsychologist.com/d3/bayes/)  
[Distribution of p-values when comparing two groups](https://rpsychologist.com/d3/pdist/)  
[Understanding the t-distribution and its normal approximation](https://rpsychologist.com/d3/tdist/)     

#### Talks
[Inverse Propensity Weighting](https://www.youtube.com/watch?v=SUq0shKLPPs)  
[Dealing with Selection Bias By Propensity Based Feature Selection](https://www.youtube.com/watch?reload=9&v=3ZWCKr0vDtc)  

#### Exploration and Cleaning
[Checklist](https://github.com/r0f1/ml_checklist).  
[impyute](https://github.com/eltonlaw/impyute) - Imputations.  
[fancyimpute](https://github.com/iskandr/fancyimpute) - Matrix completion and imputation algorithms.  
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - Resampling for imbalanced datasets.  
[tspreprocess](https://github.com/MaxBenChrist/tspreprocess) - Time series preprocessing: Denoising, Compression, Resampling.  
[Kaggler](https://github.com/jeongyoonlee/Kaggler) - Utility functions (`OneHotEncoder(min_obs=100)`)  
[pyupset](https://github.com/ImSoErgodic/py-upset) - Visualizing intersecting sets.  
[pyemd](https://github.com/wmayner/pyemd) - Earth Mover's Distance, similarity between histograms.  

#### Feature Engineering
[Talk](https://www.youtube.com/watch?v=68ABAU_V8qI)  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) - Pipeline, [examples](https://github.com/jem1031/pandas-pipelines-custom-transformers).  
[pdpipe](https://github.com/shaypal5/pdpipe) - Pipelines for DataFrames.  
[scikit-lego](https://github.com/koaning/scikit-lego) - Custom transformers for pipelines.  
[few](https://github.com/lacava/few) - Feature engineering wrapper for sklearn.  
[skoot](https://github.com/tgsmith61591/skoot) - Pipeline helper functions.  
[categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding) - Categorical encoding of variables, [vtreat (R package)](https://cran.r-project.org/web/packages/vtreat/vignettes/vtreat.html).  
[dirty_cat](https://github.com/dirty-cat/dirty_cat) - Encoding dirty categorical variables.  
[patsy](https://github.com/pydata/patsy/) - R-like syntax for statistical models.  
[mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/) - LDA.  
[featuretools](https://github.com/Featuretools/featuretools) - Automated feature engineering, [example](https://github.com/WillKoehrsen/automated-feature-engineering/blob/master/walk_through/Automated_Feature_Engineering.ipynb).  
[tsfresh](https://github.com/blue-yonder/tsfresh) - Time series feature engineering.  
[pypeln](https://github.com/cgarciae/pypeln) - Concurrent data pipelines.  

#### Feature Selection
[Talk](https://www.youtube.com/watch?v=JsArBz46_3s)  
Blog post series - [1](http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/), [2](http://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/), [3](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/), [4](http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)  
Tutorials - [1](https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn), [2](https://machinelearningmastery.com/feature-selection-machine-learning-python/)  
[sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection) - Feature selection.  
[eli5](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html#feature-selection) - Feature selection using permutation importance.  
[scikit-feature](https://github.com/jundongl/scikit-feature) - Feature selection algorithms.  
[stability-selection](https://github.com/scikit-learn-contrib/stability-selection) - Stability selection.  
[scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) - Relief-based feature selection algorithms.  
[scikit-genetic](https://github.com/manuel-calzolari/sklearn-genetic) - Genetic feature selection.  
[boruta_py](https://github.com/scikit-learn-contrib/boruta_py) - Feature selection, [explaination](https://stats.stackexchange.com/questions/264360/boruta-all-relevant-feature-selection-vs-random-forest-variables-of-importanc/264467), [example](https://www.kaggle.com/tilii7/boruta-feature-elimination).  
[linselect](https://github.com/efavdb/linselect) - Feature selection package.  
[mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/) - Exhaustive feature selection.    

#### Dimensionality Reduction
[Talk](https://www.youtube.com/watch?v=9iol3Lk6kyU)  
[prince](https://github.com/MaxHalford/prince) - Dimensionality reduction, factor analysis (PCA, MCA, CA, FAMD).  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) - Multidimensional scaling (MDS).  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) - t-distributed Stochastic Neighbor Embedding (t-SNE), [intro](https://distill.pub/2016/misread-tsne/). Faster implementations: [lvdmaaten](https://lvdmaaten.github.io/tsne/), [MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE).  
[sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) - Truncated SVD (aka LSA).  
[mdr](https://github.com/EpistasisLab/scikit-mdr) - Dimensionality reduction, multifactor dimensionality reduction (MDR).  
[umap](https://github.com/lmcinnes/umap) - Uniform Manifold Approximation and Projection, [talk](https://www.youtube.com/watch?v=nq6iPZVUxZU), [explorer](https://github.com/GrantCuster/umap-explorer).  
[FIt-SNE](https://github.com/KlugerLab/FIt-SNE) - Fast Fourier Transform-accelerated Interpolation-based t-SNE.  
[scikit-tda](https://github.com/scikit-tda/scikit-tda) - Topological Data Analysis, [paper](https://www.nature.com/articles/srep01236), [talk](https://www.youtube.com/watch?v=F2t_ytTLrQ4), [talk](https://www.youtube.com/watch?v=AWoeBzJd7uQ).  

#### Visualization
[All charts](https://datavizproject.com/), [Austrian monuments](https://github.com/njanakiev/austrian-monuments-visualization).  
[cufflinks](https://github.com/santosjorge/cufflinks) - Dynamic visualization library, wrapper for [plotly](https://plot.ly/), [medium](https://towardsdatascience.com/the-next-level-of-data-visualization-in-python-dd6e99039d5e), [example](https://github.com/WillKoehrsen/Data-Analysis/blob/master/plotly/Plotly%20Whirlwind%20Introduction.ipynb).  
[physt](https://github.com/janpipek/physt) - Better histograms, [talk](https://www.youtube.com/watch?v=ZG-wH3-Up9Y), [notebook](https://nbviewer.jupyter.org/github/janpipek/pydata2018-berlin/blob/master/notebooks/talk.ipynb).  
[matplotlib_venn](https://github.com/konstantint/matplotlib-venn) - Venn diagrams, [alternative](https://github.com/penrose/penrose).  
[joypy](https://github.com/sbebo/joypy) - Draw stacked density plots.  
[mosaic plots](https://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html) - Categorical variable visualization, [example](https://sukhbinder.wordpress.com/2018/09/18/mosaic-plot-in-python/).  
[scikit-plot](https://github.com/reiinakano/scikit-plot) - ROC curves and other visualizations for ML models.  
[yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visualizations for ML models (similar to scikit-plot).  
[bokeh](https://bokeh.pydata.org/en/latest/) - Interactive visualization library, [Examples](https://bokeh.pydata.org/en/latest/docs/user_guide/server.html), [Examples](https://github.com/WillKoehrsen/Bokeh-Python-Visualization).  
[animatplot](https://github.com/t-makaro/animatplot) - Animate plots build on matplotlib.  
[plotnine](https://github.com/has2k1/plotnine) - ggplot for Python.  
[altair](https://altair-viz.github.io/) - Declarative statistical visualization library.  
[bqplot](https://github.com/bloomberg/bqplot) - Plotting library for IPython/Jupyter Notebooks.  
[hvplot](https://github.com/pyviz/hvplot) - High-level plotting library built on top of [holoviews](http://holoviews.org/).  
[dtreeviz](https://github.com/parrt/dtreeviz) - Decision tree visualization and model interpretation.  
[chartify](https://github.com/spotify/chartify/) - Generate charts.  
[VivaGraphJS](https://github.com/anvaka/VivaGraphJS) - Graph visualization (JS package).  
[pm](https://github.com/anvaka/pm) - Navigatable 3D graph visualization (JS package), [example](https://w2v-vis-dot-hcg-team-di.appspot.com/#/galaxy/word2vec?cx=5698&cy=-5135&cz=5923&lx=0.1127&ly=0.3238&lz=-0.1680&lw=0.9242&ml=150&s=1.75&l=1&v=hc).  
[python-ternary](https://github.com/marcharper/python-ternary) - Triangle plots.  
[falcon](https://github.com/uwdata/falcon) - Interactive visualizations for big data.  

#### Dashboards
[dash](https://dash.plot.ly/gallery) - Dashboarding solution by plot.ly. Tutorial: [1](https://www.youtube.com/watch?v=J_Cy_QjG6NE), [2](https://www.youtube.com/watch?v=hRH01ZzT2NI), [3](https://www.youtube.com/watch?v=wv2MXJIdKRY), [4](https://www.youtube.com/watch?v=37Zj955LFT0), [5](https://www.youtube.com/watch?v=luixWRpp6Jo), [example](https://github.com/ned2/slapdash)    
[bokeh](https://github.com/bokeh/bokeh) - Dashboarding solution.  
[visdom](https://github.com/facebookresearch/visdom) - Dashboarding library by facebook.  
[bowtie](https://github.com/jwkvam/bowtie/) - Dashboarding solution.  
[panel](https://panel.pyviz.org/index.html) - Dashboarding solution.  
[altair example](https://github.com/xhochy/altair-vue-vega-example) - [Video](https://www.youtube.com/watch?v=4L568emKOvs).  
[voila](https://github.com/QuantStack/voila) - Turn Jupyter notebooks into standalone web applications.  

#### Geopraphical Tools
[folium](https://github.com/python-visualization/folium) - Plot geographical maps using the Leaflet.js library, [jupyter plugin](https://github.com/jupyter-widgets/ipyleaflet).  
[gmaps](https://github.com/pbugnion/gmaps) - Google Maps for Jupyter notebooks.  
[stadiamaps](https://stadiamaps.com/) - Plot geographical maps.  
[datashader](https://github.com/bokeh/datashader) - Draw millions of points on a map.  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html) - BallTree, [Example](https://tech.minodes.com/experiments-with-in-memory-spatial-radius-queries-in-python-e40c9e66cf63).  
[pynndescent](https://github.com/lmcinnes/pynndescent) - Nearest neighbor descent for approximate nearest neighbors.  
[geocoder](https://github.com/DenisCarriere/geocoder) - Geocoding of addresses, IP addresses.  
Conversion of different geo formats: [talk](https://www.youtube.com/watch?v=eHRggqAvczE), [repo](https://github.com/dillongardner/PyDataSpatialAnalysis)  
[geopandas](https://github.com/geopandas/geopandas) - Tools for geographic data  
Low Level Geospatial Tools (GEOS, GDAL/OGR, PROJ.4)  
Vector Data (Shapely, Fiona, Pyproj)  
Raster Data (Rasterio)  
Plotting (Descartes, Catropy)  
Predict economic indicators from Open Street Map [ipynb](https://github.com/njanakiev/osm-predict-economic-measurements/blob/master/osm-predict-economic-indicators.ipynb).  
[PySal](https://github.com/pysal/pysal) - Python Spatial Analysis Library.  
[geography](https://github.com/ushahidi/geograpy) - Extract countries, regions and cities from a URL or text.  

#### Recommender Systems
Examples: [1](https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/), [2](https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223), [2-ipynb](https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb), [3](https://www.kaggle.com/morrisb/how-to-recommend-anything-deep-recommender).  
[surprise](https://github.com/NicolasHug/Surprise) - Recommender, [talk](https://www.youtube.com/watch?v=d7iIb_XVkZs).  
[turicreate](https://github.com/apple/turicreate) - Recommender.  
[implicit](https://github.com/benfred/implicit) - Fast Collaborative Filtering for Implicit Feedback Datasets.  
[spotlight](https://github.com/maciejkula/spotlight) - Deep recommender models using PyTorch.  
[lightfm](https://github.com/lyst/lightfm) - Recommendation algorithms for both implicit and explicit feedback.  
[funk-svd](https://github.com/gbolmier/funk-svd) - Fast SVD.  
[pywFM](https://github.com/jfloff/pywFM) - Factorization.  

#### Decision Tree Models
[Intro to Decision Trees and Random Forests](https://victorzhou.com/blog/intro-to-random-forests/), [Intro to Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)    
[lightgbm](https://github.com/Microsoft/LightGBM) - Gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, [doc](https://sites.google.com/view/lauraepp/parameters).  
[xgboost](https://github.com/dmlc/xgboost) - Gradient boosting (GBDT, GBRT or GBM) library, [doc](https://sites.google.com/view/lauraepp/parameters), Methods for CIs: [link1](https://stats.stackexchange.com/questions/255783/confidence-interval-for-xgb-forecast), [link2](https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b).  
[catboost](https://github.com/catboost/catboost) - Gradient boosting.  
[thundergbm](https://github.com/Xtra-Computing/thundergbm) - GBDTs and Random Forest.  
[h2o](https://github.com/h2oai/h2o-3) - Gradient boosting.  
[forestci](https://github.com/scikit-learn-contrib/forest-confidence-interval) - Confidence intervals for random forests.  
[scikit-garden](https://github.com/scikit-garden/scikit-garden) - Quantile Regression.  
[grf](https://github.com/grf-labs/grf) - Generalized random forest.  
[dtreeviz](https://github.com/parrt/dtreeviz) - Decision tree visualization and model interpretation.  
[rfpimp](https://github.com/parrt/random-forest-importances) - Feature Importance for RandomForests using Permuation Importance.  
Why the default feature importance for random forests is wrong: [link](http://explained.ai/rf-importance/index.html)  
[treeinterpreter](https://github.com/andosa/treeinterpreter) - Interpreting scikit-learn's decision tree and random forest predictions.  
[bartpy](https://github.com/JakeColtman/bartpy) - Bayesian Additive Regression Trees.  
[infiniteboost](https://github.com/arogozhnikov/infiniteboost) - Combination of RFs and GBDTs.  
[merf](https://github.com/manifoldai/merf) - Mixed Effects Random Forest for Clustering, [video](https://www.youtube.com/watch?v=gWj4ZwB7f3o)  
[rrcf](https://github.com/kLabUM/rrcf) - Robust Random Cut Forest algorithm for anomaly detection on streams.  

#### Natural Language Processing (NLP) / Text Processing
[talk](https://www.youtube.com/watch?v=6zm9NC9uRkk)-[nb](https://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb), [nb2](https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html), [talk](https://www.youtube.com/watch?time_continue=2&v=sI7VpFNiy_I).  
[Text classification Intro](https://mlwhiz.com/blog/2018/12/17/text_classification/), [Preprocessing blog post](https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/).  
[gensim](https://radimrehurek.com/gensim/) - NLP, doc2vec, word2vec, text processing, topic modelling (LSA, LDA), [Example](https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html), [Coherence Model](https://radimrehurek.com/gensim/models/coherencemodel.html) for evaluation.  
Embeddings - [GloVe](https://nlp.stanford.edu/projects/glove/) ([[1](https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout)], [[2](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)]), [StarSpace](https://github.com/facebookresearch/StarSpace), [wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).  
[magnitude](https://github.com/plasticityai/magnitude) - Vector embedding utility package.  
[pyldavis](https://github.com/bmabey/pyLDAvis) - Visualization for topic modelling.  
[spaCy](https://spacy.io/) - NLP.  
[NTLK](https://www.nltk.org/) - NLP, helpful `KMeansClusterer` with `cosine_distance`.  
[pytext](https://github.com/facebookresearch/PyText) - NLP from Facebook.  
[fastText](https://github.com/facebookresearch/fastText) - Efficient text classification and representation learning.  
[annoy](https://github.com/spotify/annoy) - Approximate nearest neighbor search.  
[faiss](https://github.com/facebookresearch/faiss) - Approximate nearest neighbor search.  
[pysparnn](https://github.com/facebookresearch/pysparnn) - Approximate nearest neighbor search.  
[infomap](https://github.com/mapequation/infomap) - Cluster (word-)vectors to find topics, [example](https://github.com/mapequation/infomap/blob/master/examples/python/infomap-examples.ipynb).  
[datasketch](https://github.com/ekzhu/datasketch) - Probabilistic data structures for large data (MinHash, HyperLogLog).  
[flair](https://github.com/zalandoresearch/flair) - NLP Framework by Zalando.  
[stanfordnlp](https://github.com/stanfordnlp/stanfordnlp) - NLP Library.  

##### Papers
[Search Engine Correlation](https://arxiv.org/pdf/1107.2691.pdf)  

#### Biology

##### Sequencing
[scanpy](https://github.com/theislab/scanpy) - Analyze single-cell gene expression data, [tutorial](https://github.com/theislab/single-cell-tutorial).  

##### Image-related
[mahotas](http://luispedro.org/software/mahotas/) - Image processing (Bioinformatics), [example](https://github.com/luispedro/python-image-tutorial/blob/master/Segmenting%20cell%20images%20(fluorescent%20microscopy).ipynb).   
[imagepy](https://github.com/Image-Py/imagepy) - Software package for bioimage analysis.  
[CellProfiler](https://github.com/CellProfiler/CellProfiler) - Biological image analysis.   
[imglyb](https://github.com/imglib/imglyb) - Viewer for large images, [talk](https://www.youtube.com/watch?v=Ddo5z5qGMb8), [slides](https://github.com/hanslovsky/scipy-2019/blob/master/scipy-2019-imglyb.pdf).  
[microscopium](https://github.com/microscopium/microscopium) - Unsupervised clustering of images + viewer, [talk](https://www.youtube.com/watch?v=ytEQl9xs8FQ).  

#### Image Processing
[Talk](https://www.youtube.com/watch?v=Y5GJmnIhvFk)  
[cv2](https://github.com/skvark/opencv-python) - OpenCV, classical algorithms: [Gaussian Filter](https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html), [Morphological Transformations](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html).  
[scikit-image](https://github.com/scikit-image/scikit-image) - Image processing.  

#### Neural Networks  

##### Tutorials
[Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)  
fast.ai course - [Lessons 1-7](https://course.fast.ai/videos/?lesson=1), [Lessons 8-14](http://course18.fast.ai/lessons/lessons2.html)  
[Tensorflow without a PhD](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd) - Neural Network course by Google.  
Feature Visualization: [Blog](https://distill.pub/2017/feature-visualization/), [PPT](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)  
[Tensorflow Playground](https://playground.tensorflow.org/)  
[Visualization of optimization algorithms](https://vis.ensmallen.org/)  

##### Image Related
[keras preprocessing](https://keras.io/preprocessing/image/) - Preprocess images.  
[imgaug](https://github.com/aleju/imgaug) - More sophisticated image preprocessing.  
[imgaug_extension](https://github.com/cadenai/imgaug_extension) - Extension for imgaug.  
[albumentations](https://github.com/albu/albumentations) - Wrapper around imgaug and other libraries.  
[Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library.  
[tcav](https://github.com/tensorflow/tcav) - Interpretability method.  
[cutouts-explorer](https://github.com/mgckind/cutouts-explorer) - Image Viewer.  

#### Text Related
[ktext](https://github.com/hamelsmu/ktext) - Utilities for pre-processing text for deep learning in Keras.  
[textgenrnn](https://github.com/minimaxir/textgenrnn) - Ready-to-use LSTM for text generation.  

##### Libs
[keras](https://keras.io/) - Neural Networks on top of [tensorflow](https://www.tensorflow.org/), [examples](https://gist.github.com/candlewill/552fa102352ccce42fd829ae26277d24).  
[keras-contrib](https://github.com/keras-team/keras-contrib) - Keras community contributions.  
[keras-tuner](https://github.com/keras-team/keras-tuner) - Hyperparameter tuning for Keras.  
[hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: Convenient hyperparameter optimization wrapper.  
[elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark.  
[tflearn](https://github.com/tflearn/tflearn) - Neural Networks on top of tensorflow.  
[tensorlayer](https://github.com/tensorlayer/tensorlayer) -  Neural Networks on top of tensorflow, [tricks](https://github.com/wagamamaz/tensorlayer-tricks).  
[tensorforce](https://github.com/reinforceio/tensorforce) - Tensorflow for applied reinforcement learning.  
[fastai](https://github.com/fastai/fastai) - Neural Networks in pytorch.  
[ignite](https://github.com/pytorch/ignite) - Highlevel library for pytorch.  
[skorch](https://github.com/dnouri/skorch) - Scikit-learn compatible neural network library that wraps pytorch, [talk](https://www.youtube.com/watch?v=0J7FaLk0bmQ), [slides](https://github.com/thomasjpfan/skorch_talk).  
[Detectron](https://github.com/facebookresearch/Detectron) - Object Detection by Facebook.  
[autokeras](https://github.com/jhfjhfj1/autokeras) - AutoML for deep learning.  
[simpledet](https://github.com/TuSimple/simpledet) - Object Detection and Instance Recognition.  
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) - Plot neural networks.  
[lucid](https://github.com/tensorflow/lucid) - Neural network interpretability, [Activation Maps](https://openai.com/blog/introducing-activation-atlases/).  
[AdaBound](https://github.com/Luolc/AdaBound) - Optimizer that trains as fast as Adam and as good as SGD, [alt](https://github.com/titu1994/keras-adabound).  
[caffe](https://github.com/BVLC/caffe) - Deep learning framework, [pretrained models](https://github.com/BVLC/caffe/wiki/Model-Zoo).    
[foolbox](https://github.com/bethgelab/foolbox) - Adversarial examples that fool neural networks.  
[hiddenlayer](https://github.com/waleedka/hiddenlayer) - Training metrics.  
[imgclsmob](https://github.com/osmr/imgclsmob) - Pretrained models.  
[netron](https://github.com/lutzroeder/netron) - Visualizer for deep learning and machine learning models.  
[torchcv](https://github.com/donnyyou/torchcv) - Deep Learning in Computer Vision.  

##### Applications and Snippets
[efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch) - Promising neural network architecture.  
[CycleGAN and Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) - Various image-to-image tasks.  
[SPADE](https://github.com/nvlabs/spade) - Semantic Image Synthesis.  
[Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737), [code](https://github.com/entron/entity-embedding-rossmann), [kaggle](https://www.kaggle.com/aquatic/entity-embedding-neural-net/code)  
[Image Super-Resolution](https://github.com/idealo/image-super-resolution) - Super-scaling using a Residual Dense Network.  
Cell Segmentation - [Talk](https://www.youtube.com/watch?v=dVFZpodqJiI), Blog Posts: [1](https://www.thomasjpfan.com/2018/07/nuclei-image-segmentation-tutorial/), [2](https://www.thomasjpfan.com/2017/08/hassle-free-unets/)  
[CenterNet](https://github.com/xingyizhou/CenterNet) - Object detection.  
[deeplearning-models](https://github.com/rasbt/deeplearning-models) - Deep learning models.  

#### GPU
[cuML](https://github.com/rapidsai/cuml) - Run traditional tabular ML tasks on GPUs.  
[thundergbm](https://github.com/Xtra-Computing/thundergbm) - GBDTs and Random Forest.  
[thundersvm](https://github.com/Xtra-Computing/thundersvm) - Support Vector Machines.  

#### Regression
Understanding SVM Regression: [slides](https://cs.adelaide.edu.au/~chhshen/teaching/ML_SVR.pdf), [forum](https://www.quora.com/How-does-support-vector-regression-work), [paper](http://alex.smola.org/papers/2003/SmoSch03b.pdf)  

[pyearth](https://github.com/scikit-learn-contrib/py-earth) - Multivariate Adaptive Regression Splines (MARS), [tutorial](https://uc-r.github.io/mars).  
[pygam](https://github.com/dswah/pyGAM) - Generalized Additive Models (GAMs), [Explanation](https://multithreaded.stitchfix.com/blog/2015/07/30/gam/).  
[GLRM](https://github.com/madeleineudell/LowRankModels.jl) - Generalized Low Rank Models.  
[tweedie](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie) - Specialized distribution for zero inflated targets, [Talk](https://www.youtube.com/watch?v=-o0lpHBq85I).  

#### Classification
[Talk](https://www.youtube.com/watch?v=DkLPYccEJ8Y), [Notebook](https://github.com/ianozsvald/data_science_delivered/blob/master/ml_creating_correct_capable_classifiers.ipynb)  
[Blog post: Probability Scoring](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)  
[All classification metrics](http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf)  
[DESlib](https://github.com/scikit-learn-contrib/DESlib) - Dynamic classifier and ensemble selection  

#### Clustering
[pyclustering](https://github.com/annoviko/pyclustering) - All sorts of clustering algorithms.  
[somoclu](https://github.com/peterwittek/somoclu) - Self-organizing map.  
[hdbscan](https://github.com/scikit-learn-contrib/hdbscan) - Clustering algorithm, [talk](https://www.youtube.com/watch?v=dGsxd67IFiU).  
[nmslib](https://github.com/nmslib/nmslib) - Similarity search library and toolkit for evaluation of k-NN methods.  
[buckshotpp](https://github.com/zjohn77/buckshotpp) - Outlier-resistant and scalable clustering algorithm.  
[merf](https://github.com/manifoldai/merf) - Mixed Effects Random Forest for Clustering, [video](https://www.youtube.com/watch?v=gWj4ZwB7f3o)  

#### Interpretable Classifiers and Regressors
[skope-rules](https://github.com/scikit-learn-contrib/skope-rules) - Interpretable classifier, IF-THEN rules.  
[sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) - Interpretable classifiers, Bayesian Rule List classifier.  

#### Multi-label classification
[scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) - Multi-label classification, [talk](https://www.youtube.com/watch?v=m-tAASQA7XQ&t=18m57s).  

#### Signal Processing and Filtering
[Stanford Lecture Series on Fourier Transformation](https://see.stanford.edu/Course/EE261), [Youtube](https://www.youtube.com/watch?v=gZNm7L96pfY&list=PLB24BC7956EE040CD&index=1), [Lecture Notes](https://see.stanford.edu/materials/lsoftaee261/book-fall-07.pdf).  
[The Scientist & Engineer's Guide to Digital Signal Processing (1999)](https://www.analog.com/en/education/education-library/scientist_engineers_guide.html).  
[Kalman Filter book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) - Focuses on intuition using Jupyter Notebooks. Includes Baysian and various Kalman filters.  
[Interactive Tool](https://fiiir.com/) for FIR and IIR filters, [Examples](https://plot.ly/python/fft-filters/).  
[filterpy](https://github.com/rlabbe/filterpy) - Kalman filtering and optimal estimation library.  

#### Time Series
[statsmodels](https://www.statsmodels.org/dev/tsa.html) - Time series analysis, [seasonal decompose](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) [example](https://gist.github.com/balzer82/5cec6ad7adc1b550e7ee), [SARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html), [granger causality](http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html).  
[pyramid](https://github.com/tgsmith61591/pyramid), [pmdarima](https://github.com/tgsmith61591/pmdarima) - Wrapper for (Auto-) ARIMA.  
[pyflux](https://github.com/RJT1990/pyflux) - Time series prediction algorithms (ARIMA, GARCH, GAS, Bayesian).  
[prophet](https://github.com/facebook/prophet) - Time series prediction library.  
[pm-prophet](https://github.com/luke14free/pm-prophet) - Time series prediction and decomposition library.  
[htsprophet](https://github.com/CollinRooney12/htsprophet) - Hierarchical Time Series Forecasting using Prophet.  
[nupic](https://github.com/numenta/nupic) - Hierarchical Temporal Memory (HTM) for Time Series Prediction and Anomaly Detection.  
[tensorflow](https://github.com/tensorflow/tensorflow/) - LSTM and others, examples: [link](
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
), [link](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/timeseries), [link](https://github.com/hzy46/TensorFlow-Time-Series-Examples), [Explain LSTM](https://github.com/slundberg/shap/blob/master/notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.ipynb), seq2seq: [1](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/), [2](https://github.com/guillaume-chevalier/seq2seq-signal-prediction), [3](https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb), [4](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction)  
[tspreprocess](https://github.com/MaxBenChrist/tspreprocess) - Preprocessing: Denoising, Compression, Resampling.  
[tsfresh](https://github.com/blue-yonder/tsfresh) - Time series feature engineering.  
[thunder](https://github.com/thunder-project/thunder) - Data structures and algorithms for loading, processing, and analyzing time series data.  
[gatspy](https://www.astroml.org/gatspy/) - General tools for Astronomical Time Series, [talk](https://www.youtube.com/watch?v=E4NMZyfao2c).  
[gendis](https://github.com/IBCNServices/GENDIS) - shapelets, [example](https://github.com/IBCNServices/GENDIS/blob/master/gendis/example.ipynb).  
[tslearn](https://github.com/rtavenar/tslearn) - Time series clustering and classification, `TimeSeriesKMeans`, `TimeSeriesKMeans`.  
[pastas](https://pastas.readthedocs.io/en/latest/examples.html) - Simulation of time series.  
[fastdtw](https://github.com/slaypni/fastdtw) - Dynamic Time Warp Distance.  
[fable](https://www.rdocumentation.org/packages/fable/versions/0.0.0.9000) - Time Series Forecasting (R package).  
[CausalImpact](https://github.com/tcassou/causal_impact) - Causal Impact Analysis ([R package](https://google.github.io/CausalImpact/CausalImpact.html)).  
[pydlm](https://github.com/wwrechard/pydlm) - Bayesian time series modeling ([R package](https://cran.r-project.org/web/packages/bsts/index.html), [Blog post](http://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html))  
[PyAF](https://github.com/antoinecarme/pyaf) - Automatic Time Series Forecasting.  
[luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library from Linkedin.  
[matrixprofile-ts](https://github.com/target/matrixprofile-ts) - Detecting patterns and anomalies, [website](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html), [ppt](https://www.cs.ucr.edu/~eamonn/Matrix_Profile_Tutorial_Part1.pdf).  
[stumpy](https://github.com/TDAmeritrade/stumpy) - Another matrix profile library.  
[obspy](https://github.com/obspy/obspy) - Seismology package. Useful `classic_sta_lta` function.  
[RobustSTL](https://github.com/LeeDoYup/RobustSTL) - Robust Seasonal-Trend Decomposition.  
[seglearn](https://github.com/dmbee/seglearn) - Time Series library.  
[pyts](https://github.com/johannfaouzi/pyts) - Time series transformation and classification, [Imaging time series](https://pyts.readthedocs.io/en/latest/auto_examples/index.html#imaging-time-series).  
Turn time series into images and use Neural Nets: [example](https://gist.github.com/oguiza/c9c373aec07b96047d1ba484f23b7b47), [example](https://github.com/kiss90/time-series-classification).  

##### Time Series Evaluation

[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) - Sklearn time series split.  
[tscv](https://github.com/WenjieZ/TSCV) - Evaluation with gap.  

#### Financial Data
[pyfolio](https://github.com/quantopian/pyfolio) - Portfolio and risk analytics.  
[zipline](https://github.com/quantopian/zipline) - Algorithmic trading.  
[alphalens](https://github.com/quantopian/alphalens) - Performance analysis of predictive stock factors.  

#### Survival Analysis
[Time-dependent Cox Model in R](https://stats.stackexchange.com/questions/101353/cox-regression-with-time-varying-covariates).  
[lifelines](https://lifelines.readthedocs.io/en/latest/) - Survival analysis, Cox PH Regression, [talk](https://www.youtube.com/watch?v=aKZQUaNHYb0), [talk2](https://www.youtube.com/watch?v=fli-yE5grtY).  
[scikit-survival](https://github.com/sebp/scikit-survival) - Survival analysis.  
[xgboost](https://github.com/dmlc/xgboost) - `"objective": "survival:cox"` [NHANES example](https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html)  
[survivalstan](https://github.com/hammerlab/survivalstan) - Survival analysis, [intro](http://www.hammerlab.org/2017/06/26/introducing-survivalstan/).  
[convoys](https://github.com/better/convoys) - Analyze time lagged conversions.  
RandomSurvivalForests (R packages: randomForestSRC, ggRandomForests).  

#### Outlier Detection & Anomaly Detection
[sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html) - Isolation Forest and others.  
[pyod](https://pyod.readthedocs.io/en/latest/pyod.html) - Outlier Detection / Anomaly Detection.  
[eif](https://github.com/sahandha/eif) - Extended Isolation Forest.  
[AnomalyDetection](https://github.com/twitter/AnomalyDetection) - Anomaly detection (R package).  
[luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library from Linkedin.  
Distances for comparing histograms and detecting outliers - [Talk](https://www.youtube.com/watch?v=U7xdiGc7IRU): [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html), [Wasserstein](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html), [Energy Distance (Cramer)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html), [Kullback-Leibler divergence](https://scipy.github.io/devdocs/generated/scipy.stats.entropy.html)  

#### Ranking
[lightning](https://github.com/scikit-learn-contrib/lightning) - Large-scale linear classification, regression and ranking.  

#### Scoring
[SLIM](https://github.com/ustunb/slim-python) - Scoring systems for classification, Supersparse linear integer models.  

#### Probabilistic Modeling and Bayes
[Intro](https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html), [Guide](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)  
[PyMC3](https://docs.pymc.io/) - Baysian modelling, [intro](https://docs.pymc.io/notebooks/getting_started)  
[pomegranate](https://github.com/jmschrei/pomegranate) - Probabilistic modelling, [talk](https://www.youtube.com/watch?v=dE5j6NW-Kzg).  
[pmlearn](https://github.com/pymc-learn/pymc-learn) - Probabilistic machine learning.  
[arviz](https://github.com/arviz-devs/arviz) - Exploratory analysis of Bayesian models.  
[zhusuan](https://github.com/thu-ml/zhusuan) - Bayesian deep learning, generative models.  
[dowhy](https://github.com/Microsoft/dowhy) - Estimate causal effects.  
[edward](https://github.com/blei-lab/edward) - Probabilistic modeling, inference, and criticism, [Mixture Density Networks (MNDs)](http://edwardlib.org/tutorials/mixture-density-network), [MDN Explanation](https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca).  
[Pyro](https://github.com/pyro-ppl/pyro) - Deep Universal Probabilistic Programming.  
[tensorflow probability](https://github.com/tensorflow/probability) - Deep learning and probabilistic modelling, [talk](https://www.youtube.com/watch?v=BrwKURU-wpk), [example](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb).  

#### Stacking Models and Ensembles
[Model Stacking Blog Post](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)  
[mlxtend](https://github.com/rasbt/mlxtend) - `EnsembleVoteClassifier`, `StackingRegressor`, `StackingCVRegressor` for model stacking.  
[vecstack](https://github.com/vecxoz/vecstack) - Stacking ML models.  
[StackNet](https://github.com/kaz-Anova/StackNet) - Stacking ML models.  
[mlens](https://github.com/flennerhag/mlens) - Ensemble learning.  

#### Model Evaluation
[pycm](https://github.com/sepandhaghighi/pycm) - Multi-class confusion matrix.  
[pandas_ml](https://github.com/pandas-ml/pandas-ml) - Confusion matrix.  
Plotting learning curve: [link](http://www.ritchieng.com/machinelearning-learning-curve/).  
[yellowbrick](http://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html) - Learning curve.  

#### Model Explanation, Interpretability, Feature Importance
[Book](https://christophm.github.io/interpretable-ml-book/agnostic.html), [Examples](https://github.com/jphall663/interpretable_machine_learning_with_python)  
[shap](https://github.com/slundberg/shap) - Explain predictions of machine learning models, [talk](https://www.youtube.com/watch?v=C80SQe16Rao).  
[treeinterpreter](https://github.com/andosa/treeinterpreter) - Interpreting scikit-learn's decision tree and random forest predictions.  
[lime](https://github.com/marcotcr/lime) - Explaining the predictions of any machine learning classifier, [talk](https://www.youtube.com/watch?v=C80SQe16Rao), [Warning (Myth 7)](https://crazyoscarchang.github.io/2019/02/16/seven-myths-in-machine-learning-research/).  
[lime_xgboost](https://github.com/jphall663/lime_xgboost) - Create LIMEs for XGBoost.  
[eli5](https://github.com/TeamHG-Memex/eli5) - Inspecting machine learning classifiers and explaining their predictions.  
[lofo-importance](https://github.com/aerdem4/lofo-importance) - Leave One Feature Out Importance, [talk](https://www.youtube.com/watch?v=zqsQ2ojj7sE), examples: [1](https://www.kaggle.com/divrikwicky/pf-f-lofo-importance-on-adversarial-validation), [2](https://www.kaggle.com/divrikwicky/lofo-importance), [3](https://www.kaggle.com/divrikwicky/santanderctp-lofo-feature-importance).  
[pybreakdown](https://github.com/MI2DataLab/pyBreakDown) - Generate feature contribution plots.  
[FairML](https://github.com/adebayoj/fairml) - Model explanation, feature importance.  
[pycebox](https://github.com/AustinRochford/PyCEbox) - Individual Conditional Expectation Plot Toolbox.  
[pdpbox](https://github.com/SauceCat/PDPbox) - Partial dependence plot toolbox, [example](https://www.kaggle.com/dansbecker/partial-plots).  
[partial_dependence](https://github.com/nyuvis/partial_dependence) - Visualize and cluster partial dependence.  
[skater](https://github.com/datascienceinc/Skater) - Unified framework to enable model interpretation.  
[anchor](https://github.com/marcotcr/anchor) - High-Precision Model-Agnostic Explanations for classifiers.  
[l2x](https://github.com/Jianbo-Lab/L2X) - Instancewise feature selection as methodology for model interpretation.  
[contrastive_explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) - Contrastive explanations.  
[DrWhy](https://github.com/ModelOriented/DrWhy) - Collection of tools for explainable AI.  
[lucid](https://github.com/tensorflow/lucid) - Neural network interpretability.  
[xai](https://github.com/EthicalML/XAI) - An eXplainability toolbox for machine learning.  
[innvestigate](https://github.com/albermax/innvestigate) - A toolbox to investigate neural network predictions.  
[dalex](https://github.com/pbiecek/DALEX) - Explanations for ML models (R package).  
[interpret](https://github.com/microsoft/interpret) - Fit interpretable models, explain models (Microsoft).  

#### Automated Machine Learning
[AdaNet](https://github.com/tensorflow/adanet) - Automated machine learning based on tensorflow.  
[tpot](https://github.com/EpistasisLab/tpot) - Automated machine learning tool, optimizes machine learning pipelines.  
[auto_ml](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning for analytics & production.  
[autokeras](https://github.com/jhfjhfj1/autokeras) - AutoML for deep learning.  
[nni](https://github.com/Microsoft/nni) - Toolkit for neural architecture search and hyper-parameter tuning by Microsoft.  
[automl-gs](https://github.com/minimaxir/automl-gs) - Automated machine learning.  
[mljar](https://github.com/mljar/mljar-supervised) - Automated machine learning.  

#### Evolutionary Algorithms & Optimization
[deap](https://github.com/DEAP/deap) - Evolutionary computation framework (Genetic Algorithm, Evolution strategies).  
[evol](https://github.com/godatadriven/evol) - DSL for composable evolutionary algorithms, [talk](https://www.youtube.com/watch?v=68ABAU_V8qI&t=11m49s).  
[platypus](https://github.com/Project-Platypus/Platypus) - Multiobjective optimization.  
[autograd](https://github.com/HIPS/autograd) - Efficiently computes derivatives of numpy code.  
[nevergrad](https://github.com/facebookresearch/nevergrad) - Derivation-free optimization.  
[gplearn](https://gplearn.readthedocs.io/en/stable/) - Sklearn-like interface for genetic programming.  
[blackbox](https://github.com/paulknysh/blackbox) - Optimization of expensive black-box functions.  
Optometrist algorithm - [paper](https://www.nature.com/articles/s41598-017-06645-7).  
[DeepSwarm](https://github.com/Pattio/DeepSwarm) - Neural architecture search.  

#### Hyperparameter Tuning
[sklearn](https://scikit-learn.org/stable/index.html) - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).  
[sklearn-deap](https://github.com/rsteca/sklearn-deap) - Hyperparameter search using genetic algorithms.  
[hyperopt](https://github.com/hyperopt/hyperopt) - Hyperparameter optimization.  
[hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) - Hyperopt + sklearn.  
[optuna](https://github.com/pfnet/optuna) - Hyperparamter optimization, [Talk](https://www.youtube.com/watch?v=tcrcLRopTX0).  
[skopt](https://scikit-optimize.github.io/) - `BayesSearchCV` for Hyperparameter search.  
[tune](https://ray.readthedocs.io/en/latest/tune.html) - Hyperparameter search with a focus on deep learning and deep reinforcement learning.  
[hypergraph](https://github.com/aljabr0/hypergraph) - Global optimization methods and hyperparameter optimization.  
[bbopt](https://github.com/evhub/bbopt) - Black box hyperparameter optimization.  
[dragonfly](https://github.com/dragonfly/dragonfly) - Scalable Bayesian optimisation.  

#### Incremental Learning, Online Learning
sklearn - [PassiveAggressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html), [PassiveAggressiveRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html).  
[creme-ml](https://github.com/creme-ml/creme) - Incremental learning framework, [talk](https://www.youtube.com/watch?v=P3M6dt7bY9U).  
[Kaggler](https://github.com/jeongyoonlee/Kaggler) - Online Learning algorithms.  

#### Active Learning
[Talk](https://www.youtube.com/watch?v=0efyjq5rWS4)  
[modAL](https://github.com/modAL-python/modAL) - Active learning framework.  

#### Reinforcement Learning
[YouTube](https://www.youtube.com/playlist?list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT), [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)  
Intro to Monte Carlo Tree Search (MCTS) - [1](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/), [2](http://mcts.ai/about/index.html), [3](https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238)  
AlphaZero methodology - [1](https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning), [2](https://web.stanford.edu/~surag/posts/alphazero.html), [3](https://github.com/suragnair/alpha-zero-general), [Cheat Sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)  
[RLLib](https://ray.readthedocs.io/en/latest/rllib.html) - Library for reinforcement learning.  
[Horizon](https://github.com/facebookresearch/Horizon/) - Facebook RL framework.  

#### Frameworks
[h2o](https://github.com/h2oai/h2o-3) - Scalable machine learning.  
[turicreate](https://github.com/apple/turicreate) - Apple Machine Learning Toolkit.  
[astroml](https://github.com/astroML/astroML) - ML for astronomical data.  

#### Deployment and Lifecycle Management

##### General
[pyup](https://github.com/pyupio/pyup) - Python dependency management.  
[pypi-timemachine](https://github.com/astrofrog/pypi-timemachine) - Install packages with pip as if you were in the past.  
[pypi2nix] - Fix package versions and create reproducible environments, [Talk](https://www.youtube.com/watch?v=USDbjmxEZ_I).

##### Data Science Related
[m2cgen](https://github.com/BayesWitnesses/m2cgen) - Transpile trained ML models into other languages.  
[sklearn-porter](https://github.com/nok/sklearn-porter) - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.  
[mlflow](https://mlflow.org/) - Manage the machine learning lifecycle, including experimentation, reproducibility and deployment.  
[modelchimp](https://github.com/ModelChimp/modelchimp) - Experiment Tracking.  
[skll](https://github.com/EducationalTestingService/skll) - Command-line utilities to make it easier to run machine learning experiments.  
[BentoML](https://github.com/bentoml/BentoML) - Package and deploy machine learning models for serving in production
[dvc](https://github.com/iterative/dvc) - Versioning for ML projects.  

#### Math and Background
Gilbert Strang - [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/index.htm)  
Gilbert Strang - [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning
](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)  

#### Other
[daft](https://github.com/dfm/daft) - Render probabilistic graphical models using matplotlib.  
[unyt](https://github.com/yt-project/unyt) - Working with units.  
[scrapy](https://github.com/scrapy/scrapy) - Web scraping library.  
[VowpalWabbit](https://github.com/VowpalWabbit/vowpal_wabbit) - ML Toolkit from Microsoft.  
[metric-learn](https://github.com/metric-learn/metric-learn) - Metric learning.   

#### General Python Programming
[funcy](https://github.com/Suor/funcy) - Fancy and practical functional tools.  
[more_itertools](https://more-itertools.readthedocs.io/en/latest/) - Extension of itertools.  
[dill](https://pypi.org/project/dill/) - Serialization, alternative to pickle.  
[attrs](https://github.com/python-attrs/attrs) - Python classes without boilerplate.  
[dateparser](https://dateparser.readthedocs.io/en/latest/) - A better date parser.  
[jellyfish](https://github.com/jamesturk/jellyfish) - Approximate string matching.   

#### Blogs
[Distill.pub](https://distill.pub/) - Blog.

#### Awesome Lists and Resources
[Data Science Notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)  
[Awesome Adversarial Machine Learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning)  
[Awesome AI Booksmarks](https://github.com/goodrahstar/my-awesome-AI-bookmarks)  
[Awesome AI on Kubernetes](https://github.com/CognonicLabs/awesome-AI-kubernetes)  
[Awesome Big Data](https://github.com/onurakpolat/awesome-bigdata)  
[Awesome Business Machine Learning](https://github.com/firmai/business-machine-learning)  
[Awesome CSV](https://github.com/secretGeek/AwesomeCSV)  
[Awesome Data Science with Ruby](https://github.com/arbox/data-science-with-ruby)  
[Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)  
[Awesome ETL](https://github.com/pawl/awesome-etl)  
[Awesome Financial Machine Learning](https://github.com/firmai/financial-machine-learning)  
[Awesome GAN Applications](https://github.com/nashory/gans-awesome-applications)  
[Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning#python)  
[Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)   
[Awesome Machine Learning Operations](https://github.com/EthicalML/awesome-machine-learning-operations)  
[Awesome Network Embedding](https://github.com/chihming/awesome-network-embedding)  
[Awesome Python](https://github.com/vinta/awesome-python)   
[Awesome Python Data Science](https://github.com/krzjoa/awesome-python-datascience)   
[Awesome Python Data Science](https://github.com/thomasjpfan/awesome-python-data-science)  
[Awesome Pytorch](https://github.com/bharathgs/Awesome-pytorch-list)  
[Awesome Recommender Systems](https://github.com/grahamjenson/list_of_recommender_systems)  
[Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)  
[Awesome Sentence Embedding](https://github.com/Separius/awesome-sentence-embedding)  
[Awesome Time Series](https://github.com/MaxBenChrist/awesome_time_series_in_python)  
[Awesome Time Series Anomaly Detection](https://github.com/rob-med/awesome-TS-anomaly-detection)  
[Recommender Systems (Microsoft)](https://github.com/Microsoft/Recommenders)  
[The GAN Zoo](https://deephunt.in/the-gan-zoo-79597dc8c347) - List of Generative Adversarial Networks  
[Datascience Cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets)  

#### Things I google a lot
[Frequency codes for time series](https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)  
[Date parsing codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)  
[Feature Calculators tsfresh](https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py)  

## Contributing  
Do you know a package that should be on this list? Did you spot a package that is no longer maintained and should be removed from this list? Then feel free to read the [contribution guidelines](CONTRIBUTING.md) and submit your pull request or create a new issue.  

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
