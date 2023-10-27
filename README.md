# Awesome Data Science with Python

> A curated list of awesome resources for practicing data science using Python, including not only libraries, but also links to tutorials, code snippets, blog posts and talks.  

#### Core
[pandas](https://pandas.pydata.org/) - Data structures built on top of [numpy](https://www.numpy.org/).  
[scikit-learn](https://scikit-learn.org/stable/) - Core ML library, [intelex](https://github.com/intel/scikit-learn-intelex).  
[matplotlib](https://matplotlib.org/) - Plotting library.  
[seaborn](https://seaborn.pydata.org/) - Data visualization library based on matplotlib.  
[ydata-profiling](https://github.com/ydataai/ydata-profiling) - Descriptive statistics using `ProfileReport`.  
[sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) - Helpful `DataFrameMapper` class.  
[missingno](https://github.com/ResidentMario/missingno) - Missing data visualization.  
[rainbow-csv](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) - VSCode plugin to display .csv files with nice colors.  

#### General Python Programming
[Python Best Practices Guide](https://github.com/qiwihui/pocket_readings/issues/1148#issuecomment-874448132)  
[pyenv](https://github.com/pyenv/pyenv) - Manage multiple Python versions on your system.  
[poetry](https://github.com/python-poetry/poetry) - Dependency management.  
[pyscaffold](https://github.com/pyscaffold/pyscaffold) - Python project template generator.  
[hydra](https://github.com/facebookresearch/hydra) - Configuration management.  
[hatch](https://github.com/pypa/hatch) - Python project management.  
[more_itertools](https://more-itertools.readthedocs.io/en/latest/) - Extension of itertools.  
[tqdm](https://github.com/tqdm/tqdm) - Progress bars for for-loops. Also supports [pandas apply()](https://stackoverflow.com/a/34365537/1820480).  
[loguru](https://github.com/Delgan/loguru) - Python logging.  


#### Pandas Tricks, Alternatives and Additions
[pandasvault](https://github.com/firmai/pandasvault) - Large collection of pandas tricks.  
[polars](https://github.com/pola-rs/polars) - Multi-threaded alternative to pandas.  
[xarray](https://github.com/pydata/xarray/) - Extends pandas to n-dimensional arrays.  
[pandas_flavor](https://github.com/Zsailer/pandas_flavor) - Write custom accessors like `.str` and `.dt`.   
[duckdb](https://github.com/duckdb/duckdb) - Efficiently run SQL queries on pandas DataFrame.  

#### Pandas Parallelization
[modin](https://github.com/modin-project/modin) - Parallelization library for faster pandas `DataFrame`.  
[vaex](https://github.com/vaexio/vaex) - Out-of-Core DataFrames.  
[pandarallel](https://github.com/nalepae/pandarallel) - Parallelize pandas operations.  
[swifter](https://github.com/jmcarpenter2/swifter) - Apply any function to a pandas DataFrame faster.   

#### Environment and Jupyter
[Jupyter Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)  
[ipyflow](https://github.com/ipyflow/ipyflow) - IPython kernel for Jupyter with additional features.  
[nteract](https://nteract.io/) - Open Jupyter Notebooks with doubleclick.  
[papermill](https://github.com/nteract/papermill) - Parameterize and execute Jupyter notebooks, [tutorial](https://pbpython.com/papermil-rclone-report-1.html).  
[nbdime](https://github.com/jupyter/nbdime) - Diff two notebook files, Alternative GitHub App: [ReviewNB](https://www.reviewnb.com/).  
[RISE](https://github.com/damianavila/RISE) - Turn Jupyter notebooks into presentations.  
[qgrid](https://github.com/quantopian/qgrid) - Pandas `DataFrame` sorting.  
[lux](https://github.com/lux-org/lux) - DataFrame visualization within Jupyter.  
[pandasgui](https://github.com/adamerose/pandasgui) - GUI for viewing, plotting and analyzing Pandas DataFrames.  
[dtale](https://github.com/man-group/dtale) - View and analyze Pandas data structures, integrating with Jupyter.  
[itables](https://github.com/mwouts/itables) - Interactive tables in Jupyter.  
[handcalcs](https://github.com/connorferster/handcalcs) - More convenient way of writing mathematical equations in Jupyter.  
[notebooker](https://github.com/man-group/notebooker) - Productionize and schedule Jupyter Notebooks.  
[bamboolib](https://github.com/tkrabel/bamboolib) - Intuitive GUI for tables.  
[voila](https://github.com/QuantStack/voila) - Turn Jupyter notebooks into standalone web applications.  
[voila-gridstack](https://github.com/voila-dashboards/voila-gridstack) - Voila grid layout.  

#### Extraction
[textract](https://github.com/deanmalmgren/textract) - Extract text from any document.  

#### Big Data
[spark](https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html#work-with-dataframes) - `DataFrame` for big data, [cheatsheet](https://gist.github.com/crawles/b47e23da8218af0b9bd9d47f5242d189), [tutorial](https://github.com/ericxiao251/spark-syntax).  
[dask](https://github.com/dask/dask), [dask-ml](http://ml.dask.org/) - Pandas `DataFrame` for big data and machine learning library, [resources](https://matthewrocklin.com/blog//work/2018/07/17/dask-dev), [talk1](https://www.youtube.com/watch?v=ccfsbuqsjgI), [talk2](https://www.youtube.com/watch?v=RA_2qdipVng), [notebooks](https://github.com/dask/dask-ec2/tree/master/notebooks), [videos](https://www.youtube.com/user/mdrocklin).  
[h2o](https://github.com/h2oai/h2o-3) - Helpful `H2OFrame` class for out-of-memory dataframes.  
[datatable](https://github.com/h2oai/datatable) - Data Table for big data support.  
[cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library, [Intro](https://www.youtube.com/watch?v=6XzS5XcpicM&t=2m50s).  
[cupy](https://github.com/cupy/cupy) - NumPy-like API accelerated with CUDA.  
[ray](https://github.com/ray-project/ray/) - Flexible, high-performance distributed execution framework.  
[bottleneck](https://github.com/kwgoodman/bottleneck) - Fast NumPy array functions written in C.   
[petastorm](https://github.com/uber/petastorm) - Data access library for parquet files by Uber.  
[zarr](https://github.com/zarr-developers/zarr-python) - Distributed NumPy arrays.  
[NVTabular](https://github.com/NVIDIA/NVTabular) - Feature engineering and preprocessing library for tabular data by Nvidia.  
[tensorstore](https://github.com/google/tensorstore) - Reading and writing large multi-dimensional arrays (Google).  

#### Command line tools, CSV
[csvkit](https://github.com/wireservice/csvkit) - Command line tool for CSV files.  
[csvsort](https://pypi.org/project/csvsort/) - Sort large csv files.  

#### Classical Statistics

##### Correlation
[phik](https://github.com/kaveio/phik) - Correlation between categorical, ordinal and interval variables.  

##### Packages
[statsmodels](https://www.statsmodels.org/stable/index.html) - Statistical tests.  
[linearmodels](https://github.com/bashtage/linearmodels) - Instrumental variable and panel data models.  
[pingouin](https://github.com/raphaelvallat/pingouin) - Statistical tests. [Pairwise correlation between columns of pandas DataFrame](https://pingouin-stats.org/generated/pingouin.pairwise_corr.html)   
[scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) - Statistical tests.  
[scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) - Statistical post-hoc tests for pairwise multiple comparisons.   
Bland-Altman Plot [1](https://pingouin-stats.org/generated/pingouin.plot_blandaltman.html), [2](http://www.statsmodels.org/dev/generated/statsmodels.graphics.agreement.mean_diff_plot.html) - Plot for agreement between two methods of measurement.  
[ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)  

##### Statistical Tests
[test_proportions_2indep](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.test_proportions_2indep.html) - Proportion test.  
[G-Test](https://en.wikipedia.org/wiki/G-test) - Alternative to chi-square test, [power_divergence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html).  

##### Comparing Two Populations
[torch-two-sample](https://github.com/josipd/torch-two-sample) - Friedman-Rafsky Test: Compare two population based on a multivariate generalization of the Runstest. [Explanation](https://www.real-statistics.com/multivariate-statistics/multivariate-normal-distribution/friedman-rafsky-test/), [Application](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5014134/)  

##### Interim Analyses / Sequential Analysis / Stopping
[Sequential Analysis](https://en.wikipedia.org/wiki/Sequential_analysis) - Wikipedia.  
[sequential](https://cran.r-project.org/web/packages/Sequential/Sequential.pdf) - Exact Sequential Analysis for Poisson and Binomial Data (R package).  
[confseq](https://github.com/gostevehoward/confseq) - Uniform boundaries, confidence sequences, and always-valid p-values.  

##### Visualizations
[Great Overview over Visualizations](https://textvis.lnu.se/)  
[Dependent Propabilities](https://static.laszlokorte.de/stochastic/)  
[Null Hypothesis Significance Testing (NHST) and Sample Size Calculation](https://rpsychologist.com/d3/NHST/)  
[Correlation](https://rpsychologist.com/d3/correlation/)  
[Cohen's d](https://rpsychologist.com/d3/cohend/)  
[Confidence Interval](https://rpsychologist.com/d3/CI/)  
[Equivalence, non-inferiority and superiority testing](https://rpsychologist.com/d3/equivalence/)  
[Bayesian two-sample t test](https://rpsychologist.com/d3/bayes/)  
[Distribution of p-values when comparing two groups](https://rpsychologist.com/d3/pdist/)  
[Understanding the t-distribution and its normal approximation](https://rpsychologist.com/d3/tdist/)  

##### Talks
[Inverse Propensity Weighting](https://www.youtube.com/watch?v=SUq0shKLPPs)  
[Dealing with Selection Bias By Propensity Based Feature Selection](https://www.youtube.com/watch?reload=9&v=3ZWCKr0vDtc)  

##### Texts
[Modes, Medians and Means: A Unifying Perspective](https://www.johnmyleswhite.com/notebook/2013/03/22/modes-medians-and-means-an-unifying-perspective/)   
[Using Norms to Understand Linear Regression](https://www.johnmyleswhite.com/notebook/2013/03/22/using-norms-to-understand-linear-regression/)   
[Verifying the Assumptions of Linear Models](https://github.com/erykml/medium_articles/blob/master/Statistics/linear_regression_assumptions.ipynb)  
[Mediation and Moderation Intro](https://ademos.people.uic.edu/Chapter14.html)  
[Montgomery et al. - How conditioning on post-treatment variables can ruin your experiment and what to do about it](https://cpb-us-e1.wpmucdn.com/sites.dartmouth.edu/dist/5/2293/files/2021/03/post-treatment-bias.pdf)  
[Greenland - Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4877414/)  
[Blume - Second-generation p-values: Improved rigor, reproducibility, & transparency in statistical analyses](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188299)  
[Lindeløv - Common statistical tests are linear models](https://lindeloev.github.io/tests-as-linear/)    
[Chatruc - The Central Limit Theorem and its misuse](https://web.archive.org/web/20191229234155/https://lambdaclass.com/data_etudes/central_limit_theorem_misuse/)  
[Al-Saleh - Properties of the Standard Deviation that are Rarely Mentioned in Classrooms](http://www.stat.tugraz.at/AJS/ausg093/093Al-Saleh.pdf)   
[Wainer - The Most Dangerous Equation](http://nsmn1.uh.edu/dgraur/niv/themostdangerousequation.pdf)  
[Gigerenzer - The Bias Bias in Behavioral Economics](https://www.nowpublishers.com/article/Details/RBE-0092)  
[Cook - Estimating the chances of something that hasn’t happened yet](https://www.johndcook.com/blog/2010/03/30/statistical-rule-of-three/)  
[Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing](https://www.researchgate.net/publication/316652618_Same_Stats_Different_Graphs_Generating_Datasets_with_Varied_Appearance_and_Identical_Statistics_through_Simulated_Annealing), [Youtube](https://www.youtube.com/watch?v=DbJyPELmhJc)  
[How large is that number in the Law of Large Numbers?](https://thepalindrome.org/p/how-large-that-number-in-the-law)  
[The Prosecutor's Fallacy](https://www.cebm.ox.ac.uk/news/views/the-prosecutors-fallacy)  

#### Epidemiology
[R Epidemics Consortium](https://www.repidemicsconsortium.org/projects/) - Large tool suite for working with epidemiological data (R packages). [Github](https://github.com/reconhub)   
[incidence2](https://github.com/reconhub/incidence2) - Computation, handling, visualisation and simple modelling of incidence (R package).  
[EpiEstim](https://github.com/mrc-ide/EpiEstim) - Estimate time varying instantaneous reproduction number R during epidemics (R package) [paper](https://academic.oup.com/aje/article/178/9/1505/89262).  
[researchpy](https://github.com/researchpy/researchpy) - Helpful `summary_cont()` function for summary statistics (Table 1).  
[zEpid](https://github.com/pzivich/zEpid) - Epidemiology analysis package, [Tutorial](https://github.com/pzivich/Python-for-Epidemiologists).  
[tipr](https://github.com/LucyMcGowan/tipr) - Sensitivity analyses for unmeasured confounders (R package).  
[quartets](https://github.com/r-causal/quartets) - Anscombe’s Quartet, Causal Quartet, [Datasaurus Dozen](https://github.com/jumpingrivers/datasauRus) and others (R package).    

#### Exploration and Cleaning
[Checklist](https://github.com/r0f1/ml_checklist).  
[pyjanitor](https://github.com/pyjanitor-devs/pyjanitor) - Clean messy column names.  
[pandera](https://github.com/unionai-oss/pandera) - Data / Schema validation.  
[impyute](https://github.com/eltonlaw/impyute) - Imputations.  
[fancyimpute](https://github.com/iskandr/fancyimpute) - Matrix completion and imputation algorithms.  
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - Resampling for imbalanced datasets.  
[tspreprocess](https://github.com/MaxBenChrist/tspreprocess) - Time series preprocessing: Denoising, Compression, Resampling.  
[Kaggler](https://github.com/jeongyoonlee/Kaggler) - Utility functions (`OneHotEncoder(min_obs=100)`)  

#### Noisy Labels
[cleanlab](https://github.com/cleanlab/cleanlab) - Machine learning with noisy labels, finding mislabelled data, and uncertainty quantification. Also see awesome list below.  
[doubtlab](https://github.com/koaning/doubtlab) - Find bad or noisy labels.

#### Train / Test Split
[iterative-stratification](https://github.com/trent-b/iterative-stratification) - Stratification of multilabel data.  

#### Feature Engineering
[Vincent Warmerdam: Untitled12.ipynb](https://www.youtube.com/watch?v=yXGCKqo5cEY) - Using df.pipe()  
[Vincent Warmerdam: Winning with Simple, even Linear, Models](https://www.youtube.com/watch?v=68ABAU_V8qI)  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) - Pipeline, [examples](https://github.com/jem1031/pandas-pipelines-custom-transformers).  
[pdpipe](https://github.com/shaypal5/pdpipe) - Pipelines for DataFrames.  
[scikit-lego](https://github.com/koaning/scikit-lego) - Custom transformers for pipelines.  
[categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding) - Categorical encoding of variables, [vtreat (R package)](https://cran.r-project.org/web/packages/vtreat/vignettes/vtreat.html).  
[dirty_cat](https://github.com/dirty-cat/dirty_cat) - Encoding dirty categorical variables.  
[patsy](https://github.com/pydata/patsy/) - R-like syntax for statistical models.  
[mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/) - LDA.  
[featuretools](https://github.com/Featuretools/featuretools) - Automated feature engineering, [example](https://github.com/WillKoehrsen/automated-feature-engineering/blob/master/walk_through/Automated_Feature_Engineering.ipynb).  
[tsfresh](https://github.com/blue-yonder/tsfresh) - Time series feature engineering.  
[temporian](https://github.com/google/temporian) - Time series feature engineering by Google.  
[pypeln](https://github.com/cgarciae/pypeln) - Concurrent data pipelines.  
[feature-engine](https://github.com/feature-engine/feature_engine) - Encoders, transformers, etc.  

#### Computer Vision
[Intro to Computer Vision](https://www.youtube.com/playlist?list=PLjMXczUzEYcHvw5YYSU92WrY8IwhTuq7p)  

#### Feature Selection
[Overview Paper](https://www.sciencedirect.com/science/article/pii/S016794731930194X), [Talk](https://www.youtube.com/watch?v=JsArBz46_3s), [Repo](https://github.com/Yimeng-Zhang/feature-engineering-and-feature-selection)    
Blog post series - [1](http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/), [2](http://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/), [3](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/), [4](http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/)  
Tutorials - [1](https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn), [2](https://machinelearningmastery.com/feature-selection-machine-learning-python/)  
[sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection) - Feature selection.  
[eli5](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html#feature-selection) - Feature selection using permutation importance.  
[scikit-feature](https://github.com/jundongl/scikit-feature) - Feature selection algorithms.  
[stability-selection](https://github.com/scikit-learn-contrib/stability-selection) - Stability selection.  
[scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) - Relief-based feature selection algorithms.  
[scikit-genetic](https://github.com/manuel-calzolari/sklearn-genetic) - Genetic feature selection.  
[boruta_py](https://github.com/scikit-learn-contrib/boruta_py) - Feature selection, [explaination](https://stats.stackexchange.com/questions/264360/boruta-all-relevant-feature-selection-vs-random-forest-variables-of-importanc/264467), [example](https://www.kaggle.com/tilii7/boruta-feature-elimination).  
[Boruta-Shap](https://github.com/Ekeany/Boruta-Shap) - Boruta feature selection algorithm + shapley values.  
[linselect](https://github.com/efavdb/linselect) - Feature selection package.  
[mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/) - Exhaustive feature selection.     
[BoostARoota](https://github.com/chasedehan/BoostARoota) - Xgboost feature selection algorithm.  
[INVASE](https://github.com/jsyoon0823/INVASE) - Instance-wise Variable Selection using Neural Networks.  
[SubTab](https://github.com/AstraZeneca/SubTab) - Subsetting Features of Tabular Data for Self-Supervised Representation Learning, AstraZeneca.  
[mrmr](https://github.com/smazzanti/mrmr) - Maximum Relevance and Minimum Redundancy Feature Selection, [Website](http://home.penglab.com/proj/mRMR/).  
[arfs](https://github.com/ThomasBury/arfs) - All Relevant Feature Selection.  
[VSURF](https://github.com/robingenuer/VSURF) - Variable Selection Using Random Forests (R package) [doc](https://www.rdocumentation.org/packages/VSURF/versions/1.1.0/topics/VSURF).  
[FeatureSelectionGA](https://github.com/kaushalshetty/FeatureSelectionGA) - Feature Selection using Genetic Algorithm.  

#### Subset Selection
[apricot](https://github.com/jmschrei/apricot) - Selecting subsets of data sets to train machine learning models quickly.  
[ducks](https://github.com/manimino/ducks) - Index data for fast lookup by any combination of fields.  

#### Dimensionality Reduction / Representation Learning

##### Selection
Check also the Clustering section and self-supervised learning section for ideas!  
[Review](https://members.loria.fr/moberger/Enseignement/AVR/Exposes/TR_Dimensiereductie.pdf)  
  
PCA - [link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)    
Autoencoder - [link](https://blog.keras.io/building-autoencoders-in-keras.html)  
Isomaps - [link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap)    
LLE - [link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html)  
Force-directed graph drawing - [link](https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.draw_graph.html#scanpy.tl.draw_graph)    
MDS - [link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)  
Diffusion Maps - [link](https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.diffmap.html)  
t-SNE - [link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)    
NeRV - [link](https://github.com/ziyuang/pynerv), [paper](https://www.jmlr.org/papers/volume11/venna10a/venna10a.pdf)  
MDR - [link](https://github.com/EpistasisLab/scikit-mdr)  
UMAP - [link](https://github.com/lmcinnes/umap)  
Random Projection - [link](https://scikit-learn.org/stable/modules/random_projection.html)  
Ivis - [link](https://github.com/beringresearch/ivis)   
SimCLR - [link](https://github.com/lightly-ai/lightly)  

##### Neural-network based
[esvit](https://github.com/microsoft/esvit) - Vision Transformers for Representation Learning (Microsoft).  
[MCML](https://github.com/pachterlab/MCML) - Semi-supervised dimensionality reduction of Multi-Class, Multi-Label data (sequencing data) [paper](https://www.biorxiv.org/content/10.1101/2021.08.25.457696v1).  

##### Packages
[Dangers of PCA (paper)](https://www.nature.com/articles/s41598-022-14395-4).  
[Talk](https://www.youtube.com/watch?v=9iol3Lk6kyU), [tsne intro](https://distill.pub/2016/misread-tsne/). 
[sklearn.manifold](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold) and [sklearn.decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) - PCA, t-SNE, MDS, Isomaps and others.  
Additional plots for PCA - Factor Loadings, Cumulative Variance Explained, [Correlation Circle Plot](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_pca_correlation_graph/), [Tweet](https://twitter.com/rasbt/status/1555999903398219777/photo/1)  
[sklearn.random_projection](https://scikit-learn.org/stable/modules/random_projection.html) - Johnson-Lindenstrauss lemma, Gaussian random projection, Sparse random projection.  
[sklearn.cross_decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition) - Partial least squares, supervised estimators for dimensionality reduction and regression.  
[prince](https://github.com/MaxHalford/prince) - Dimensionality reduction, factor analysis (PCA, MCA, CA, FAMD).  
Faster t-SNE implementations: [lvdmaaten](https://lvdmaaten.github.io/tsne/), [MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE), [FIt-SNE](https://github.com/KlugerLab/FIt-SNE)
[umap](https://github.com/lmcinnes/umap) - Uniform Manifold Approximation and Projection, [talk](https://www.youtube.com/watch?v=nq6iPZVUxZU), [explorer](https://github.com/GrantCuster/umap-explorer), [explanation](https://pair-code.github.io/understanding-umap/), [parallel version](https://docs.rapids.ai/api/cuml/stable/api.html).  
[humap](https://github.com/wilsonjr/humap) - Hierarchical UMAP.  
[sleepwalk](https://github.com/anders-biostat/sleepwalk/) - Explore embeddings, interactive visualization (R package).  
[somoclu](https://github.com/peterwittek/somoclu) - Self-organizing map.  
[scikit-tda](https://github.com/scikit-tda/scikit-tda) - Topological Data Analysis, [paper](https://www.nature.com/articles/srep01236), [talk](https://www.youtube.com/watch?v=F2t_ytTLrQ4), [talk](https://www.youtube.com/watch?v=AWoeBzJd7uQ), [paper](https://www.uncg.edu/mat/faculty/cdsmyth/topological-approaches-skin.pdf).  
[giotto-tda](https://github.com/giotto-ai/giotto-tda) - Topological Data Analysis.  
[ivis](https://github.com/beringresearch/ivis) - Dimensionality reduction using Siamese Networks.  
[trimap](https://github.com/eamid/trimap) - Dimensionality reduction using triplets.  
[scanpy](https://github.com/theislab/scanpy) - [Force-directed graph drawing](https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.draw_graph.html#scanpy.tl.draw_graph), [Diffusion Maps](https://scanpy.readthedocs.io/en/stable/api/scanpy.tl.diffmap.html).  
[direpack](https://github.com/SvenSerneels/direpack) - Projection pursuit, Sufficient dimension reduction, Robust M-estimators.  
[DBS](https://cran.r-project.org/web/packages/DatabionicSwarm/vignettes/DatabionicSwarm.html) - DatabionicSwarm (R package).  
[contrastive](https://github.com/abidlabs/contrastive) - Contrastive PCA.  
[scPCA](https://github.com/PhilBoileau/scPCA) - Sparse contrastive PCA (R package).  
[tmap](https://github.com/reymond-group/tmap) - Visualization library for large, high-dimensional data sets.  
[lollipop](https://github.com/neurodata/lollipop) - Linear Optimal Low Rank Projection.  
[linearsdr](https://github.com/HarrisQ/linearsdr) - Linear Sufficient Dimension Reduction (R package).  
[PHATE](https://github.com/KrishnaswamyLab/PHATE) - Tool for visualizing high dimensional data.  

#### Visualization
[All charts](https://datavizproject.com/), [Austrian monuments](https://github.com/njanakiev/austrian-monuments-visualization).  
[Better heatmaps and correlation plots](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec).  
[Example notebooks for interactive visualizations](https://github.com/nicolaskruchten/pydata_global_2021/tree/main)(Plotly,Seaborn, Holoviz, Altair)  
[cufflinks](https://github.com/santosjorge/cufflinks) - Dynamic visualization library, wrapper for [plotly](https://plot.ly/), [medium](https://towardsdatascience.com/the-next-level-of-data-visualization-in-python-dd6e99039d5e), [example](https://github.com/WillKoehrsen/Data-Analysis/blob/master/plotly/Plotly%20Whirlwind%20Introduction.ipynb).  
[physt](https://github.com/janpipek/physt) - Better histograms, [talk](https://www.youtube.com/watch?v=ZG-wH3-Up9Y), [notebook](https://nbviewer.jupyter.org/github/janpipek/pydata2018-berlin/blob/master/notebooks/talk.ipynb).  
[fast-histogram](https://github.com/astrofrog/fast-histogram) - Fast histograms.  
[matplotlib_venn](https://github.com/konstantint/matplotlib-venn) - Venn diagrams, [alternative](https://github.com/penrose/penrose).  
[joypy](https://github.com/sbebo/joypy) - Draw stacked density plots (=ridge plots), [Ridge plots in seaborn](https://seaborn.pydata.org/examples/kde_ridgeplot.html).  
[mosaic plots](https://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html) - Categorical variable visualization, [example](https://sukhbinder.wordpress.com/2018/09/18/mosaic-plot-in-python/).  
[scikit-plot](https://github.com/reiinakano/scikit-plot) - ROC curves and other visualizations for ML models.  
[yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visualizations for ML models (similar to scikit-plot).  
[bokeh](https://bokeh.pydata.org/en/latest/) - Interactive visualization library, [Examples](https://bokeh.pydata.org/en/latest/docs/user_guide/server.html), [Examples](https://github.com/WillKoehrsen/Bokeh-Python-Visualization).  
[lets-plot](https://github.com/JetBrains/lets-plot) - Plotting library.  
[animatplot](https://github.com/t-makaro/animatplot) - Animate plots build on matplotlib.  
[plotnine](https://github.com/has2k1/plotnine) - ggplot for Python.  
[altair](https://altair-viz.github.io/) - Declarative statistical visualization library.  
[bqplot](https://github.com/bloomberg/bqplot) - Plotting library for IPython/Jupyter Notebooks.  
[hvplot](https://github.com/pyviz/hvplot) - High-level plotting library built on top of [holoviews](http://holoviews.org/).  
[dtreeviz](https://github.com/parrt/dtreeviz) - Decision tree visualization and model interpretation.  
[chartify](https://github.com/spotify/chartify/) - Generate charts.  
[VivaGraphJS](https://github.com/anvaka/VivaGraphJS) - Graph visualization (JS package).  
[pm](https://github.com/anvaka/pm) - Navigatable 3D graph visualization (JS package).  
[python-ternary](https://github.com/marcharper/python-ternary) - Triangle plots.  
[falcon](https://github.com/uwdata/falcon) - Interactive visualizations for big data.  
[hiplot](https://github.com/facebookresearch/hiplot) - High dimensional Interactive Plotting.  
[visdom](https://github.com/fossasia/visdom) - Live Visualizations.  
[mpl-scatter-density](https://github.com/astrofrog/mpl-scatter-density) - Scatter density plots. Alternative to 2d-histograms.   
[ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) - Complex heatmaps for multidimensional genomic data (R package).  
[largeVis](https://github.com/elbamos/largeVis) - Visualize embeddings (t-SNE etc.) (R package).  
[proplot](https://github.com/proplot-dev/proplot) - Matplotlib wrapper.  
[morpheus](https://software.broadinstitute.org/morpheus/) - Broad Institute tool matrix visualization and analysis software. [Source](https://github.com/cmap/morpheus.js), Tutorial: [1](https://www.youtube.com/watch?v=0nkYDeekhtQ), [2](https://www.youtube.com/watch?v=r9mN6MsxUb0), [Code](https://github.com/broadinstitute/BBBC021_Morpheus_Exercise).  
[jupyter-scatter](https://github.com/flekschas/jupyter-scatter) - Interactive 2D scatter plot widget for Jupyter.  

#### Colors
[palettable](https://github.com/jiffyclub/palettable) - Color palettes from [colorbrewer2](https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3).  
[colorcet](https://github.com/holoviz/colorcet) - Collection of perceptually uniform colormaps.  
[Named Colors Wheel](https://arantius.github.io/web-color-wheel/) - Color wheel for all named HTML colors.  

#### Dashboards
[py-shiny](https://github.com/rstudio/py-shiny) - Shiny for Python, [talk](https://www.youtube.com/watch?v=ijRBbtT2tgc).  
[superset](https://github.com/apache/superset) - Dashboarding solution by Apache.  
[streamlit](https://github.com/streamlit/streamlit) - Dashboarding solution. [Resources](https://github.com/marcskovmadsen/awesome-streamlit), [Gallery](http://awesome-streamlit.org/) [Components](https://www.streamlit.io/components), [bokeh-events](https://github.com/ash2shukla/streamlit-bokeh-events).  
[mercury](https://github.com/mljar/mercury) - Convert Python notebook to web app, [Example](https://github.com/pplonski/dashboard-python-jupyter-notebook).  
[dash](https://dash.plot.ly/gallery) - Dashboarding solution by plot.ly. [Resources](https://github.com/ucg8j/awesome-dash).  
[visdom](https://github.com/facebookresearch/visdom) - Dashboarding library by Facebook.  
[panel](https://panel.pyviz.org/index.html) - Dashboarding solution.  
[altair example](https://github.com/xhochy/altair-vue-vega-example) - [Video](https://www.youtube.com/watch?v=4L568emKOvs).  
[voila](https://github.com/QuantStack/voila) - Turn Jupyter notebooks into standalone web applications.  
[voila-gridstack](https://github.com/voila-dashboards/voila-gridstack) - Voila grid layout.  

#### UI
[gradio](https://github.com/gradio-app/gradio) - Create UIs for your machine learning model.  

#### Survey Tools
[samplics](https://github.com/samplics-org/samplics) - Sampling techniques for complex survey designs.  

#### Geographical Tools
[folium](https://github.com/python-visualization/folium) - Plot geographical maps using the Leaflet.js library, [jupyter plugin](https://github.com/jupyter-widgets/ipyleaflet).  
[gmaps](https://github.com/pbugnion/gmaps) - Google Maps for Jupyter notebooks.  
[stadiamaps](https://stadiamaps.com/) - Plot geographical maps.  
[datashader](https://github.com/bokeh/datashader) - Draw millions of points on a map.  
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html) - BallTree.  
[pynndescent](https://github.com/lmcinnes/pynndescent) - Nearest neighbor descent for approximate nearest neighbors.  
[geocoder](https://github.com/DenisCarriere/geocoder) - Geocoding of addresses, IP addresses.  
Conversion of different geo formats: [talk](https://www.youtube.com/watch?v=eHRggqAvczE), [repo](https://github.com/dillongardner/PyDataSpatialAnalysis)  
[geopandas](https://github.com/geopandas/geopandas) - Tools for geographic data  
Low Level Geospatial Tools (GEOS, GDAL/OGR, PROJ.4)  
Vector Data (Shapely, Fiona, Pyproj)  
Raster Data (Rasterio)  
Plotting (Descartes, Catropy)  
[Predict economic indicators from Open Street Map](https://janakiev.com/blog/osm-predict-economic-indicators/).   
[PySal](https://github.com/pysal/pysal) - Python Spatial Analysis Library.  
[geography](https://github.com/ushahidi/geograpy) - Extract countries, regions and cities from a URL or text.  
[cartogram](https://go-cart.io/cartogram) - Distorted maps based on population.  

#### Recommender Systems
Examples: [1](https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/), [2](https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223), [2-ipynb](https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb), [3](https://www.kaggle.com/morrisb/how-to-recommend-anything-deep-recommender).  
[surprise](https://github.com/NicolasHug/Surprise) - Recommender, [talk](https://www.youtube.com/watch?v=d7iIb_XVkZs).  
[implicit](https://github.com/benfred/implicit) - Fast Collaborative Filtering for Implicit Feedback Datasets.  
[spotlight](https://github.com/maciejkula/spotlight) - Deep recommender models using PyTorch.  
[lightfm](https://github.com/lyst/lightfm) - Recommendation algorithms for both implicit and explicit feedback.  
[funk-svd](https://github.com/gbolmier/funk-svd) - Fast SVD.  

#### Decision Tree Models
[Intro to Decision Trees and Random Forests](https://victorzhou.com/blog/intro-to-random-forests/), Intro to Gradient Boosting [1](https://explained.ai/gradient-boosting/), [2](https://www.gormanalysis.com/blog/gradient-boosting-explained/), [Decision Tree Visualization](https://explained.ai/decision-tree-viz/index.html)    
[lightgbm](https://github.com/Microsoft/LightGBM) - Gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, [doc](https://sites.google.com/view/lauraepp/parameters).  
[xgboost](https://github.com/dmlc/xgboost) - Gradient boosting (GBDT, GBRT or GBM) library, [doc](https://sites.google.com/view/lauraepp/parameters), Methods for CIs: [link1](https://stats.stackexchange.com/questions/255783/confidence-interval-for-xgb-forecast), [link2](https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b).  
[catboost](https://github.com/catboost/catboost) - Gradient boosting.  
[h2o](https://github.com/h2oai/h2o-3) -  Gradient boosting and general machine learning framework.  
[pycaret](https://github.com/pycaret/pycaret) - Wrapper for xgboost, lightgbm, catboost etc.  
[forestci](https://github.com/scikit-learn-contrib/forest-confidence-interval) - Confidence intervals for random forests.  
[grf](https://github.com/grf-labs/grf) - Generalized random forest.  
[dtreeviz](https://github.com/parrt/dtreeviz) - Decision tree visualization and model interpretation.  
[Nuance](https://github.com/SauceCat/Nuance) - Decision tree visualization.  
[rfpimp](https://github.com/parrt/random-forest-importances) - Feature Importance for RandomForests using Permuation Importance.  
Why the default feature importance for random forests is wrong: [link](http://explained.ai/rf-importance/index.html)  
[bartpy](https://github.com/JakeColtman/bartpy) - Bayesian Additive Regression Trees.  
[merf](https://github.com/manifoldai/merf) - Mixed Effects Random Forest for Clustering, [video](https://www.youtube.com/watch?v=gWj4ZwB7f3o)  
[groot](https://github.com/tudelft-cda-lab/GROOT) - Robust decision trees.  
[linear-tree](https://github.com/cerlymarco/linear-tree) - Trees with linear models at the leaves.  

#### Natural Language Processing (NLP) / Text Processing
[talk](https://www.youtube.com/watch?v=6zm9NC9uRkk)-[nb](https://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb), [nb2](https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html), [talk](https://www.youtube.com/watch?time_continue=2&v=sI7VpFNiy_I).  
[Text classification Intro](https://mlwhiz.com/blog/2018/12/17/text_classification/), [Preprocessing blog post](https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/).  
[gensim](https://radimrehurek.com/gensim/) - NLP, doc2vec, word2vec, text processing, topic modelling (LSA, LDA), [Example](https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html), [Coherence Model](https://radimrehurek.com/gensim/models/coherencemodel.html) for evaluation.  
Embeddings - [GloVe](https://nlp.stanford.edu/projects/glove/) ([[1](https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout)], [[2](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)]), [StarSpace](https://github.com/facebookresearch/StarSpace), [wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/), [visualization](https://projector.tensorflow.org/).  
[magnitude](https://github.com/plasticityai/magnitude) - Vector embedding utility package.  
[pyldavis](https://github.com/bmabey/pyLDAvis) - Visualization for topic modelling.  
[spaCy](https://spacy.io/) - NLP.  
[NTLK](https://www.nltk.org/) - NLP, helpful `KMeansClusterer` with `cosine_distance`.  
[pytext](https://github.com/facebookresearch/PyText) - NLP from Facebook.  
[fastText](https://github.com/facebookresearch/fastText) - Efficient text classification and representation learning.  
[annoy](https://github.com/spotify/annoy) - Approximate nearest neighbor search.  
[faiss](https://github.com/facebookresearch/faiss) - Approximate nearest neighbor search.  
[pysparnn](https://github.com/facebookresearch/pysparnn) - Approximate nearest neighbor search.  
[infomap](https://github.com/mapequation/infomap) - Cluster (word-)vectors to find topics.  
[datasketch](https://github.com/ekzhu/datasketch) - Probabilistic data structures for large data (MinHash, HyperLogLog).  
[flair](https://github.com/zalandoresearch/flair) - NLP Framework by Zalando.  
[stanza](https://github.com/stanfordnlp/stanza) - NLP Library.  
[Chatistics](https://github.com/MasterScrat/Chatistics) - Turn Messenger, Hangouts, WhatsApp and Telegram chat logs into DataFrames.  
[textdistance](https://github.com/life4/textdistance) - Collection for comparing distances between two or more sequences.  

#### Bio Image Analysis
[Awesome Cytodata](https://github.com/cytodata/awesome-cytodata)  

##### Tutorials
[bioimaging.org](https://www.bioimagingguide.org/welcome.html) - A biologists guide to planning and performing quantitative bioimaging experiments.  
[Bio-image Analysis Notebooks](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/intro.html) - Large collection of image processing workflows, including [point-spread-function estimation](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/18a_deconvolution/extract_psf.html) and [deconvolution](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/18a_deconvolution/introduction_deconvolution.html), [3D cell segmentation](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/Segmentation_3D.html), [feature extraction](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/22_feature_extraction/statistics_with_pyclesperanto.html) using [pyclesperanto](https://github.com/clEsperanto/pyclesperanto_prototype) and others.  
[python_for_microscopists](https://github.com/bnsreenu/python_for_microscopists) - Notebooks and associated [youtube channel](https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w/videos) for a variety of image processing tasks.  

##### Datasets
[jump-cellpainting](https://github.com/jump-cellpainting/datasets) - Cellpainting dataset.  
[MedMNIST](https://github.com/MedMNIST/MedMNIST) - Datasets for 2D and 3D Biomedical Image Classification.  
[CytoImageNet](https://github.com/stan-hua/CytoImageNet) - Huge diverse dataset like ImageNet but for cell images.  
[Haghighi](https://github.com/carpenterlab/2021_Haghighi_NatureMethods) - Gene Expression and Morphology Profiles.  
[broadinstitute/lincs-profiling-complementarity](https://github.com/broadinstitute/lincs-profiling-complementarity) - Cellpainting vs. L1000 assay.  

#### Biostatistics / Robust statistics
[MinCovDet](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html) - Robust estimator of covariance, RMPV, [Paper](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/wics.1421), [App1](https://journals.sagepub.com/doi/10.1177/1087057112469257?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed&), [App2](https://www.cell.com/cell-reports/pdf/S2211-1247(21)00694-X.pdf).  
[moderated z-score](https://clue.io/connectopedia/replicate_collapse) - Weighted average of z-scores based on Spearman correlation.  
[winsorize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html#scipy.stats.mstats.winsorize) - Simple adjustment of outliers.  

#### High-Content Screening Assay Design
[Zhang XHD (2008) - Novel analytic criteria and effective plate designs for quality control in genome-wide RNAi screens](https://slas-discovery.org/article/S2472-5552(22)08204-1/pdf)  
[Iversen - A Comparison of Assay Performance Measures in Screening Assays, Signal Window, Z′ Factor, and Assay Variability Ratio](https://www.slas-discovery.org/article/S2472-5552(22)08460-X/pdf)
[Z-factor](https://en.wikipedia.org/wiki/Z-factor) - Measure of statistical effect size.  
[Z'-factor](https://link.springer.com/referenceworkentry/10.1007/978-3-540-47648-1_6298) - Measure of statistical effect size.  
[CV](https://en.wikipedia.org/wiki/Coefficient_of_variation) - Coefficient of variation.  
[SSMD](https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference) - Strictly standardized mean difference.  
[Signal Window](https://www.intechopen.com/chapters/48130) - Assay quality measurement.  

#### Microscopy + Assay
[BD Spectrum Viewer](https://www.bdbiosciences.com/en-us/resources/bd-spectrum-viewer) - Calculate spectral overlap, bleed through for fluorescence microscopy dyes.  
[SpectraViewer](https://www.perkinelmer.com/lab-products-and-services/spectraviewer) - Visualize the spectral compatibility of fluorophores (PerkinElmer).  
[Thermofisher Spectrum Viewer](https://www.thermofisher.com/order/stain-it) - Thermofisher Spectrum Viewer.  
[Microscopy Resolution Calculator](https://www.microscope.healthcare.nikon.com/microtools/resolution-calculator) - Calculate resolution of images (Nikon).  
[PlateEditor](https://github.com/vindelorme/PlateEditor) - Drug Layout for plates, [app](https://plateeditor.sourceforge.io/), [zip](https://sourceforge.net/projects/plateeditor/), [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252488).  

##### Image Formats and Converters
OME-Zarr - [paper](https://www.biorxiv.org/content/10.1101/2023.02.17.528834v1.full), [standard](https://ngff.openmicroscopy.org/latest/)  
[bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) - Various formats to zarr.  
[raw2ometiff](https://github.com/glencoesoftware/raw2ometiff) - Zarr to tiff.  
[BatchConvert](https://github.com/Euro-BioImaging/BatchConvert) - Wrapper for bioformats2raw to parallelize conversions with nextflow, [video](https://www.youtube.com/watch?v=DeCWV274l0c).  
REMBI model - Recommended Metadata for Biological Images, BioImage Archive: [Study Component Guidance](https://www.ebi.ac.uk/bioimage-archive/rembi-help-examples/), [File List Guide](https://www.ebi.ac.uk/bioimage-archive/help-file-list/), [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8606015/), [video](https://www.youtube.com/watch?v=GVmfOpuP2_c), [spreadsheet](https://docs.google.com/spreadsheets/d/1Ck1NeLp-ZN4eMGdNYo2nV6KLEdSfN6oQBKnnWU6Npeo/edit#gid=1023506919)  

##### Matrix Formats
[anndata](https://github.com/scverse/anndata) - annotated data matrices in memory and on disk, [Docs](https://anndata.readthedocs.io/en/latest/index.html).  
[muon](https://github.com/scverse/muon) - Multimodal omics framework.  
[mudata](https://github.com/scverse/mudata) - Multimodal Data (.h5mu) implementation.  
[bdz](https://github.com/openssbd/bdz) - Zarr-based format for storing quantitative biological dynamics data.  

#### Image Viewers
[vizarr](https://github.com/hms-dbmi/vizarr) - Browser-based image viewer for zarr format.  
[avivator](https://github.com/hms-dbmi/viv) - Browser-based image viewer for tiff files.  
[napari](https://github.com/napari/napari) - Image viewer and image processing tool.    
[Fiji](https://fiji.sc/) - General purpose tool. Image viewer and image processing tool.  
[OMERO](https://www.openmicroscopy.org/omero/) - Image viewer for high-content screening. [IDR](https://idr.openmicroscopy.org/) uses OMERO. [Intro](https://www.youtube.com/watch?v=nSCrMO_c-5s)   
[fiftyone](https://github.com/voxel51/fiftyone) - Viewer and tool for building high-quality datasets and computer vision models.  
Image Data Explorer - Microscopy Image Viewer, [Shiny App](https://shiny-portal.embl.de/shinyapps/app/01_image-data-explorer), [Video](https://www.youtube.com/watch?v=H8zIZvOt1MA).  
[ImSwitch](https://github.com/ImSwitch/ImSwitch) - Microscopy Image Viewer, [Doc](https://imswitch.readthedocs.io/en/stable/gui.html), [Video](https://www.youtube.com/watch?v=XsbnMkGSPQQ).  
[pixmi](https://github.com/piximi/piximi) - Web-based image annotation and classification tool, [App](https://www.piximi.app/).  
[DeepCell Label](https://label.deepcell.org/) - Data labeling tool to segment images, [Video](https://www.youtube.com/watch?v=zfsvUBkEeow).  
  
##### Image Restoration and Denoising
[aydin](https://github.com/royerlab/aydin) - Image denoising.  
[DivNoising](https://github.com/juglab/DivNoising) - Unsupervised denoising method.  
[CSBDeep](https://github.com/CSBDeep/CSBDeep) - Content-aware image restoration, [Project page](https://csbdeep.bioimagecomputing.com/tools/).  

##### Illumination correction + Bleed through correction
[skimage](https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) - Illumination correction (CLAHE).  
[cidre](https://github.com/smithk/cidre) - Illumination correction method for optical microscopy.  
[BaSiCPy](https://github.com/peng-lab/BaSiCPy) - Background and Shading Correction of Optical Microscopy Images, [BaSiC](https://github.com/marrlab/BaSiC).  
[cytoflow](https://github.com/cytoflow/cytoflow) - Flow cytometry. Includes Bleedthrough correction methods.  
Linear unmixing in Fiji for Bleedthrough Correction - [Youtube](https://www.youtube.com/watch?v=W90qs0J29v8).  
Bleedthrough Correction using Lumos and Fiji - [Link](https://imagej.net/plugins/lumos-spectral-unmixing).  

##### Platforms and Pipelines
[CellProfiler](https://github.com/CellProfiler/CellProfiler), [CellProfilerAnalyst](https://github.com/CellProfiler/CellProfiler-Analyst) - Create image analysis pipelines.  
[fractal](https://fractal-analytics-platform.github.io/) - Framework to process high-content imaging data.  
[atomai](https://github.com/pycroscopy/atomai) - Deep and Machine Learning for Microscopy.  
[py-clesperanto](https://github.com/clesperanto/pyclesperanto_prototype/) - Tools for 3D microscopy analysis, [deskewing](https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/transforms/deskew.ipynb) and lots of other tutorials, interacts with napari.  
[qupath](https://github.com/qupath/qupath) - Image analysis.  

##### Microscopy Pipelines
[SCIP](https://scalable-cytometry-image-processing.readthedocs.io/en/latest/usage.html) - Image processing pipeline on top of Dask.  
[DeepCell Kiosk](https://github.com/vanvalenlab/kiosk-console/tree/master) - Image analysis platform.  
[IMCWorkflow](https://github.com/BodenmillerGroup/IMCWorkflow/) - Image analysis pipeline using [steinbock](https://github.com/BodenmillerGroup/steinbock), [Twitter](https://twitter.com/NilsEling/status/1715020265963258087), [Paper](https://www.nature.com/articles/s41596-023-00881-0), [workflow](https://bodenmillergroup.github.io/IMCDataAnalysis/).  

##### Labsyspharm
[mcmicro](https://github.com/labsyspharm/mcmicro) - Multiple-choice microscopy pipeline, [Website](https://mcmicro.org/overview/), [Paper](https://www.nature.com/articles/s41592-021-01308-y).  
[MCQuant](https://github.com/labsyspharm/quantification) - Quantification of cell features.  
[cylinter](https://github.com/labsyspharm/cylinter) - Quality assurance for microscopy images, [Website](https://labsyspharm.github.io/cylinter/).  
[ashlar](https://github.com/labsyspharm/ashlar) - Whole-slide microscopy image stitching and registration.  
[scimap](https://github.com/labsyspharm/scimap) - Spatial Single-Cell Analysis Toolkit.  

##### Cell Segmentation
[microscopy-tree](https://biomag-lab.github.io/microscopy-tree/) - Review of cell segmentation algorithms, [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0962892421002518).  
[BioImage.IO](https://bioimage.io/#/) - BioImage Model Zoo.  
[MEDIAR](https://github.com/Lee-Gihun/MEDIAR) - Cell segmentation.  
[cellpose](https://github.com/mouseland/cellpose) - Cell segmentation. [Paper](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1), [Dataset](https://www.cellpose.org/dataset).  
[stardist](https://github.com/stardist/stardist) - Cell segmentation with Star-convex Shapes.  
[UnMicst](https://github.com/HMS-IDAC/UnMicst) - Identifying Cells and Segmenting Tissue.  
[nnUnet](https://github.com/MIC-DKFZ/nnUNet) - 3D biomedical image segmentation.  
[allencell](https://www.allencell.org/segmenter.html) - Tools for 3D segmentation, classical and deep learning methods.  
[Cell-ACDC](https://github.com/SchmollerLab/Cell_ACDC) - Python GUI for cell segmentation and tracking.  
[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki) - Deep-Learning in Microscopy.  
[EmbedSeg](https://github.com/juglab/EmbedSeg) - Embedding-based Instance Segmentation.  
[micro-sam](https://github.com/computational-cell-analytics/micro-sam) - SegmentAnything for Microscopy.  
[deepcell-tf](https://github.com/vanvalenlab/deepcell-tf/tree/master) - Cell segmentation, [DeepCell](https://deepcell.org/).  

##### Cell Segmentation Datasets
[cellpose](https://www.cellpose.org/dataset) - Cell images.  
[omnipose](http://www.cellpose.org/dataset_omnipose) - Cell images.  
[LIVECell](https://github.com/sartorius-research/LIVECell) - Cell images.  
[Sartorius](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview) - Neurons.  
[EmbedSeg](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0) - 2D + 3D images.  

##### Evaluation
[seg-eval](https://github.com/lstrgar/seg-eval) - Cell segmentation performance evaluation without Ground Truth labels, [Paper](https://www.biorxiv.org/content/10.1101/2023.02.23.529809v1.full.pdf).  

##### Feature Engineering Images
[Computer vision challenges in drug discovery - Maciej Hermanowicz](https://www.youtube.com/watch?v=Y5GJmnIhvFk)  
[CellProfiler](https://github.com/CellProfiler/CellProfiler) - Biological image analysis.   
[scikit-image](https://github.com/scikit-image/scikit-image) - Image processing.  
[scikit-image regionprops](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) - Regionprops: area, eccentricity, extent.  
[mahotas](https://github.com/luispedro/mahotas) - Zernike, Haralick, LBP, and TAS features, [example](https://github.com/luispedro/python-image-tutorial/blob/master/Segmenting%20cell%20images%20(fluorescent%20microscopy).ipynb).   
[pyradiomics](https://github.com/AIM-Harvard/pyradiomics) - Radiomics features from medical imaging.  
[pyefd](https://github.com/hbldh/pyefd) - Elliptical feature descriptor, approximating a contour with a Fourier series.  
[pyvips](https://github.com/libvips/pyvips/tree/master) - Faster image processing operations.  

#### Domain Adaptation / Batch-Effect Correction 
[Tran - A benchmark of batch-effect correction methods for single-cell RNA sequencing data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9), [Code](https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking).  
[R Tutorial on correcting batch effects](https://broadinstitute.github.io/2019_scWorkshop/correcting-batch-effects.html).  
[harmonypy](https://github.com/slowkow/harmonypy) - Fuzzy k-means and locally linear adjustments.  
[pyliger](https://github.com/welch-lab/pyliger) - Batch-effect correction, [R package](https://github.com/welch-lab/liger).  
[nimfa](https://github.com/mims-harvard/nimfa) - Nonnegative matrix factorization.  
[scgen](https://github.com/theislab/scgen) - Batch removal. [Doc](https://scgen.readthedocs.io/en/stable/).  
[CORAL](https://github.com/google-research/google-research/tree/30e54523f08d963ced3fbb37c00e9225579d2e1d/correct_batch_effects_wdn) - Correcting for Batch Effects Using Wasserstein Distance, [Code](https://github.com/google-research/google-research/blob/30e54523f08d963ced3fbb37c00e9225579d2e1d/correct_batch_effects_wdn/transform.py#L152), [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7050548/).   
[adapt](https://github.com/adapt-python/adapt) - Awesome Domain Adaptation Python Toolbox.  
[pytorch-adapt](https://github.com/KevinMusgrave/pytorch-adapt) - Various neural network models for domain adaptation.  

##### Sequencing
[Single cell tutorial](https://github.com/theislab/single-cell-tutorial).  
[PyDESeq2](https://github.com/owkin/PyDESeq2) - Analyzing RNA-seq data.  
[cellxgene](https://github.com/chanzuckerberg/cellxgene) - Interactive explorer for single-cell transcriptomics data.  
[scanpy](https://github.com/theislab/scanpy) - Analyze single-cell gene expression data, [tutorial](https://github.com/theislab/single-cell-tutorial).  
[besca](https://github.com/bedapub/besca) - Beyond single-cell analysis.  
[janggu](https://github.com/BIMSBbioinfo/janggu) - Deep Learning for Genomics.  
[gdsctools](https://github.com/CancerRxGene/gdsctools) - Drug responses in the context of the Genomics of Drug Sensitivity in Cancer project, ANOVA, IC50, MoBEM, [doc](https://gdsctools.readthedocs.io/en/master/).  
[monkeybread](https://github.com/immunitastx/monkeybread) - Analysis of single-cell spatial transcriptomics data.  

##### Drug discovery
[TDC](https://github.com/mims-harvard/TDC/tree/main) - Drug Discovery and Development.  
[DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) - Deep Learning Based Molecular Modelling and Prediction Toolkit.  

#### Neural Networks
[Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) - Stanford CS class.  
[mit6874](https://mit6874.github.io/) - Computational Systems Biology: Deep Learning in the Life Sciences.  
[ConvNet Shape Calculator](https://madebyollin.github.io/convnet-calculator/) - Calculate output dimensions of Conv2D layer.  
[Great Gradient Descent Article](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9).  
[Intro to semi-supervised learning](https://lilianweng.github.io/lil-log/2021/12/05/semi-supervised-learning.html).  

##### Tutorials & Viewer
[fast.ai course](https://course.fast.ai/) - Practical Deep Learning for Coders.  
[Tensorflow without a PhD](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd) - Neural Network course by Google.  
Feature Visualization: [Blog](https://distill.pub/2017/feature-visualization/), [PPT](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)  
[Tensorflow Playground](https://playground.tensorflow.org/)  
[Visualization of optimization algorithms](http://vis.ensmallen.org/), [Another visualization](https://github.com/jettify/pytorch-optimizer)    
[cutouts-explorer](https://github.com/mgckind/cutouts-explorer) - Image Viewer.  

##### Image Related
[imgaug](https://github.com/aleju/imgaug) - More sophisticated image preprocessing.  
[Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library.  
[keras preprocessing](https://keras.io/preprocessing/image/) - Preprocess images.  
[albumentations](https://github.com/albu/albumentations) - Wrapper around imgaug and other libraries.  
[augmix](https://github.com/google-research/augmix) - Image augmentation from Google.  
[kornia](https://github.com/kornia/kornia) - Image augmentation, feature extraction and loss functions.  
[augly](https://github.com/facebookresearch/AugLy) - Image, audio, text, video augmentation from Facebook.  
[pyvips](https://github.com/libvips/pyvips/tree/master) - Faster image processing operations.  

##### Lossfunction Related
[SegLoss](https://github.com/JunMa11/SegLoss) - List of loss functions for medical image segmentation.  

##### Activation Functions
[rational_activations](https://github.com/ml-research/rational_activations) - Rational activation functions.  

##### Text Related
[ktext](https://github.com/hamelsmu/ktext) - Utilities for pre-processing text for deep learning in Keras.   
[textgenrnn](https://github.com/minimaxir/textgenrnn) - Ready-to-use LSTM for text generation.  
[ctrl](https://github.com/salesforce/ctrl) - Text generation.  

##### Neural network and deep learning frameworks
[OpenMMLab](https://github.com/open-mmlab) - Framework for segmentation, classification and lots of other computer vision tasks.  
[caffe](https://github.com/BVLC/caffe) - Deep learning framework, [pretrained models](https://github.com/BVLC/caffe/wiki/Model-Zoo).  
[mxnet](https://github.com/apache/incubator-mxnet) - Deep learning framework, [book](https://d2l.ai/index.html).  

##### Libs General
[keras](https://keras.io/) - Neural Networks on top of [tensorflow](https://www.tensorflow.org/), [examples](https://gist.github.com/candlewill/552fa102352ccce42fd829ae26277d24).  
[keras-contrib](https://github.com/keras-team/keras-contrib) - Keras community contributions.  
[keras-tuner](https://github.com/keras-team/keras-tuner) - Hyperparameter tuning for Keras.  
[hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: Convenient hyperparameter optimization wrapper.  
[elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark.  
[tflearn](https://github.com/tflearn/tflearn) - Neural Networks on top of TensorFlow.  
[tensorlayer](https://github.com/tensorlayer/tensorlayer) - Neural Networks on top of TensorFlow, [tricks](https://github.com/wagamamaz/tensorlayer-tricks).  
[tensorforce](https://github.com/reinforceio/tensorforce) - TensorFlow for applied reinforcement learning.  
[autokeras](https://github.com/jhfjhfj1/autokeras) - AutoML for deep learning.  
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) - Plot neural networks.  
[lucid](https://github.com/tensorflow/lucid) - Neural network interpretability, [Activation Maps](https://openai.com/blog/introducing-activation-atlases/).  
[tcav](https://github.com/tensorflow/tcav) - Interpretability method.  
[AdaBound](https://github.com/Luolc/AdaBound) - Optimizer that trains as fast as Adam and as good as SGD, [alt](https://github.com/titu1994/keras-adabound).  
[foolbox](https://github.com/bethgelab/foolbox) - Adversarial examples that fool neural networks.  
[hiddenlayer](https://github.com/waleedka/hiddenlayer) - Training metrics.  
[imgclsmob](https://github.com/osmr/imgclsmob) - Pretrained models.  
[netron](https://github.com/lutzroeder/netron) - Visualizer for deep learning and machine learning models.  
[ffcv](https://github.com/libffcv/ffcv) - Fast dataloader.  

##### Libs PyTorch
[Good PyTorch Introduction](https://cs230.stanford.edu/blog/pytorch/)    
[skorch](https://github.com/dnouri/skorch) - Scikit-learn compatible neural network library that wraps PyTorch, [talk](https://www.youtube.com/watch?v=0J7FaLk0bmQ), [slides](https://github.com/thomasjpfan/skorch_talk).  
[fastai](https://github.com/fastai/fastai) - Neural Networks in PyTorch.  
[timm](https://github.com/rwightman/pytorch-image-models) - PyTorch image models.  
[ignite](https://github.com/pytorch/ignite) - Highlevel library for PyTorch.  
[torchcv](https://github.com/donnyyou/torchcv) - Deep Learning in Computer Vision.  
[pytorch-optimizer](https://github.com/jettify/pytorch-optimizer) - Collection of optimizers for PyTorch.  
[pytorch-lightning](https://github.com/PyTorchLightning/PyTorch-lightning) - Wrapper around PyTorch.  
[lightly](https://github.com/lightly-ai/lightly) - MoCo, SimCLR, SimSiam, Barlow Twins, BYOL, NNCLR.  
[MONAI](https://github.com/project-monai/monai) - Deep learning in healthcare imaging.  
[kornia](https://github.com/kornia/kornia) - Image transformations, epipolar geometry, depth estimation.  
[torchinfo](https://github.com/TylerYep/torchinfo) - Nice model summary.  
[lovely-tensors](https://github.com/xl0/lovely-tensors/) - Inspect tensors, mean, std, inf values.  

##### Distributed Libs
[flexflow](https://github.com/flexflow/FlexFlow) - Distributed TensorFlow Keras and PyTorch.  
[horovod](https://github.com/horovod/horovod) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.  

##### Architecture Visualization
[Awesome List](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network).  
[netron](https://github.com/lutzroeder/netron) - Viewer for neural networks.  
[visualkeras](https://github.com/paulgavrikov/visualkeras) - Visualize Keras networks.  

##### Object detection / Instance Segmentation
[Metrics reloaded: Recommendations for image analysis validation](https://arxiv.org/abs/2206.01653) - Guide for choosing correct image analysis metrics, [Code](https://github.com/Project-MONAI/MetricsReloaded), [Twitter Thread](https://twitter.com/lena_maierhein/status/1625450342006521857)  
[Good Yolo Explanation](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)  
[yolact](https://github.com/dbolya/yolact) - Fully convolutional model for real-time instance segmentation.  
[EfficientDet Pytorch](https://github.com/toandaominh1997/EfficientDet.Pytorch), [EfficientDet Keras](https://github.com/xuannianz/EfficientDet) - Scalable and Efficient Object Detection.  
[detectron2](https://github.com/facebookresearch/detectron2) - Object Detection (Mask R-CNN) by Facebook.  
[simpledet](https://github.com/TuSimple/simpledet) - Object Detection and Instance Recognition.  
[CenterNet](https://github.com/xingyizhou/CenterNet) - Object detection.  
[FCOS](https://github.com/tianzhi0549/FCOS) - Fully Convolutional One-Stage Object Detection.  
[norfair](https://github.com/tryolabs/norfair) - Real-time 2D object tracking.  
[Detic](https://github.com/facebookresearch/Detic) -  Detector with image classes that can use image-level labels (facebookresearch).  
[EasyCV](https://github.com/alibaba/EasyCV) - Image segmentation, classification, metric-learning, object detection, pose estimation.  

##### Image Classification
[nfnets](https://github.com/ypeleg/nfnets-keras) - Neural network.   
[efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch) - Neural network.   
[pycls](https://github.com/facebookresearch/pycls) - PyTorch image classification networks: ResNet, ResNeXt, EfficientNet, and RegNet (by Facebook).  

##### Applications and Snippets
[SPADE](https://github.com/nvlabs/spade) - Semantic Image Synthesis.  
[Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737), [code](https://github.com/entron/entity-embedding-rossmann), [kaggle](https://www.kaggle.com/aquatic/entity-embedding-neural-net/code)  
[Image Super-Resolution](https://github.com/idealo/image-super-resolution) - Super-scaling using a Residual Dense Network.  
Cell Segmentation - [Talk](https://www.youtube.com/watch?v=dVFZpodqJiI), Blog Posts: [1](https://www.thomasjpfan.com/2018/07/nuclei-image-segmentation-tutorial/), [2](https://www.thomasjpfan.com/2017/08/hassle-free-unets/)  
[deeplearning-models](https://github.com/rasbt/deeplearning-models) - Deep learning models.  

##### Variational Autoencoders (VAEs)
[Variational Autoencoder Explanation Video](https://www.youtube.com/watch?v=9zKuYvjFFS8)  
[disentanglement_lib](https://github.com/google-research/disentanglement_lib) - BetaVAE, FactorVAE, BetaTCVAE, DIP-VAE.  
[ladder-vae-pytorch](https://github.com/addtt/ladder-vae-pytorch) - Ladder Variational Autoencoders (LVAE).  
[benchmark_VAE](https://github.com/clementchadebec/benchmark_VAE) - Unifying Generative Autoencoder implementations.  

##### Generative Adversarial Networks (GANs)
[Awesome GAN Applications](https://github.com/nashory/gans-awesome-applications)  
[The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo) - List of Generative Adversarial Networks.  
[CycleGAN and Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) - Various image-to-image tasks.  
[TensorFlow GAN implementations](https://github.com/hwalsuklee/tensorflow-generative-model-collections)  
[PyTorch GAN implementations](https://github.com/znxlwm/pytorch-generative-model-collections)  
[PyTorch GAN implementations](https://github.com/eriklindernoren/PyTorch-GAN#adversarial-autoencoder)  
[StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) - PyTorch GAN implementations.  

##### Transformers
[SegFormer](https://github.com/NVlabs/SegFormer) - Simple and Efficient Design for Semantic Segmentation with Transformers.  
[esvit](https://github.com/microsoft/esvit) - Efficient self-supervised Vision Transformers.  
[nystromformer](https://github.com/Rishit-dagli/Nystromformer) - More efficient transformer because of approximate self-attention.  

##### Deep learning on structured data
[Great overview for deep learning for tabular data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html)  

##### Graph-Based Neural Networks
[How to do Deep Learning on Graphs with Graph Convolutional Networks](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)  
[Introduction To Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/)  
[An attempt at demystifying graph deep learning](https://ericmjl.github.io/essays-on-data-science/machine-learning/graph-nets/)  
[ogb](https://ogb.stanford.edu/) - Open Graph Benchmark, Benchmark datasets.  
[networkx](https://github.com/networkx/networkx) - Graph library.  
[cugraph](https://github.com/rapidsai/cugraph) - RAPIDS, Graph library on the GPU.  
[pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) - Various methods for deep learning on graphs.  
[dgl](https://github.com/dmlc/dgl) - Deep Graph Library.  
[graph_nets](https://github.com/deepmind/graph_nets) - Build graph networks in TensorFlow, by DeepMind.  

#### Model conversion
[hummingbird](https://github.com/microsoft/hummingbird) - Compile trained ML models into tensor computations (by Microsoft).  

#### GPU
[cuML](https://github.com/rapidsai/cuml) - RAPIDS, Run traditional tabular ML tasks on GPUs, [Intro](https://www.youtube.com/watch?v=6XzS5XcpicM&t=2m50s).  
[thundergbm](https://github.com/Xtra-Computing/thundergbm) - GBDTs and Random Forest.  
[thundersvm](https://github.com/Xtra-Computing/thundersvm) - Support Vector Machines.  
Legate Numpy - Distributed Numpy array multiple using GPUs by Nvidia (not released yet) [video](https://www.youtube.com/watch?v=Jxxs_moibog).  

#### Regression
Understanding SVM Regression: [slides](https://cs.adelaide.edu.au/~chhshen/teaching/ML_SVR.pdf), [forum](https://www.quora.com/How-does-support-vector-regression-work), [paper](http://alex.smola.org/papers/2003/SmoSch03b.pdf)  

[pyearth](https://github.com/scikit-learn-contrib/py-earth) - Multivariate Adaptive Regression Splines (MARS), [tutorial](https://uc-r.github.io/mars).  
[pygam](https://github.com/dswah/pyGAM) - Generalized Additive Models (GAMs), [Explanation](https://multithreaded.stitchfix.com/blog/2015/07/30/gam/).  
[GLRM](https://github.com/madeleineudell/LowRankModels.jl) - Generalized Low Rank Models.  
[tweedie](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie) - Specialized distribution for zero inflated targets, [Talk](https://www.youtube.com/watch?v=-o0lpHBq85I).  
[MAPIE](https://github.com/scikit-learn-contrib/MAPIE) - Estimating prediction intervals.  
[Regressio](https://github.com/brendanartley/Regressio) - Regression and Spline models.  

#### Polynomials
[orthopy](https://github.com/nschloe/orthopy) - Orthogonal polynomials in all shapes and sizes.  

#### Classification
[Talk](https://www.youtube.com/watch?v=DkLPYccEJ8Y), [Notebook](https://github.com/ianozsvald/data_science_delivered/blob/master/ml_creating_correct_capable_classifiers.ipynb)  
[Blog post: Probability Scoring](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)  
[All classification metrics](http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf)  
[DESlib](https://github.com/scikit-learn-contrib/DESlib) - Dynamic classifier and ensemble selection.  
[human-learn](https://github.com/koaning/human-learn) - Create and tune classifier based on your rule set.  

#### Metric Learning
[Contrastive Representation Learning](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html)  
  
[metric-learn](https://github.com/scikit-learn-contrib/metric-learn) - Supervised and weakly-supervised metric learning algorithms.  
[pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) - PyTorch metric learning.  
[deep_metric_learning](https://github.com/ronekko/deep_metric_learning) - Methods for deep metric learning.  
[ivis](https://bering-ivis.readthedocs.io/en/latest/supervised.html) - Metric learning using siamese neural networks.  
[TensorFlow similarity](https://github.com/tensorflow/similarity) - Metric learning.  

#### Distance Functions
[scipy.spatial](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) - All kinds of distance metrics.  
[pyemd](https://github.com/wmayner/pyemd) - Earth Mover's Distance / Wasserstein distance, similarity between histograms. [OpenCV implementation](https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html), [POT implementation](https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html)   
[dcor](https://github.com/vnmabus/dcor)  - Distance correlation and related Energy statistics.  
[GeomLoss](https://www.kernel-operations.io/geomloss/) - Kernel norms, Hausdorff divergences, Debiased Sinkhorn divergences (=approximation of Wasserstein distance).  

#### Self-supervised Learning
[lightly](https://github.com/lightly-ai/lightly) - MoCo, SimCLR, SimSiam, Barlow Twins, BYOL, NNCLR.  
[vissl](https://github.com/facebookresearch/vissl) - Self-Supervised Learning with PyTorch: RotNet, Jigsaw, NPID, ClusterFit, PIRL, SimCLR, MoCo, DeepCluster, SwAV.  

#### Clustering
[Overview of clustering algorithms applied image data (= Deep Clustering)](https://deepnotes.io/deep-clustering).  
[Clustering with Deep Learning: Taxonomy and New Methods](https://arxiv.org/pdf/1801.07648.pdf).  
[Hierarchical Cluster Analysis (R Tutorial)](https://uc-r.github.io/hc_clustering) - Dendrogram, Tanglegram  
[hdbscan](https://github.com/scikit-learn-contrib/hdbscan) - Clustering algorithm, [talk](https://www.youtube.com/watch?v=dGsxd67IFiU), [blog](https://towardsdatascience.com/understanding-hdbscan-and-density-based-clustering-121dbee1320e).  
[pyclustering](https://github.com/annoviko/pyclustering) - All sorts of clustering algorithms.  
[FCPS](https://github.com/Mthrun/FCPS) -  Fundamental Clustering Problems Suite (R package).  
[GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) - Generalized k-means clustering using a mixture of Gaussian distributions, [video](https://www.youtube.com/watch?v=aICqoAG5BXQ).  
[nmslib](https://github.com/nmslib/nmslib) - Similarity search library and toolkit for evaluation of k-NN methods.  
[merf](https://github.com/manifoldai/merf) - Mixed Effects Random Forest for Clustering, [video](https://www.youtube.com/watch?v=gWj4ZwB7f3o)  
[tree-SNE](https://github.com/isaacrob/treesne) - Hierarchical clustering algorithm based on t-SNE.  
[MiniSom](https://github.com/JustGlowing/minisom) - Pure Python implementation of the Self Organizing Maps.  
[distribution_clustering](https://github.com/EricElmoznino/distribution_clustering), [paper](https://arxiv.org/abs/1804.02624), [related paper](https://arxiv.org/abs/2003.07770), [alt](https://github.com/r0f1/distribution_clustering).  
[phenograph](https://github.com/dpeerlab/phenograph) - Clustering by community detection.  
[FastPG](https://github.com/sararselitsky/FastPG) - Clustering of single cell data (RNA). Improvement of phenograph, [Paper](https://www.researchgate.net/publication/342339899_FastPG_Fast_clustering_of_millions_of_single_cells).  
[HypHC](https://github.com/HazyResearch/HypHC) - Hyperbolic Hierarchical Clustering.  
[BanditPAM](https://github.com/ThrunGroup/BanditPAM) - Improved k-Medoids Clustering.  
[dendextend](https://github.com/talgalili/dendextend) - Comparing dendrograms (R package).  
[DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM) - Deep Clustering With An Unknown Number of Clusters.  

##### Clustering Evalutation
[Wagner, Wagner - Comparing Clusterings - An Overview](https://publikationen.bibliothek.kit.edu/1000011477/812079)
* [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
* [Normalized Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
* [Adjusted Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
* [Fowlkes-Mallows Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html)
* [Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
* [Variation of Information](https://gist.github.com/jwcarr/626cbc80e0006b526688), [Julia](https://clusteringjl.readthedocs.io/en/latest/varinfo.html)
* [Pair Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html)
* [Consensus Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.consensus_score.html) - The similarity of two sets of biclusters.

[Assessing the quality of a clustering (video)](https://www.youtube.com/watch?v=Mf6MqIS2ql4)   
[fpc](https://cran.r-project.org/web/packages/fpc/index.html) - Various methods for clustering and cluster validation (R package).  
* Minimum distance between any two clusters
* Distance between centroids
* p-separation index: Like minimum distance. Look at the average distance to nearest point in different cluster for p=10% "border" points in any cluster. Measuring density, measuring mountains vs valleys
* Estimate density by weighted count of close points 

Other measures:
* Within-cluster average distance
* Mean of within-cluster average distance over nearest-cluster average distance (silhouette score)
* Within-cluster similarity measure to normal/uniform
* Within-cluster (squared) distance to centroid (this is the k-Means loss function)
* Correlation coefficient between distance we originally had to the distance the are induced by the clustering (Huberts Gamma)
* Entropy of cluster sizes
* Average largest within-cluster gap
* Variation of clusterings on bootstrapped data

#### Multi-label classification
[scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) - Multi-label classification, [talk](https://www.youtube.com/watch?v=m-tAASQA7XQ&t=18m57s).  

#### Signal Processing and Filtering
[Stanford Lecture Series on Fourier Transformation](https://see.stanford.edu/Course/EE261), [Youtube](https://www.youtube.com/watch?v=gZNm7L96pfY&list=PLB24BC7956EE040CD&index=1), [Lecture Notes](https://see.stanford.edu/materials/lsoftaee261/book-fall-07.pdf).  
[Visual Fourier explanation](https://dsego.github.io/demystifying-fourier/).  
[The Scientist & Engineer's Guide to Digital Signal Processing (1999)](https://www.analog.com/en/education/education-library/scientist_engineers_guide.html) - Chapter 3 has good introduction to Bessel, Butterworth and Chebyshev filters.  
[Kalman Filter article](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures).  
[Kalman Filter book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) - Focuses on intuition using Jupyter Notebooks. Includes Bayesian and various Kalman filters.  
[Interactive Tool](https://fiiir.com/) for FIR and IIR filters, [Examples](https://plot.ly/python/fft-filters/).  
[filterpy](https://github.com/rlabbe/filterpy) - Kalman filtering and optimal estimation library.  

#### Filtering in Python
[scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
* [Butterworth low-pass filter example](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform)
* [Savitzky–Golay filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html), [W](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)  
[pandas.Series.rolling](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html) - Choose appropriate `win_type`.  

#### Geometry
[geomstats](https://github.com/geomstats/geomstats) - Computations and statistics on manifolds with geometric structures.  

#### Time Series
[statsmodels](https://www.statsmodels.org/dev/tsa.html) - Time series analysis, [seasonal decompose](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) [example](https://gist.github.com/balzer82/5cec6ad7adc1b550e7ee), [SARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html), [granger causality](http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html).  
[kats](https://github.com/facebookresearch/kats) - Time series prediction library by Facebook.  
[prophet](https://github.com/facebook/prophet) - Time series prediction library by Facebook.  
[neural_prophet](https://github.com/ourownstory/neural_prophet) - Time series prediction built on PyTorch.  
[pyramid](https://github.com/tgsmith61591/pyramid), [pmdarima](https://github.com/tgsmith61591/pmdarima) - Wrapper for (Auto-) ARIMA.  
[modeltime](https://cran.r-project.org/web/packages/modeltime/index.html) - Time series forecasting framework (R package).  
[pyflux](https://github.com/RJT1990/pyflux) - Time series prediction algorithms (ARIMA, GARCH, GAS, Bayesian).  
[atspy](https://github.com/firmai/atspy) - Automated Time Series Models.  
[pm-prophet](https://github.com/luke14free/pm-prophet) - Time series prediction and decomposition library.  
[htsprophet](https://github.com/CollinRooney12/htsprophet) - Hierarchical Time Series Forecasting using Prophet.  
[nupic](https://github.com/numenta/nupic) - Hierarchical Temporal Memory (HTM) for Time Series Prediction and Anomaly Detection.  
[tensorflow](https://github.com/tensorflow/tensorflow/) - LSTM and others, examples: [link](
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
), [link](https://github.com/hzy46/TensorFlow-Time-Series-Examples), seq2seq: [1](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/), [2](https://github.com/guillaume-chevalier/seq2seq-signal-prediction), [3](https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb), [4](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction)  
[tspreprocess](https://github.com/MaxBenChrist/tspreprocess) - Preprocessing: Denoising, Compression, Resampling.  
[tsfresh](https://github.com/blue-yonder/tsfresh) - Time series feature engineering.  
[tsfel](https://github.com/fraunhoferportugal/tsfel) - Time series feature extraction.  
[thunder](https://github.com/thunder-project/thunder) - Data structures and algorithms for loading, processing, and analyzing time series data.  
[gatspy](https://www.astroml.org/gatspy/) - General tools for Astronomical Time Series, [talk](https://www.youtube.com/watch?v=E4NMZyfao2c).  
[gendis](https://github.com/IBCNServices/GENDIS) - shapelets, [example](https://github.com/IBCNServices/GENDIS/blob/master/gendis/example.ipynb).  
[tslearn](https://github.com/rtavenar/tslearn) - Time series clustering and classification, `TimeSeriesKMeans`, `TimeSeriesKMeans`.  
[pastas](https://github.com/pastas/pastas) - Analysis of Groundwater Time Series.  
[fastdtw](https://github.com/slaypni/fastdtw) - Dynamic Time Warp Distance.  
[fable](https://www.rdocumentation.org/packages/fable/versions/0.0.0.9000) - Time Series Forecasting (R package).  
[pydlm](https://github.com/wwrechard/pydlm) - Bayesian time series modelling ([R package](https://cran.r-project.org/web/packages/bsts/index.html), [Blog post](http://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html))  
[PyAF](https://github.com/antoinecarme/pyaf) - Automatic Time Series Forecasting.  
[luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library from Linkedin.  
[matrixprofile-ts](https://github.com/target/matrixprofile-ts) - Detecting patterns and anomalies, [website](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html), [ppt](https://www.cs.ucr.edu/~eamonn/Matrix_Profile_Tutorial_Part1.pdf), [alternative](https://github.com/matrix-profile-foundation/mass-ts).  
[stumpy](https://github.com/TDAmeritrade/stumpy) - Another matrix profile library.  
[obspy](https://github.com/obspy/obspy) - Seismology package. Useful `classic_sta_lta` function.  
[RobustSTL](https://github.com/LeeDoYup/RobustSTL) - Robust Seasonal-Trend Decomposition.  
[seglearn](https://github.com/dmbee/seglearn) - Time Series library.  
[pyts](https://github.com/johannfaouzi/pyts) - Time series transformation and classification, [Imaging time series](https://pyts.readthedocs.io/en/latest/auto_examples/index.html#imaging-time-series).  
Turn time series into images and use Neural Nets: [example](https://gist.github.com/oguiza/c9c373aec07b96047d1ba484f23b7b47), [example](https://github.com/kiss90/time-series-classification).  
[sktime](https://github.com/alan-turing-institute/sktime), [sktime-dl](https://github.com/uea-machine-learning/sktime-dl) - Toolbox for (deep) learning with time series.   
[adtk](https://github.com/arundo/adtk) - Time Series Anomaly Detection.  
[rocket](https://github.com/angus924/rocket) - Time Series classification using random convolutional kernels.  
[luminaire](https://github.com/zillow/luminaire) - Anomaly Detection for time series.  
[etna](https://github.com/tinkoff-ai/etna) - Time Series library.  
[Chaos Genius](https://github.com/chaos-genius/chaos_genius) - ML powered analytics engine for outlier/anomaly detection and root cause analysis.  

##### Time Series Evaluation
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) - Sklearn time series split.  
[tscv](https://github.com/WenjieZ/TSCV) - Evaluation with gap.  

#### Financial Data and Trading
Tutorial on using cvxpy: [1](https://calmcode.io/cvxpy-one/the-stigler-diet.html), [2](https://calmcode.io/cvxpy-two/introduction.html)  
[pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/whatsnew.html) - Read stock data.  
[yfinance](https://github.com/ranaroussi/yfinance) - Read stock data from Yahoo Finance.  
[findatapy](https://github.com/cuemacro/findatapy) - Read stock data from various sources.  
[ta](https://github.com/bukosabino/ta) - Technical analysis library.  
[backtrader](https://github.com/mementum/backtrader) - Backtesting for trading strategies.  
[surpriver](https://github.com/tradytics/surpriver) - Find high moving stocks before they move using anomaly detection and machine learning.  
[ffn](https://github.com/pmorissette/ffn) - Financial functions.  
[bt](https://github.com/pmorissette/bt) - Backtesting algorithms.  
[alpaca-trade-api-python](https://github.com/alpacahq/alpaca-trade-api-python) - Commission-free trading through API.  
[eiten](https://github.com/tradytics/eiten) - Eigen portfolios, minimum variance portfolios and other algorithmic investing strategies.  
[tf-quant-finance](https://github.com/google/tf-quant-finance) - Quantitative finance tools in TensorFlow, by Google.  
[quantstats](https://github.com/ranaroussi/quantstats) - Portfolio management.  
[Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) - Portfolio optimization and strategic asset allocation.  
[OpenBBTerminal](https://github.com/OpenBB-finance/OpenBBTerminal) - Terminal.  
[mplfinance](https://github.com/matplotlib/mplfinance) - Financial markets data visualization.  

##### Quantopian Stack
[pyfolio](https://github.com/quantopian/pyfolio) - Portfolio and risk analytics.  
[zipline](https://github.com/quantopian/zipline) - Algorithmic trading.  
[alphalens](https://github.com/quantopian/alphalens) - Performance analysis of predictive stock factors.  
[empyrical](https://github.com/quantopian/empyrical) - Financial risk metrics.  
[trading_calendars](https://github.com/quantopian/trading_calendars) - Calendars for various securities exchanges.  

#### Survival Analysis
[Time-dependent Cox Model in R](https://stats.stackexchange.com/questions/101353/cox-regression-with-time-varying-covariates).  
[lifelines](https://lifelines.readthedocs.io/en/latest/) - Survival analysis, Cox PH Regression, [talk](https://www.youtube.com/watch?v=aKZQUaNHYb0), [talk2](https://www.youtube.com/watch?v=fli-yE5grtY).  
[scikit-survival](https://github.com/sebp/scikit-survival) - Survival analysis.  
[xgboost](https://github.com/dmlc/xgboost) - `"objective": "survival:cox"` [NHANES example](https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html)  
[survivalstan](https://github.com/hammerlab/survivalstan) - Survival analysis, [intro](http://www.hammerlab.org/2017/06/26/introducing-survivalstan/).  
[convoys](https://github.com/better/convoys) - Analyze time lagged conversions.  
RandomSurvivalForests (R packages: randomForestSRC, ggRandomForests).  
[pysurvival](https://github.com/square/pysurvival) - Survival analysis.  
[DeepSurvivalMachines](https://github.com/autonlab/DeepSurvivalMachines) - Fully Parametric Survival Regression.  
[auton-survival](https://github.com/autonlab/auton-survival) - Regression, Counterfactual Estimation, Evaluation and Phenotyping with Censored Time-to-Events.  

#### Outlier Detection & Anomaly Detection
[sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html) - Isolation Forest and others.  
[pyod](https://pyod.readthedocs.io/en/latest/pyod.html) - Outlier Detection / Anomaly Detection.  
[eif](https://github.com/sahandha/eif) - Extended Isolation Forest.  
[AnomalyDetection](https://github.com/twitter/AnomalyDetection) - Anomaly detection (R package).  
[luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library from Linkedin.  
Distances for comparing histograms and detecting outliers - [Talk](https://www.youtube.com/watch?v=U7xdiGc7IRU): [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html), [Wasserstein](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html), [Energy Distance (Cramer)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html), [Kullback-Leibler divergence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html).  
[banpei](https://github.com/tsurubee/banpei) - Anomaly detection library based on singular spectrum transformation.  
[telemanom](https://github.com/khundman/telemanom) - Detect anomalies in multivariate time series data using LSTMs.  
[luminaire](https://github.com/zillow/luminaire) - Anomaly Detection for time series.  
[rrcf](https://github.com/kLabUM/rrcf) - Robust Random Cut Forest algorithm for anomaly detection on streams.  

#### Concept Drift & Domain Shift
[TorchDrift](https://github.com/TorchDrift/TorchDrift) - Drift Detection for PyTorch Models.  
[alibi-detect](https://github.com/SeldonIO/alibi-detect) - Algorithms for outlier, adversarial and drift detection.  
[evidently](https://github.com/evidentlyai/evidently) - Evaluate and monitor ML models from validation to production.  
[Lipton et al. - Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916).  
[Bu et al. - A pdf-Free Change Detection Test Based on Density Difference Estimation](https://ieeexplore.ieee.org/document/7745962).  

#### Ranking
[lightning](https://github.com/scikit-learn-contrib/lightning) - Large-scale linear classification, regression and ranking.  

#### Causal Inference
[CS 594 Causal Inference and Learning](https://www.cs.uic.edu/~elena/courses/fall19/cs594cil.html)  
[Statistical Rethinking](https://github.com/rmcelreath/stat_rethinking_2022) - Video Lecture Series, Bayesian Statistics, Causal Models, [R](https://bookdown.org/content/4857/), [python](https://github.com/pymc-devs/resources/tree/master/Rethinking_2), [numpyro1](https://github.com/asuagar/statrethink-course-numpyro-2019), [numpyro2](https://fehiepsi.github.io/rethinking-numpyro/), [tensorflow-probability](https://github.com/ksachdeva/rethinking-tensorflow-probability).  
[Python Causality Handbook](https://github.com/matheusfacure/python-causality-handbook)  
[dowhy](https://github.com/py-why/dowhy) - Estimate causal effects.  
[CausalImpact](https://github.com/tcassou/causal_impact) - Causal Impact Analysis ([R package](https://google.github.io/CausalImpact/CausalImpact.html)).  
[causallib](https://github.com/IBM/causallib) - Modular causal inference analysis and model evaluations by IBM, [examples](https://github.com/IBM/causallib/tree/master/examples).  
[causalml](https://github.com/uber/causalml) - Causal inference by Uber.  
[upliftml](https://github.com/bookingcom/upliftml) - Causal inference by Booking.com.  
[EconML](https://github.com/microsoft/EconML) - Heterogeneous Treatment Effects Estimation by Microsoft.  
[causality](https://github.com/akelleh/causality) - Causal analysis using observational datasets.  
[DoubleML](https://github.com/DoubleML/doubleml-for-py) - Machine Learning + Causal inference, [Tweet](https://twitter.com/ChristophMolnar/status/1574338002305880068), [Presentation](https://scholar.princeton.edu/sites/default/files/bstewart/files/felton.chern_.slides.20190318.pdf), [Paper](https://arxiv.org/abs/1608.00060v1).  

##### Papers
[Bours - Confounding](https://edisciplinas.usp.br/pluginfile.php/5625667/mod_resource/content/3/Nontechnicalexplanation-counterfactualdefinition-confounding.pdf)  
[Bours - Effect Modification and Interaction](https://www.sciencedirect.com/science/article/pii/S0895435621000330)  

#### Probabilistic Modelling and Bayes
[Intro](https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html), [Guide](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)  
[PyMC3](https://www.pymc.io/projects/docs/en/stable/learn.html) - Bayesian modelling.  
[numpyro](https://github.com/pyro-ppl/numpyro) - Probabilistic programming with numpy, built on [pyro](https://github.com/pyro-ppl/pyro).  
[pomegranate](https://github.com/jmschrei/pomegranate) - Probabilistic modelling, [talk](https://www.youtube.com/watch?v=dE5j6NW-Kzg).  
[pmlearn](https://github.com/pymc-learn/pymc-learn) - Probabilistic machine learning.  
[arviz](https://github.com/arviz-devs/arviz) - Exploratory analysis of Bayesian models.  
[zhusuan](https://github.com/thu-ml/zhusuan) - Bayesian deep learning, generative models.  
[edward](https://github.com/blei-lab/edward) - Probabilistic modelling, inference, and criticism, [Mixture Density Networks (MNDs)](http://edwardlib.org/tutorials/mixture-density-network), [MDN Explanation](https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca).  
[Pyro](https://github.com/pyro-ppl/pyro) - Deep Universal Probabilistic Programming.  
[TensorFlow probability](https://github.com/tensorflow/probability) - Deep learning and probabilistic modelling, [talk1](https://www.youtube.com/watch?v=KJxmC5GCWe4), [notebook talk1](https://github.com/AlxndrMlk/PyDataGlobal2021/blob/main/00_PyData_Global_2021_nb_full.ipynb), [talk2](https://www.youtube.com/watch?v=BrwKURU-wpk), [example](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb).  
[bambi](https://github.com/bambinos/bambi) - High-level Bayesian model-building interface on top of PyMC3.  
[neural-tangents](https://github.com/google/neural-tangents) - Infinite Neural Networks.  
[bnlearn](https://github.com/erdogant/bnlearn) - Bayesian networks, parameter learning, inference and sampling methods.  

#### Gaussian Processes
[Visualization](http://www.infinitecuriosity.org/vizgp/), [Article](https://distill.pub/2019/visual-exploration-gaussian-processes/)  
[GPyOpt](https://github.com/SheffieldML/GPyOpt) - Gaussian process optimization.   
[GPflow](https://github.com/GPflow/GPflow) - Gaussian processes (TensorFlow).  
[gpytorch](https://gpytorch.ai/) - Gaussian processes (PyTorch).  

#### Stacking Models and Ensembles
[Model Stacking Blog Post](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)  
[mlxtend](https://github.com/rasbt/mlxtend) - `EnsembleVoteClassifier`, `StackingRegressor`, `StackingCVRegressor` for model stacking.  
[vecstack](https://github.com/vecxoz/vecstack) - Stacking ML models.  
[StackNet](https://github.com/kaz-Anova/StackNet) - Stacking ML models.  
[mlens](https://github.com/flennerhag/mlens) - Ensemble learning.  
[combo](https://github.com/yzhao062/combo) - Combining ML models (stacking, ensembling).  

#### Model Evaluation
[evaluate](https://github.com/huggingface/evaluate) - Evaluate machine learning models (huggingface).  
[pycm](https://github.com/sepandhaghighi/pycm) - Multi-class confusion matrix.  
[pandas_ml](https://github.com/pandas-ml/pandas-ml) - Confusion matrix.  
Plotting learning curve: [link](http://www.ritchieng.com/machinelearning-learning-curve/).  
[yellowbrick](http://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html) - Learning curve.  
[pyroc](https://github.com/noudald/pyroc) - Receiver Operating Characteristic (ROC) curves.  

#### Model Uncertainty
[awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction) - Uncertainty quantification.  
[uncertainty-toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox) - Predictive uncertainty quantification, calibration, metrics, and visualization.  

#### Model Explanation, Interpretability, Feature Importance
[Princeton - Reproducibility Crisis in ML‑based Science](https://sites.google.com/princeton.edu/rep-workshop)   
[Book](https://christophm.github.io/interpretable-ml-book/agnostic.html), [Examples](https://github.com/jphall663/interpretable_machine_learning_with_python)  
scikit-learn - [Permutation Importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) (can be used on any trained classifier) and [Partial Dependence](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html)  
[shap](https://github.com/slundberg/shap) - Explain predictions of machine learning models, [talk](https://www.youtube.com/watch?v=C80SQe16Rao), [Good Shap intro](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/).  
[treeinterpreter](https://github.com/andosa/treeinterpreter) - Interpreting scikit-learn's decision tree and random forest predictions.  
[lime](https://github.com/marcotcr/lime) - Explaining the predictions of any machine learning classifier, [talk](https://www.youtube.com/watch?v=C80SQe16Rao), [Warning (Myth 7)](https://crazyoscarchang.github.io/2019/02/16/seven-myths-in-machine-learning-research/).  
[lime_xgboost](https://github.com/jphall663/lime_xgboost) - Create LIMEs for XGBoost.  
[eli5](https://github.com/TeamHG-Memex/eli5) - Inspecting machine learning classifiers and explaining their predictions.  
[lofo-importance](https://github.com/aerdem4/lofo-importance) - Leave One Feature Out Importance, [talk](https://www.youtube.com/watch?v=zqsQ2ojj7sE).  
[pybreakdown](https://github.com/MI2DataLab/pyBreakDown) - Generate feature contribution plots.  
[pycebox](https://github.com/AustinRochford/PyCEbox) - Individual Conditional Expectation Plot Toolbox.  
[pdpbox](https://github.com/SauceCat/PDPbox) - Partial dependence plot toolbox, [example](https://www.kaggle.com/dansbecker/partial-plots).  
[partial_dependence](https://github.com/nyuvis/partial_dependence) - Visualize and cluster partial dependence.  
[contrastive_explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) - Contrastive explanations.  
[DrWhy](https://github.com/ModelOriented/DrWhy) - Collection of tools for explainable AI.  
[lucid](https://github.com/tensorflow/lucid) - Neural network interpretability.  
[xai](https://github.com/EthicalML/XAI) - An eXplainability toolbox for machine learning.  
[innvestigate](https://github.com/albermax/innvestigate) - A toolbox to investigate neural network predictions.  
[dalex](https://github.com/pbiecek/DALEX) - Explanations for ML models (R package).  
[interpretml](https://github.com/interpretml/interpret) - Fit interpretable models, explain models.  
[shapash](https://github.com/MAIF/shapash) - Model interpretability.  
[imodels](https://github.com/csinva/imodels) - Interpretable ML package.  
[captum](https://github.com/pytorch/captum) - Model interpretability and understanding for PyTorch.  

#### Automated Machine Learning
[AdaNet](https://github.com/tensorflow/adanet) - Automated machine learning based on TensorFlow.  
[tpot](https://github.com/EpistasisLab/tpot) - Automated machine learning tool, optimizes machine learning pipelines.  
[autokeras](https://github.com/jhfjhfj1/autokeras) - AutoML for deep learning.  
[nni](https://github.com/Microsoft/nni) - Toolkit for neural architecture search and hyper-parameter tuning by Microsoft.  
[mljar](https://github.com/mljar/mljar-supervised) - Automated machine learning.  
[automl_zero](https://github.com/google-research/google-research/tree/master/automl_zero) - Automatically discover computer programs that can solve machine learning tasks from Google.  
[AlphaPy](https://github.com/ScottfreeLLC/AlphaPy) - Automated Machine Learning using scikit-learn xgboost, LightGBM and others.  

#### Graph Representation Learning
[Karate Club](https://github.com/benedekrozemberczki/karateclub) - Unsupervised learning on graphs.   
[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) - Graph representation learning with PyTorch.   
[DLG](https://github.com/dmlc/dgl) - Graph representation learning with TensorFlow.   

#### Convex optimization
[cvxpy](https://github.com/cvxgrp/cvxpy) - Modelling language for convex optimization problems. Tutorial: [1](https://calmcode.io/cvxpy-one/the-stigler-diet.html), [2](https://calmcode.io/cvxpy-two/introduction.html)  

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
[evotorch](https://github.com/nnaisense/evotorch) - Evolutionary computation library built on Pytorch.  

#### Hyperparameter Tuning
[sklearn](https://scikit-learn.org/stable/index.html) - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).  
[sklearn-deap](https://github.com/rsteca/sklearn-deap) - Hyperparameter search using genetic algorithms.  
[hyperopt](https://github.com/hyperopt/hyperopt) - Hyperparameter optimization.  
[hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) - Hyperopt + sklearn.  
[optuna](https://github.com/pfnet/optuna) - Hyperparamter optimization, [Talk](https://www.youtube.com/watch?v=tcrcLRopTX0).  
[skopt](https://scikit-optimize.github.io/) - `BayesSearchCV` for Hyperparameter search.  
[tune](https://ray.readthedocs.io/en/latest/tune.html) - Hyperparameter search with a focus on deep learning and deep reinforcement learning.  
[bbopt](https://github.com/evhub/bbopt) - Black box hyperparameter optimization.  
[dragonfly](https://github.com/dragonfly/dragonfly) - Scalable Bayesian optimisation.  
[botorch](https://github.com/pytorch/botorch) - Bayesian optimization in PyTorch.  
[ax](https://github.com/facebook/Ax) - Adaptive Experimentation Platform by Facebook.  
[lightning-hpo](https://github.com/Lightning-AI/lightning-hpo) - Hyperparameter optimization based on optuna.  

#### Incremental Learning, Online Learning
sklearn - [PassiveAggressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html), [PassiveAggressiveRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html).  
[river](https://github.com/online-ml/river) - Online machine learning.  
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

#### Deployment and Lifecycle Management

##### Workflow Scheduling and Orchestration
[nextflow](https://github.com/goodwright/nextflow.py) - Run scripts and workflow graphs in Docker image using Google Life Sciences, AWS Batch, [Website](https://github.com/nextflow-io/nextflow).   
[airflow](https://github.com/apache/airflow) - Schedule and monitor workflows.  
[prefect](https://github.com/PrefectHQ/prefect) - Python specific workflow scheduling.  
[dagster](https://github.com/dagster-io/dagster) - Development, production and observation of data assets.  
[ploomber](https://github.com/ploomber/ploomber) - Workflow orchestration.  
[kestra](https://github.com/kestra-io/kestra) - Workflow orchestration.  
[cml](https://github.com/iterative/cml) - CI/CD for Machine Learning Projects.  
[rocketry](https://github.com/Miksus/rocketry) - Task scheduling.  
[huey](https://github.com/coleifer/huey) - Task queue.  

##### Containerization and Docker
[Reduce size of docker images (video)](https://www.youtube.com/watch?v=Z1Al4I4Os_A)  
[Optimize Docker Image Size](https://www.augmentedmind.de/2022/02/06/optimize-docker-image-size/)  
[cog](https://github.com/replicate/cog) - Facilitates building Docker images.  

##### Data Versioning, Databases, Pipelines and Model Serving
[dvc](https://github.com/iterative/dvc) - Version control for large files.  
[kedro](https://github.com/quantumblacklabs/kedro) - Build data pipelines.  
[feast](https://github.com/feast-dev/feast) - Feature store. [Video](https://www.youtube.com/watch?v=_omcXenypmo).  
[pinecone](https://www.pinecone.io/) - Database for vector search applications.  
[truss](https://github.com/basetenlabs/truss) - Serve ML models.  
[milvus](https://github.com/milvus-io/milvus) - Vector database for similarity search.  
[mlem](https://github.com/iterative/mlem) - Version and deploy your ML models following GitOps principles.  

##### Data Science Related
[m2cgen](https://github.com/BayesWitnesses/m2cgen) - Transpile trained ML models into other languages.  
[sklearn-porter](https://github.com/nok/sklearn-porter) - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.  
[mlflow](https://mlflow.org/) - Manage the machine learning lifecycle, including experimentation, reproducibility and deployment.  
[skll](https://github.com/EducationalTestingService/skll) - Command-line utilities to make it easier to run machine learning experiments.  
[BentoML](https://github.com/bentoml/BentoML) - Package and deploy machine learning models for serving in production.  
[dagster](https://github.com/dagster-io/dagster) - Tool with focus on dependency graphs.  
[knockknock](https://github.com/huggingface/knockknock) - Be notified when your training ends.  
[metaflow](https://github.com/Netflix/metaflow) - Lifecycle Management Tool by Netflix.  
[cortex](https://github.com/cortexlabs/cortex) - Deploy machine learning models.  
[Neptune](https://neptune.ai) - Experiment tracking and model registry.  
[clearml](https://github.com/allegroai/clearml) - Experiment Manager, MLOps and Data-Management.  
[polyaxon](https://github.com/polyaxon/polyaxon) - MLOps.  
[sematic](https://github.com/sematic-ai/sematic) - Deploy machine learning models.  
[zenml](https://github.com/zenml-io/zenml) - MLOPs.  

#### Math and Background
[All kinds of math and statistics resources](https://realnotcomplex.com/)  
Gilbert Strang - [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/index.htm)  
Gilbert Strang - [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning
](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)  

#### Resources
[Distill.pub](https://distill.pub/) - Blog.   
[Machine Learning Videos](https://github.com/dustinvtran/ml-videos)  
[Data Science Notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)  
[Recommender Systems (Microsoft)](https://github.com/Microsoft/Recommenders)  
[Datascience Cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets)   

##### Guidelines 
[datasharing](https://github.com/jtleek/datasharing) - Guide to data sharing.  

##### Books
[Blum - Foundations of Data Science](https://www.cs.cornell.edu/jeh/book.pdf?file=book.pdf)  
[Chan - Introduction to Probability for Data Science](https://probability4datascience.com/index.html)  
[Colonescu - Principles of Econometrics with R](https://bookdown.org/ccolonescu/RPoE4/)  

##### Other Awesome Lists
[Awesome Adversarial Machine Learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning)    
[Awesome AI Booksmarks](https://github.com/goodrahstar/my-awesome-AI-bookmarks)    
[Awesome AI on Kubernetes](https://github.com/CognonicLabs/awesome-AI-kubernetes)    
[Awesome Big Data](https://github.com/onurakpolat/awesome-bigdata)    
[Awesome Business Machine Learning](https://github.com/firmai/business-machine-learning)    
[Awesome Causality](https://github.com/rguo12/awesome-causality-algorithms)    
[Awesome Community Detection](https://github.com/benedekrozemberczki/awesome-community-detection)    
[Awesome CSV](https://github.com/secretGeek/AwesomeCSV)  
[Awesome Cytodata](https://github.com/cytodata/awesome-cytodata)  
[Awesome Data Science with Ruby](https://github.com/arbox/data-science-with-ruby)   
[Awesome Dash](https://github.com/ucg8j/awesome-dash)   
[Awesome Decision Trees](https://github.com/benedekrozemberczki/awesome-decision-tree-papers)    
[Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)   
[Awesome ETL](https://github.com/pawl/awesome-etl)   
[Awesome Financial Machine Learning](https://github.com/firmai/financial-machine-learning)   
[Awesome Fraud Detection](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)   
[Awesome GAN Applications](https://github.com/nashory/gans-awesome-applications)   
[Awesome Graph Classification](https://github.com/benedekrozemberczki/awesome-graph-classification)   
[Awesome Industry Machine Learning](https://github.com/firmai/industry-machine-learning)  
[Awesome Gradient Boosting](https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers)   
[Awesome Learning with Label Noise](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise)  
[Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning#python)    
[Awesome Machine Learning Books](http://matpalm.com/blog/cool_machine_learning_books/)  
[Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)     
[Awesome Machine Learning Operations](https://github.com/EthicalML/awesome-machine-learning-operations)   
[Awesome Monte Carlo Tree Search](https://github.com/benedekrozemberczki/awesome-monte-carlo-tree-search-papers)   
[Awesome Neural Network Visualization](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)  
[Awesome Online Machine Learning](https://github.com/MaxHalford/awesome-online-machine-learning)  
[Awesome Pipeline](https://github.com/pditommaso/awesome-pipeline)  
[Awesome Public APIs](https://github.com/public-apis/public-apis)  
[Awesome Python](https://github.com/vinta/awesome-python)   
[Awesome Python Data Science](https://github.com/krzjoa/awesome-python-datascience)   
[Awesome Python Data Science](https://github.com/thomasjpfan/awesome-python-data-science)  
[Awesome Pytorch](https://github.com/bharathgs/Awesome-pytorch-list)  
[Awesome Quantitative Finance](https://github.com/wilsonfreitas/awesome-quant)  
[Awesome Recommender Systems](https://github.com/grahamjenson/list_of_recommender_systems)  
[Awesome Single Cell](https://github.com/seandavi/awesome-single-cell)  
[Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)  
[Awesome Sentence Embedding](https://github.com/Separius/awesome-sentence-embedding)  
[Awesome Time Series](https://github.com/MaxBenChrist/awesome_time_series_in_python)  
[Awesome Time Series Anomaly Detection](https://github.com/rob-med/awesome-TS-anomaly-detection)  
[Awesome Visual Attentions](https://github.com/MenghaoGuo/Awesome-Vision-Attentions)  
[Awesome Visual Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer)  

#### Lectures
[NYU Deep Learning SP21](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) - YouTube Playlist.   

#### Things I google a lot
[Color Codes](https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#categorical-colors)  
[Frequency codes for time series](https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)  
[Date parsing codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)  

## Contributing  
Do you know a package that should be on this list? Did you spot a package that is no longer maintained and should be removed from this list? Then feel free to read the [contribution guidelines](CONTRIBUTING.md) and submit your pull request or create a new issue.  

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
