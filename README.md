# PsiZ: A Psychological Embedding Package

NOTE: This README is in the process of being updated for release 0.4.0. Not all information is accurate.

## What's in a name?
The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Purpose

PsiZ provides the computational tools to infer a continuous, multivariate stimulus representation using similarity relations. It integrates well-established cognitive theory with contemporary computational methods.

## Installation

There is not yet a stable version (nor an official release of this library). All APIs are subject to change and all releases are alpha.

To install the latest development version, clone from GitHub and instal the local repo using pip.
1. Use `git` to clone the latest version to your local machine: `git clone https://github.com/roads/psiz.git`
2. Use `pip` to install the cloned repo (using editable mode): `pip install -e /local/path/to/psiz`
By using editable mode, you can easily update your local copy by use `git pull origin master` inside your local copy of the repo. You do not have to re-install with `pip`.

The package can also be obtained by:
* Manually downloading the latest version at https://github.com/roads/psiz.git
* Use git to clone a specific release, for example: `git clone https://github.com/roads/psiz.git --branch v0.3.0`
* Using PyPi to install older alpha releases: ``pip install psiz``. The versions available through PyPI lag behind the latest GitHub version.

**Note:** PsiZ also requires TensorFlow. In older versions of TensorFlow, CPU only versions were targeted separately. For Tensorflow >=2.0, both CPU-only and GPU versions are obtained via `tensorflow`. The current `setup.py` file fulfills this dependency by downloading the `tensorflow` package using `pip`.

## Quick Start

Similarity relations can be collected using a variety of paradigms. You will need to use the appropriate model for your data. In addition to a model choice, you need to provide two additional pieces of information:

1. The observed similarity relations (referred to as observations or *obs*).
2. The number of unique stimuli that will be in your embedding (`n_stimuli`).


The following minimalist example uses a `Rank` psychological embedding to model a predefined set of ordinal similarity relations.
```python
import psiz

# Load observations from a predefined dataset.
(obs, catalog) = psiz.datasets.load('birds-16')
# Create a TensorFlow embedding layer.
# NOTE: Since we will use masking, we increment n_stimuli by one.
embedding = tf.keras.layers.Embedding(
    catalog.n_stimuli+1, mask_zero=True
)
# Create a Rank model that uses the TensorFlow Keras API.
model = psiz.models.Rank(embedding=embedding)
# Wrap the model in convenient proxy class.
emb = psiz.models.Proxy(model)
# Compile the model.
emb.compile()
# Fit the psychological embedding using observations.
emb.fit(obs)
# Optionally save the fitted model in HDF5 format.
emb.save('my_embedding.h5')
```

## Trials and Observations
Inference is performed by fitting a model to a set of observations. In this package, a single observation is comprised of trial where multiple stimuli that have been judged by an agent (human or machine) based on their similarity. There are currently three different types of trials: *rank*, *rate* and *sort*.

### Rank

In the simplest case, an observation is obtained from a trial consisting of three stimuli: a *query* stimulus (*q*) and two *reference* stimuli (*a* and *b*). An agent selects the reference stimulus that they believe is more similar to the query stimulus. For this simple trial, there are two possible outcomes. If the agent selected reference *a*, then the observation for the *i*th trial would be recorded as the vector: 

D<sub>*i*</sub> = [*q* *a* *b*]

Alternatively, if the agent had selected reference *b*, the observation would be recorded as:

D<sub>*i*</sub> = [*q* *b* *a*]

In addition to a simple *triplet* rank trial, this package is designed to handle a number of different rank trial configurations. A trial may have 2-8 reference stimuli and an agent may be required to select and rank more than one reference stimulus. A companion Open Access article dealing with rank trials is available at https://link.springer.com/article/10.3758/s13428-019-01285-3.

### Rate

In the simplest case, an observation is obtained from a trial consisting of two stimuli. An agent provides a numerical rating regarding the similarity between the stimuli. *This functionality is not currently available and is under development.*

### Sort

In the simplest case, an observation is obtained from a trial consisting of three stimuli. Ag agent sorts the stimuli into two groups based on similarity. *This functionality is not currently available and is under development.*

## Using Your Own Data

To use your own data, you should place your data in an appropriate subclass of `psiz.trials.Observations`. Once the `Observations` object has been created, you can save it to disk by calling its `save` method. It can be loaded later using the function `psiz.trials.load(filepath)`. Consider the following example that creates random rank observations:

```python
import numpy as np
import psiz

# Let's assume that we have 10 unique stimuli.
stimuli_list = np.arange(0, 10, dtype=int)

# Let's create 100 trials, where each trial is composed of a query and
# four references. We will also assume that participants selected two
# references (in order of their similarity to the query.)
n_trial = 100
n_reference = 4
stimulus_set = np.empty([n_trial, n_reference + 1], dtype=int)
n_select = 2 * np.ones((n_trial), dtype=int)
for i_trial in range(n_trial):
    # Randomly selected stimuli and randomly simulate behavior for each
    # trial (one query, four references).
    stimulus_set[i_trial, :] = np.random.choice(
        stimuli_list, n_reference + 1, replace=False
    )

# Create the observations object and save it to disk.
obs = psiz.trials.RankObservations(stimulus_set, n_select=n_select)
obs.save('path/to/obs.hdf5')

# Load the observations from disk.
obs = psiz.trials.load('path/to/obs.hdf5')
```
Note that the values in `stimulus_set` are assumed to be contiguous integers [0, N[, where N is the number of unique stimuli. Their order is also important. The query is listed in the first column, an agent's selected references are listed second (in order of selection if there are more than two) and then any remaining unselected references are listed (in any order).

## Design Philosophy

PsiZ is built around the TensorFlow ecosystem and strives to follow TensorFlow idioms as closely as possible.

### Model, Layer, Variable

Package-defined models are built by sub-classing `tf.keras.Model`. Components of a model are built using the `tf.keras.layers.Layer` API. Free parameters are implemented as a `tf.Variable`.

In PsiZ, a psychological embedding can be thought of as having two major components. The first component is a conventional embedding which models stimulus coordinates in psychological space. In the simplest case, this is implemented using `tf.keras.layers.Embedding`. The second component embodies the *psychological* aspect of the model and includes parameterized distance, similarity, and choice functions.

PsiZ includes a number of predefined layers to facilitate the construction of new Models. For example, there are four predefined similarity functions (implemented as subclasses of `tf.keras.layers.Layer`):

1. `psiz.keras.layers.InverseSimilarity`
2. `psiz.keras.layers.ExponentialSimilarity`
3. `psiz.keras.layers.HeavyTailedSimilarity`
4. `psiz.keras.layers.StudentsTSimilarity`

Each similarity function has its own set of parameters (i.e., `tf.Variable`s). The `ExponentialSimilarity`, which is widely used in psychology, has four variables. Users can also implement there own similarity functions by sub-classing `tf.keras.layers.Layers`. See [a relative link](CONTRIBUTING.md) for more guidance.

### Deviations from TensorFlow
<!-- TODO -->
* restarts
* multiple possible trial configurations
* likelihood loss (minor deviation)

### Compile and Fit
<!-- TODO -->
Just like a typical TensorFlow model, all trainable variables are optimized to minimize loss. 

## Common Use Cases

### Selecting the Dimensionality.
<!-- TODO -->
By default, an embedding will be inferred using two dimensions. Typically you will want to set the dimensionality to something else. This can easily be done using the keyword `n_dim` during embedding initialization. The dimensionality can also be determined using a cross-validation procedure `psiz.dimensionality.search`.
```python
n_stimuli = 100

model_spec = {
    'model': psiz.models.Exponential,
    'n_stimuli': n_stimuli,
    'n_group': 1,
    'modifier': None
}

search_summary = psiz.dimensionality.search(obs, model_spec)
n_dim = search_summary['dim_best']

emb = psiz.models.Exponential(n_stimuli, n_dim=4)
emb.fit(obs)
```

### Multiple Groups
By default, the embedding will be inferred assuming that all observations were obtained from a single population or group. If you would like to infer an embedding with multiple group-level parameters, use the `n_group` keyword during embedding initialization. When you create your `psiz.trials.observations` object you will also need to set the `group_id` attribute to indicate the group-membership of each trial.
```python
n_stimuli = 100
emb = psiz.models.Exponential(n_stimuli, n_dim=4, n_group=2)
emb.fit(obs)
```

### Fixing Free Parameters
If you know some of the free parameters already, you can set them to the desired value and then make those parameters untrainable.
```python
n_stimuli = 100
emb = psiz.models.Exponential(n_stimuli, n_dim=2)
emb.rho = 2.0
emb.tau = 1.0
emb.trainable({'rho': False, 'tau': False})
emb.fit(obs)
```

If you are also performing a dimensionality search, these constraints can be passed in using a post-initialization `modifier` function.
```python
n_stimuli = 100

def modifier_func(emb):
    emb.rho = 2.0
    emb.tau = 1.0
    emb.trainable({'rho': False, 'tau': False})
    return emb

model_spec = {
    'model': psiz.models.Exponential,
    'n_stimuli': n_stimuli,
    'n_group': 1,
    'modifier': modifier_func
}

search_summary = psiz.dimensionality.search(obs, model_spec)
n_dim = search_summary['dim_best']
```

## Modules
* `keras` - A module containing Keras related classes.
* `catalog` - Class for storing stimulus information.
* `datasets` - Functions for loading some pre-defined catalogs and observations.
* `dimensionality` - Routine for selecting the dimensionality of the embedding.
* `generator` - Generate new trials randomly or using active selection.
* `models` - A set of pre-defined psychological embedding models.
* `preprocess` - Functions for preprocessing observations.
* `restart` - Classes and functionality for performing model restarts.
* `simulate` - Simulate an agent making similarity judgments.
* `trials` - Classes and functions for creating and managing observations.
* `utils` - Utility functions.
* `visualize` - Functions for visualizing embeddings.

## Authors
* Brett D. Roads
* Michael C. Mozer
* See also the list of contributors who participated in this project.

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.

## Code of Conduct
This project uses a Code of Conduct [a relative link](CODE.md) adapted from the [Contributor Covenant][homepage], version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

## References
* van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet
   embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
   international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
* Roads, B. D., & Mozer, M. C. (2019). Obtaining psychological
   embeddings through joint kernel and metric learning. Behavior Research
   Methods. 51(5), 2180-2193. doi:10.3758/s13428-019-01285-3
* Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The
   Caltech-UCSD Birds-200-2011 Dataset (Tech. Rep. No. CNS-TR-2011-001).
   California Institute of Technology.
