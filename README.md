# Keras ngram embedding

Train a text embedding from scratch, using Keras! The text is broken down into pairs of words, and whether those 
words ever appear within a one-sided window of `n` words. 

For example, given the text `Alice was beginning to get very tired of sitting by her sister on the
bank`, and a window of 2 words, we might get samples:

[[`Alice`, `was`], True]
[[`Alice`, `beginning`], True]
[[`Alice`, `tired`], False]

## Quick start

To create an embedding, just do the following: 

```python
# Set up Anaconda virtual environment
conda env create -f environment.yml --force

# Activate Anaconda virtual environment
source activate thesaurus

# Run code
cd bin/
python main.py
```

## Repo structure

 - `bin/main.py`: Code entry point
 

### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use 
environment from file*, and *Change environments (activate/deactivate)*). 

## Contact
Feel free to contact me at 13herger `<at>` gmail `<dot>` com
