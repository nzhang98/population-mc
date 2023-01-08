import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from collections import Counter
from utils import generate_dataset, GaussianPrior, GaussianLlhood

class PMC():
    def __init__(self, X == None, mus = None, baseline_threshold, parameters):
        self.X = X
        self.N = len(self.X)
        