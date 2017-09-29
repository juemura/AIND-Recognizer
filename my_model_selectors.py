import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on BIC scores
        BIC = float("inf")
        best_n = self.n_constant
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n, n_iter=1000).fit(self.x, self.lengths)
                logL = model.score(self.X, self.lengths)
                temp = (-2)*logL + (n*n+2*n*model.n_features-1)*math.log(len(self.sequences))
                if BIC > temp:
                    BIC = temp
                    best_n = n
                break
            except:
                pass
        return GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdfs
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on DIC scores
        DIC = float("inf")
        best_n = self.n_constant
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n, n_iter=1000).fit(self.x, self.lengths)
                logL = model.score(self.X, self.lengths)
                competing_words = [w for w in self.words if w != self.this_word]
                competing_logLs = 0
                for w in competing_words:
                    x,l = self.hwords[w]
                    competing_logLs += model.score(x,l)
                temp = logL - competing_logLs / len(competing_words)
                if DIC > temp:
                    DIC = temp
                    best_n = n
                break
            except:
                pass
        return GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # DONE implement model selection using CV
        highest_logL_mean = float("-inf")
        best_n = self.n_constant
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                logL_arr = []
                split_method = KFold()
                indices = split_method.split(self.sequences)
                for cv_train_idx, cv_test_idx in indices:
                    x_train, len_train = combine_sequences(cv_train_idx, self.sequences)
                    model = GaussianHMM(n_components=n, n_iter=1000).fit(x_train, len_train)
                    x_test, len_test = combine_sequences(cv_test_idx, self.sequences)
                    logL = model.score(x_test, len_test)
                    logL_arr.append(logL)
                if highest_logL_mean < np.mean(logL_arr):
                    highest_logL_mean = np.mean(logL_arr)
                    best_n = n
                break
            except:
                pass
        return GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)


