"""
This module implements the semi-supervised training algorithm for Naive Bayes

The algorithm assumes binary features and two classes

Standard Naive Bayes formula:
  P(Feature_1|Class)*P(Feature_2|Class)*P(Class)

We use the Bernoulli distribution for modelling P(Features|Class):
  P(Class) * P(Feature_1|Class)^x * (1-P(Feature_1|Class))^(1-x) *
  P(Feature_2|Class)^x * (1-P(Feature_2|Class))^(1-x)

It uses the Bernoulli event model for sparse data as described in 
"Predictive modeling with Big Data: is Bigger really Better?"

Instead of initializing the first model based on the labeled data, it
initializes the first model using class distributions of certain features P(Class|Feature)

By using Bayes' rule we can compute P(Feature|Class) = P(Class|Feature)*P(Feature)/P(Class)

where we are given P(Class|Feature) and P(Class) and can compute P(Feature)

The algorithm implements expectation-maximization iterations. Using the initial model
it estimates the class probabilities of each sample P(Class|Sample) (the E-step)

In the M-step new feature coefficients are computed by multiplying each feature with samples' probability and normalizing by the sum of all sample class probabilities P(Class|Sample)

This is iterated many times until convergence of maximum iterations are reached

This is done in log space so we can use fast numpy matrix-vector multiplication
"""

import logging
import copy
import operator

from scipy.sparse import csr_matrix
import numpy as np

from sklearn import preprocessing

def smoothing(m, sum_class_posterior):
  return (m.sum(axis=0))/(sum_class_posterior)

def hstack(m1, m2):
  return np.hstack((m1, m2))

def apply(m, func):
  return func(m)

def _check_adjacency(m):
  row_sum = m.sum(axis=1)
  nonzero_rows = np.any(row_sum==0) #no row should be all zeros
  return (m.data!=1).sum()!=0 or nonzero_rows

class NaiveBayes():
  def __init__(self, 
      n_iter=10, 
      stop_criterion=0.1,
      keep_history=True):
    """
    Only for binary classification

    The model can either use sparse matrices or SparkMatrix

    :param n_iter: maximum number of EM-iterations
    :param stop_criterion: EM stops when difference is smaller than stop_criterion
    :param keep_history: keep feature likelihoods and posteriors for every iteration of EM
    
    """
    self.n_iter = n_iter
    self.stop_criterion = stop_criterion
    self.keep_history = keep_history

    if keep_history:
      self.history = {
        "feature_likelihoods": [],
        "feature_posteriors": [],
        "log_likelihood": []
      }

  def _initialize(self):
    """
    Initializes class_priors, intercept_, neg_intercept_,
    fixed_feature_indices_ and fixed_feature_likelihoods_

    To anchor/guide the unsupervised learning we fix the features
    for which we have P(Class|Feature) estimates
    fixed_feature_likelihoods_ saves the feature likelihoods P(Feature|Class)
    for those features for which we have P(Class|Feature)
    """
    with JsonReader(self.initial_parameters_file) as fin:
      self.initial_model_parameters_ = fin.read()

    #if not specified at initialization, try to get priors from initial model parameters,
    #otherwise set default class prior of 0.5
    if self.class_priors is None:
      self.class_priors = [0.5, 0.5]
      self.class_priors[1] = self.initial_model_parameters_.pop("prior", 0.5) 
      self.class_priors[0] = 1.0 - self.class_priors[1]
 
    #set intercepts based on class priors
    self.intercept_ = np.log(self.class_priors[1])
    self.neg_intercept_ = np.log(self.class_priors[0])

    #compute fixed feature likelihoods P(Feature|Class)
    self.fixed_feature_likelihoods_ = [[],[]]
    self.fixed_feature_indices_ = []
    for feature_name, feature_posterior in self.initial_model_parameters_.items():
      try:
        feature_index = self.colmeta2index[feature_name]
      except KeyError:
        logging.warning(f"feature {feature_name} not present in data")
        continue
      self.fixed_feature_indices_.append(feature_index)
      self.fixed_feature_likelihoods_[1].append(
        feature_posterior*self.feature_priors_[0, feature_index]/self.class_priors[1]
      )
      self.fixed_feature_likelihoods_[0].append(
        (1.0-feature_posterior)*self.feature_priors_[0, feature_index]/self.class_priors[0]
      )
   
    #set fixed part of linear model coefficients
    for i in [0,1]:
      self.fixed_feature_likelihoods_[i] = np.array(self.fixed_feature_likelihoods_[i])
      assert np.all(self.fixed_feature_likelihoods_[i]<=1)
    self.fixed_feature_indices_ = np.array(self.fixed_feature_indices_)
  
  def set_params(self, **parameters):
    for parameter, value in parameters.items():
        setattr(self, parameter, value)
    return self

  def _compute_coefficients(self):
    self.coef_ = np.log(self.parameters_) - np.log(1-self.parameters_)
    
    intercept_constants = np.sum(np.log(1-self.parameters_), axis=1).reshape((self.n_classes_,))
    self.intercept_ = np.log(self.intercept_parameters_) + intercept_constants

  def _bootstrap_model(self, X, y):
    """
    """
    self.classes_ = sorted(np.unique(y))
    self.n_classes_ = len(self.classes_)-1

    self.n_unlabeled_ = np.sum(y==-1)
    self.n_labeled_ = np.sum(y!=-1)

    self.parameters_ = np.zeros((self.n_classes_, X.shape[1]))
    self.intercept_parameters_ = np.zeros((self.n_classes_,))
    self.coef_ = np.zeros((self.n_classes_, X.shape[1]))
    self.intercept_ = np.zeros((self.n_classes_,))
    self.n_samples_ = np.zeros((self.n_classes_,))

    for i, class_label in enumerate(self.classes_[1:]):
      self.parameters_[i,:] = np.sum(X[y==class_label,:], axis=0)
      self.n_samples_[i] = np.sum(y==class_label)
      self.parameters_[i,:] /= self.n_samples_[i]

      self.intercept_parameters_[i] = self.n_samples_[i] / self.n_labeled_
    
    self._compute_coefficients()

  def _update_model(self, X, y, posterior_class_probabilities):
    
    for i, class_label in enumerate(self.classes_[1:]):
      self.parameters_[i,:] = ( np.sum(X[y==class_label,:], axis=0) + 
        np.sum(X[y==-1,:] * posterior_class_probabilities[y==-1,i], axis=0) )

      sum_probas = np.sum(posterior_class_probabilities[y==-1,i])
      self.parameters_[i,:] /= (self.n_samples_[i] + sum_probas)

      self.intercept_parameters_[i] = (
          (self.n_samples_[i] + sum_probas) / X.shape[0]
      )

    self._compute_coefficients()

  def _expected_log_joint_likelihood(self, X):
    """
    :returns float: the joint log likelihood (marginalized over the classes)
                    to check we are maximizing it in every iteration and to decide on stopping
    """

    pred = self.predict_unnormalized(X)
    sum_pred = pred.sum(axis=1)
    
    return apply(sum_pred, np.log).sum(axis=0).sum()

  def _em_iterations(self, X, y):
    """
    :param feature_matrix csr_matrix: sparse matrix where non-zero elements are 1
    """
    prev_log_likelihood = self._expected_log_joint_likelihood(X)
    if self.keep_history:
      self.history["log_likelihood"].append(prev_log_likelihood)
    
    logging.info(f"Initial likelihood {prev_log_likelihood}")
    
    for i in range(self.n_iter):
      posterior_class_probabilities = self.predict_proba(X)
      
      self._update_model(X, y, posterior_class_probabilities)
      
      log_likelihood = self._expected_log_joint_likelihood(X)
     
      if self.keep_history:
        self.history["log_likelihood"].append(log_likelihood)
      
      likelihood_difference = log_likelihood - prev_log_likelihood
      logging.info(f"""iteration {i}, log likelihood: {log_likelihood}, difference: {likelihood_difference}""")
      
      assert likelihood_difference >= 0
      if likelihood_difference < self.stop_criterion:
        break
      
      prev_log_likelihood = log_likelihood
    
    logging.info(f"Final likelihood {prev_log_likelihood}")
 
  def fit(self, X, y, **kwargs):
    """
    Fit the model according to the given training data.
    
    :param X: sparse matrix, shape (n_bottom_nodes, n_top_nodes)
        Training vector, where n_bottom_nodes is the number of bottom nodes and
        n_top_nodes is the number of top nodes.
    :param y: array-like, shape (n_bottom nodes, 1)
        Not used
    """
    if self.check_adjacency and _check_adjacency(X):
      raise ValueError("Input matrix X should only contain ones or zeros")

    self._bootstrap_model(X, y)
    self._em_iterations(X, y)

    return self
    
  def predict(self, X):
    return X * self.coef_.transpose() + self.intercept_.reshape((1, self.n_classes_))

  def predict_proba(self, X):
    """
    Probability estimates.
 
    :param X: matrix of shape = [n_samples, n_features]
    :returns: matrix of shape = [n_samples, 2] 
              where second column are the probabilities for second class (positive class)
    """
    if self.check_adjacency and _check_adjacency(X):
      raise ValueError("Input matrix X should only contain ones or zeros")

    pred = np.exp(self.predict(X))
    sum_pred = pred.sum(axis=1).reshape((X.shape[0],1))

    result = pred / sum_pred

    return result
