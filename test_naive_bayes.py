import pytest

from scipy.sparse import csr_matrix

from naive_bayes_em import *

@pytest.fixture(scope="session")
def data_matrix():
  return csr_matrix([
    [1,1],
    [1,0],
    [1,0],
    [0,1],
    [1,0], ####
    [1,1],
    [0,1],
    [0,1],
    [1,0],
    [0,1]
    ]
  )

@pytest.fixture(scope="session")
def labels():
  return np.array([
    0,
    0,
    0,
    0,
    -1,
    1,
    1,
    1,
    1,
    -1
  ])


@pytest.fixture(scope="session")
def model(data_matrix, labels):
  nb = NaiveBayes()
  nb._bootstrap_model(data_matrix, labels)
  return nb

class TestNaiveBayes():

  def test_bootstrap_model(self, model):
    expected_params = np.array([
      [3/4,0.5],
      [0.5,3/4]
    ])
    expected_bias_params = np.array([
      0.5, 0.5
    ])

    assert np.allclose(model.parameters_, expected_params)
    assert np.allclose(model.intercept_parameters_, expected_bias_params)

    expected_intercept = np.array([
      2*np.log(0.5)+np.log(1/4), 
      2*np.log(0.5)+np.log(1/4)
    ])
    assert np.allclose(expected_intercept, model.intercept_)

    expected_coef = np.log(expected_params) - np.log(1-expected_params)
    assert np.allclose(expected_coef, model.coef_)

  def test_predict_proba(self, model, data_matrix):
    expected_proba = np.array([
      [0.5,0.5],
      [3/4,1/4],
      [3/4,1/4],
      [1/4,3/4],
      [3/4,1/4],
      [0.5,0.5],
      [1/4,3/4],
      [1/4,3/4],
      [3/4,1/4],
      [1/4,3/4],
    ])

    proba = model.predict_proba(data_matrix)
    assert np.allclose(proba,expected_proba)

  def test_fit(self, model, data_matrix, labels):
    nb = NaiveBayes(n_iter=1)
    nb.fit(data_matrix, labels)

    expected_params = np.array([
      [(3+3/4)/(4+1),(2+1/4)/(4+1)],
      [(2+1/4)/(4+1), (3+3/4)/(4+1)]
    ])

    assert np.allclose(expected_params, nb.parameters_)
