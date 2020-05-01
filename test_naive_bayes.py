import pytest

from scipy.sparse import csr_matrix

from naive_bayes_em import *

@pytest.fixture(scope="session")
def data_matrix():
  return csr_matrix([
    [0.0,1.0,0.0,0.0,1.0],
    [0.0,0.0,1.0,0.0,0.0],
    [0.0,1.0,1.0,0.0,0.0],
    [0.0,0.0,0.0,1.0,1.0],
    [1.0,0.0,1.0,0.0,0.0],
    [1.0,0.0,0.0,1.0,0.0]]
  )

@pytest.fixture(scope="session")
def labels():
  return np.array([
    0,
    0,
    -1,
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
      [0,0.5,0.5,0,0.5],
      [0.5,0,0.5,0.5,0.5]
    ])
    expected_bias_params = np.array([
      2/4,2/4
    ])

    assert np.allclose(model.parameters_, expected_params)
    assert np.allclose(model.intercept_parameters_, expected_bias_params)

    expected_intercept = np.array([
      4*np.log(0.5), 5*np.log(0.5)
    ])
    assert np.allclose(expected_intercept, model.intercept_)

    expected_coef = np.log(expected_params) - np.log(1-expected_params)
    assert np.allclose(expected_coef, model.coef_)
