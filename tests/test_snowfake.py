"""Test snowfake"""
import pytest
from snowfake import Snowfake
import snowfake

# Figure 15b.
params =  {
    'rho': 0.35,
    'beta': 1.4,
    'alpha': 0.001,
    'θ': 0.015,
    'κ': 0.05,
    'μ': 0.015,
    'γ': 0.01,
    'σ': 0.00005,  # Growth per step.
}

def test_snowfake_error():
    """Test the basics.
    """
    with pytest.raises(AttributeError):
        # Need to pass params.
        s = Snowfake(size=15)

def test_snowfake():
    """Test the basics.
    """
    s = Snowfake(size=15, **params)
    assert s.a.shape == (15, 15)
    assert s.a.sum() == 1
    assert abs(s.d.sum() - s.ρ * (15 * 15 - 1)) < 1e-6

def test_grow():
    s = Snowfake(size=15, random=2021, **params)
    s.grow(max_epochs=20)
    assert s._epochs == 20
    assert s.a.sum() == 7

def test_repr():
    s = Snowfake(size=15, random=2021, **params)
    assert s.__repr__() == 'Snowfake(size=15, random=2021, ρ=0.35, β=1.4, α=0.001, θ=0.015, κ=0.05, μ=0.015, γ=0.01, σ=5e-05)'

    s.grow(max_epochs=3)
    assert s.status() == "Snowfake(size=15, random=2021, epochs=3, attachments=1)"

def test_rectify():
    s = Snowfake(size=15, random=False, **params)
    s.grow(max_epochs=3)
    assert s.rectify('c').shape == (15, 15)

def test_rectify_skimage():
    """
    skimage's DeprecationWarnings are ignored; see pytest.ini.
    """
    s = Snowfake(size=15, random=False, **params)
    s.grow(max_epochs=3)
    # This is a bug really, but it's what it does right now...
    assert s.rectify_skimage('c').shape == (12, 12)

def test_random():
    s = snowfake.random()
    assert 'random' in s.params

    s = snowfake.random(rho=0.35, seed=42)
    assert s.params['ρ'] == 0.35
