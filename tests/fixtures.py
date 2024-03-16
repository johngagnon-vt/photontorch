""" test fixtures """

# pragma: no cover

#############
## Imports ##
#############

import torch
import pytest
import numpy as np

import photontorch as pt


######################
## Useful Functions ##
######################


def default_components():
    excluded = ["Component"]
    items = [(name, cls) for name, cls in pt.__dict__.items() if _is_component(cls)]
    for name, cls in items:
        try:
            if not isinstance(cls, pt.Component):
                continue
        except:
            continue
        if name[0] != "_" and name[0] == name[0].upper() and name not in excluded:
            yield cls()


##############
## Fixtures ##
##############


## PyTorch
@pytest.fixture
def gen():
    """default pytorch random generator"""
    return torch.Generator(device="cpu").manual_seed(42)


## Environments


@pytest.fixture
def tenv():
    """default time domain environment"""
    return pt.Environment(num_t=7, num_wl=2)


@pytest.fixture
def fenv():
    """default frequency domain environment"""
    return pt.Environment(wl=np.linspace(1.5, 1.6, 100), freqdomain=True)


## Components


@pytest.fixture
def comp():
    """default base component"""
    return pt.Component()


@pytest.fixture
def wg():
    """default base waveguide"""
    return pt.Waveguide()


@pytest.fixture
def s():
    """default source"""
    return pt.Source()


@pytest.fixture
def d():
    """default detector"""
    return pt.Detector()


## Networks


@pytest.fixture
def unw():
    """default unterminated network"""
    with pt.Network() as nw:
        nw.wg1 = nw.wg2 = pt.Waveguide(length=5e-6)
        nw.link(1, "0:wg1:1", "0:wg2:1", 0)
    return nw


@pytest.fixture
def nw():
    """default network (source-waveguide-detector)"""
    with pt.Network() as nw:
        nw.wg = pt.Waveguide(length=1e-5)
        nw.s = pt.Source()
        nw.d = pt.Detector()
        nw.link("s:0", "0:wg:1", "0:d")
    return nw


@pytest.fixture
def rnw():
    """default ring network"""
    return pt.RingNetwork(2, 6).terminate()


@pytest.fixture
def reck():
    """default reck network"""
    return pt.ReckNxN(4).terminate()


@pytest.fixture
def clements():
    """default reck network"""
    return pt.ClementsNxN(4).terminate()


## Detectors


@pytest.fixture
def lpdet():
    return pt.LowpassDetector(
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        filter_order=4,
    )


@pytest.fixture
def photodet():
    return pt.Photodetector(
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        responsivity=1.0,
        dark_current=1e-10,
        load_resistance=1e6,
        filter_order=4,
        seed=9,
    )


## Connectors


@pytest.fixture
def conn():
    """default connector"""
    wg = pt.Waveguide()
    s = pt.Source()
    d = pt.Detector()
    conn = wg["ab"] * s["a"] * d["b"]
    return conn
