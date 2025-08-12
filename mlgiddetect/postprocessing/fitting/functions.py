from enum import Enum
from abc import abstractmethod
from typing import Tuple, Dict, Callable

import numpy as np


class Gaussian():
    

    def get_func( x: float, m: float, c: float, a: float, mu: float, sigma: float) -> float: 
        return m*x + c + a * np.exp(-.5 * ((x - mu)/sigma) ** 2 ) / (np.sqrt(2*np.pi)*sigma)



    def get_func(self, background: Background) -> Callable:


        return func

    def __call__(self, x: np.ndarray, background: Background, *params) -> np.ndarray:
        return self.func(x, *params[:self.NUM]) + background(x, *params[self.NUM:])


    def bounds(x: np.ndarray, y: np.ndarray, init_params: np.ndarray) -> tuple:
        radius = (max(x) - min(x)) / 2
        m_init = init_params[0]
        c_init = init_params[1]
        a_init = init_params[2]
        mu_init = init_params[3]
        m_max = m_init + abs(m_init) / 2 + .5
        m_min = m_init - abs(m_init) / 2
        c_max =  c_init + abs(c_init) / 2 + 1
        c_min = c_init - abs(c_init) / 2
        a_max = max(y.max(), a_init * 2)
        a_min = min(a_init, 0)
        mu_max = max(x)
        mu_min = min(x)
        sigma_max = radius
        sigma_min = 0

        return [m_min, c_min, a_min, mu_min, sigma_min], [m_max, c_max, a_max, mu_max, sigma_max]

    def gauss_plus_linear( x: float, m: float, c: float, a: float, mu: float, sigma: float) -> float: 
        return m*x + c + a * np.exp(-.5 * ((x - mu)/sigma) ** 2 ) / (np.sqrt(2*np.pi)*sigma)



from .utils import Roi, _update_bounds
from .background import Background

__all__ = ['FittingType', 'FittingFunction', 'FITTING_FUNCTIONS', 'Gaussian', 'Lorentzian']


class FittingType(Enum):
    gaussian = 'Gaussian'
    lorentzian = 'Lorentzian'


class FittingFunction(object):
    NAME: str = ''
    PARAM_NAMES: tuple = ()
    NUM: int = 0
    TYPE: FittingType = None
    is_default: bool = True

    def __init__(self, *args, **kwargs):
        pass

    def get_func(self, background: Background) -> Callable:
        def func(x: np.ndarray, *params):
            return self.__call__(x, background, *params)

        return func

    def __call__(self, x: np.ndarray, background: Background, *params) -> np.ndarray:
        return self.func(x, *params[:self.NUM]) + background(x, *params[self.NUM:])

    @staticmethod
    @abstractmethod
    def func(x: np.ndarray, *params):
        pass

    @staticmethod
    @abstractmethod
    def set_roi_from_params(roi: Roi, params: list):
        pass

    @staticmethod
    @abstractmethod
    def set_params_from_roi(roi: Roi, params: list):
        pass

    @staticmethod
    @abstractmethod
    def _bounds(x: np.ndarray, y: np.ndarray, roi: Roi, background: Background):
        pass

    @classmethod
    def bounds(cls, x: np.ndarray, y: np.ndarray, roi: Roi, background: Background, params_from_roi: bool = False):
        if params_from_roi:
            return _update_bounds(roi, list(cls.PARAM_NAMES) + list(background.PARAM_NAMES),
                                  *cls._bounds(x, y, roi, background))
        else:
            return cls._bounds(x, y, roi, background)


class Gaussian(FittingFunction):
    NAME: str = 'Gaussian'
    PARAM_NAMES: tuple = ('peak height', 'radius', 'width')
    NUM = 3
    TYPE = FittingType.gaussian

    @staticmethod
    def func(x: np.ndarray, *params):
        amp, mu, sigma, *_ = params
        return amp * np.exp(- 2 * (x - mu) ** 2 / sigma ** 2)

    @staticmethod
    def set_roi_from_params(roi: Roi, params: list):
        roi.radius = params[1]
        roi.width = params[2]

    @staticmethod
    def _bounds(x: np.ndarray, y: np.ndarray, roi: Roi, background: Background):
        init_b, upper_b, lower_b = background.bounds(x, y, roi)
        amp, amp_max, amp_min = background.amp_bounds(x, y, init_b)

        return ([amp, roi.radius, roi.width] + init_b,
                [amp_max, roi.radius + roi.width / 2, roi.width * 2] + upper_b,
                [amp_min, roi.radius - roi.width / 2, 0] + lower_b)


class Lorentzian(Gaussian):
    NAME: str = 'Lorentzian'
    PARAM_NAMES: tuple = ('peak height', 'radius', 'width')
    TYPE = FittingType.lorentzian

    @staticmethod
    def func(x: np.ndarray, *params):
        amp, mu, sigma, *_ = params
        w = (sigma / 2) ** 2
        return amp * w / (w + (x - mu) ** 2)


FITTING_FUNCTIONS: Dict[FittingType, FittingFunction.__class__] = {
    FittingType.gaussian: Gaussian,
    FittingType.lorentzian: Lorentzian,
}