# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple
import logging

import numpy as np
from scipy.optimize import curve_fit

from .functions import FittingFunction
from .background import Background
from .utils import Roi
from .range_strategy import RangeStrategy


@dataclass
class Fit:
    roi: Roi

    r_range: Tuple[float, float]
    x_range: Tuple[int, int]
    x: np.ndarray
    y: np.ndarray

    init_curve: np.ndarray
    init_params: list
    lower_bounds: list
    upper_bounds: list

    fitted_params: list
    fit_errors: list
    fitting_curve: np.ndarray
    background_curve: np.ndarray

    fitting_function: FittingFunction
    background: Background
    range_strategy: RangeStrategy
    sigma: float = None
    x_profile: np.ndarray = None
    y_profile: np.ndarray = None



    def do_fit(self) -> None:
        if not self.y.size or not self.x.size:
            return
        try:
            func = self.fitting_function.get_func(self.background)
            popt, pcov = curve_fit(func, self.x, self.y, self.init_params,
                                   bounds=self.bounds)
            perr = np.sqrt(np.diag(pcov))

            self.fitted_params = popt.tolist()
            self.init_params = self.fitted_params
            self.fit_errors = perr.tolist()
            self.fitting_curve = func(self.x, *popt)
            self.background_curve = self.background(self.x, *popt)
            self.init_curve = self.fitting_curve
            self.update_roi_fit_dict()
            self.fitting_function.set_roi_from_params(self.roi, self.fitted_params)

        except (ValueError, RuntimeError) as err:
            logging.exception(err)
            return

    def set_roi_from_range(self):
        r1, r2 = self.r_range
        roi = self.roi
        roi.radius = (r1 + r2) / 2
        roi.width = (r2 - r1) / 2

    def update_roi_fit_dict(self):
        self.roi.fitted_parameters = {}

        if self.fitted_params is not None:
            self.roi.fitted_parameters.update(dict(zip(self.param_names, self.fitted_params)))
            self.roi.fitted_parameters['fitted_params'] = self.fitted_params
            self.roi.fitted_parameters['fit_errors'] = self.fit_errors

        self.roi.fitted_parameters['fitting_function'] = self.fitting_function.TYPE
        self.roi.fitted_parameters['background'] = self.background.TYPE
        self.roi.fitted_parameters['lower_bounds'] = self.lower_bounds
        self.roi.fitted_parameters['init_params'] = self.init_params
        self.roi.fitted_parameters['upper_bounds'] = self.upper_bounds
        self.roi.fitted_parameters['r_range'] = self.r_range
        self.roi.fitted_parameters['x_range'] = self.x_range


def fit(x: np.array, y:np.array, fit_function = gauss_plus_linear, init_params_function = gauss_init_params, bound_function = gauss_bounds) -> np.ndarray:
    init_params = init_params_function(x,y)
    bounds = bound_function(x, y, init_params)
    popt, pcov = curve_fit(fit_function, x, y, init_params, bounds=bounds, method='trf')
    return popt


def gauss_init_params(x: np.ndarray, y:np.ndarray) -> np.ndarray:
    radius = (max(x) - min(x)) / 2
    a_init = y.max() - (y[0] + y[-1]) / 2
    dx = (x[-1] - x[0]) or 1
    m_init = (y[-1] - y[0]) / dx
    c_init = y[0] - x[0] * m_init
    mu_init = np.mean(x)    
    sigma_init = radius/2
    return m_init, c_init, a_init, mu_init, sigma_init 

def gauss_plus_linear( x: float, m: float, c: float, a: float, mu: float, sigma: float) -> float: 
    return m*x + c + a * np.exp(-.5 * ((x - mu)/sigma) ** 2 ) / (np.sqrt(2*np.pi)*sigma)

