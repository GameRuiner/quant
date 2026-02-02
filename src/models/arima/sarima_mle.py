"""
Based on
    SARIMAX Model

    Author: Chad Fulton
    License: Simplified-BSD
"""

from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.mlemodel import MLEModel
import numpy as np
from warnings import warn

from statsmodels.tsa.statespace.tools import companion_matrix

from models.arima.tools import diff, prepare_trend_data, prepare_exog, prepare_trend_spec, lagmat


class SARIMAModel(MLEModel):
    """SARIMA(p,d,q)(P,D,Q)s model"""
    def __init__(self, y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None):
        integrated = order[1] > 0
        if trend is None:
            trend = "c" if not integrated else "n"
        self._spec = SARIMAXSpecification(y, order=order, seasonal_order=seasonal_order, trend=trend)
        self._params = SARIMAXParams(self._spec)

        # Lag polynomials
        self._params.ar_params = -1
        self.polynomial_ar = self._params.ar_poly.coef
        self._polynomial_ar = self.polynomial_ar.copy()

        self._params.ma_params = 1
        self.polynomial_ma = self._params.ma_poly.coef
        self._polynomial_ma = self.polynomial_ma.copy()

        self._params.seasonal_ar_params = -1
        self.polynomial_seasonal_ar = self._params.seasonal_ar_poly.coef
        self._polynomial_seasonal_ar = self.polynomial_seasonal_ar.copy()

        self._params.seasonal_ma_params = 1
        self.polynomial_seasonal_ma = self._params.seasonal_ma_poly.coef
        self._polynomial_seasonal_ma = self.polynomial_seasonal_ma.copy()

        # Deterministic trend polynomial
        self.trend = trend
        self.trend_offset = 1
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._polynomial_trend = self.polynomial_trend.copy()
        self._k_trend = self.k_trend

        # Model orders
        # Note: k_ar, k_ma, k_seasonal_ar, k_seasonal_ma do not include the
        # constant term, so they may be zero.
        # Note: for a typical ARMA(p,q) model, p = k_ar_params = k_ar - 1 and
        # q = k_ma_params = k_ma - 1, although this may not be true for models
        # with arbitrary log polynomials.
        self.k_ar = self._spec.max_ar_order
        self.k_ar_params = self._spec.k_ar_params
        self.k_seasonal_ar = self._spec.max_seasonal_ar_order * self._spec.seasonal_periods
        self.k_ma = self._spec.max_ma_order
        self.k_ma_params = self._spec.k_ma_params
        self.k_seasonal_ma = self._spec.max_seasonal_ma_order * self._spec.seasonal_periods
        self.k_seasonal_ar_params = self._spec.k_seasonal_ar_params
        self._k_order = max(self.k_ar + self.k_seasonal_ar, self.k_ma + self.k_seasonal_ma + 1)
        self.k_seasonal_ma_params = self._spec.k_seasonal_ma_params

        self.k_seasonal_diff = int(seasonal_order[1])
        self.k_diff = int(order[1])

        # Model order
        # (this is used internally in a number of locations)
        self._k_order = max(self.k_ar + self.k_seasonal_ar,
                            self.k_ma + self.k_seasonal_ma + 1)

        # Make internal copies of the differencing orders because if we use
        # simple differencing, then we will need to internally use zeros after
        # the simple differencing has been performed
        self._k_diff = self.k_diff
        self._k_seasonal_diff = self.k_seasonal_diff

        # Model parameters
        self.seasonal_periods = seasonal_order[3]
        self.mle_regression = True
        self.time_varying_regression = False
        self.measurement_error = False

        # Exogenous data
        (self._k_exog, exog) = prepare_exog(None)
        # (we internally use _k_exog for mechanics so that the public attribute
        # can be overridden by subclasses)
        self.k_exog = self._k_exog

        # Redefine mle_regression to be true only if it was previously set to
        # true and there are exogenous regressors
        self.mle_regression = (
            self.mle_regression and exog is not None and self._k_exog > 0
        )
        # State regression is regression with coefficients estimated within
        # the state vector
        self.state_regression = (
            not self.mle_regression and exog is not None and self._k_exog > 0
        )
        # If all we have is a regression (so k_ar = k_ma = 0), then put the
        # error term as measurement error
        if self.state_regression and self._k_order == 0:
            self.measurement_error = True

        k_states = self._k_order + self.seasonal_periods * self._k_seasonal_diff + self._k_diff

        self._k_states_diff = (
            self._k_diff + self.seasonal_periods * self._k_seasonal_diff
        )

        # Number of positive definite elements of the state covariance matrix
        k_posdef = int(self._k_order > 0)
        # Only have an error component to the states if k_posdef > 0
        self.state_error = k_posdef > 0
        if self.state_regression and self.time_varying_regression:
            k_posdef += self._k_exog

        # Number of positive definite elements of the state covariance matrix
        k_posdef = int(self._k_order > 0)
        # Only have an error component to the states if k_posdef > 0
        self.state_error = k_posdef > 0
        if self.state_regression and self.time_varying_regression:
            k_posdef += self._k_exog

        self.k_states = k_states
        self.k_posdef = k_posdef

        # Initialize the statespace
        super().__init__(
            y, exog=exog, k_states=k_states, k_posdef=k_posdef
        )

        self.ssm['design'] = self.initial_design
        # REMOVE
        self.ssm['state_intercept'] = self.initial_state_intercept

        self.ssm['transition'] = self.initial_transition
        self.ssm['selection'] = self.initial_selection


        if self.ssm.initialization is None:
            self.initialize_default()



    def prepare_data(self):
        endog, exog = super().prepare_data()

        # Reset the nobs
        self.nobs = endog.shape[0]

        # Cache the arrays for calculating the intercept from the trend
        # components
        self._trend_data = prepare_trend_data(
            self.polynomial_trend, self._k_trend, self.nobs, self.trend_offset)

        return endog, exog

    def initialize(self):
        super().initialize()

        # Cache the indexes of included polynomial orders (for update below)
        # (but we do not want the index of the constant term, so exclude the
        # first index)
        self._polynomial_ar_idx = np.nonzero(self.polynomial_ar)[0][1:]
        self._polynomial_ma_idx = np.nonzero(self.polynomial_ma)[0][1:]
        self._polynomial_seasonal_ar_idx = np.nonzero(
            self.polynomial_seasonal_ar
        )[0][1:]
        self._polynomial_seasonal_ma_idx = np.nonzero(
            self.polynomial_seasonal_ma
        )[0][1:]

        # Save the indices corresponding to the reduced form lag polynomial
        # parameters in the transition and selection matrices so that they
        # do not have to be recalculated for each update()
        start_row = self._k_states_diff
        end_row = start_row + self.k_ar + self.k_seasonal_ar
        col = self._k_states_diff
        self.transition_ar_params_idx = (
            np.s_['transition', start_row:end_row, col]
        )

        start_row += 1
        end_row = start_row + self.k_ma + self.k_seasonal_ma
        col = 0
        self.selection_ma_params_idx = (
            np.s_['selection', start_row:end_row, col]
        )

        # Cache indices for exog variances in the state covariance matrix
        if self.state_regression and self.time_varying_regression:
            idx = np.diag_indices(self.k_posdef)
            self._exog_variance_idx = ('state_cov', idx[0][-self._k_exog:],
                                       idx[1][-self._k_exog:])


    def initialize_default(self, approximate_diffuse_variance=None):
        """Initialize default"""
        if approximate_diffuse_variance is None:
            approximate_diffuse_variance = self.ssm.initial_variance
        diffuse_type = 'approximate_diffuse'

        k_diffuse_states = self.k_states
        self.loglikelihood_burn = k_diffuse_states

        init = Initialization(
            self.k_states,
            approximate_diffuse_variance=approximate_diffuse_variance)

        init.set(None, diffuse_type)

        self.ssm.initialization = init

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.
        """
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        params_trend = None
        params_exog = None
        params_ar = None
        params_ma = None
        params_seasonal_ar = None
        params_seasonal_ma = None
        params_exog_variance = None
        params_measurement_variance = None
        params_variance = None

        # Extract the parameters
        start = end = 0
        end += self._k_trend
        params_trend = params[start:end]
        start += self._k_trend
        if self.mle_regression:
            end += self._k_exog
            params_exog = params[start:end]
            start += self._k_exog
        end += self.k_ar_params
        params_ar = params[start:end]
        start += self.k_ar_params
        end += self.k_ma_params
        params_ma = params[start:end]
        start += self.k_ma_params
        end += self.k_seasonal_ar_params
        params_seasonal_ar = params[start:end]
        start += self.k_seasonal_ar_params
        end += self.k_seasonal_ma_params
        params_seasonal_ma = params[start:end]
        start += self.k_seasonal_ma_params
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            params_exog_variance = params[start:end]
            start += self._k_exog
        if self.measurement_error:
            params_measurement_variance = params[start]
            start += 1
            end += 1
        if self.state_error:
            params_variance = params[start]
        # start += 1
        # end += 1

        # Update lag polynomials
        if self.k_ar > 0:
            if self._polynomial_ar.dtype == params.dtype:
                self._polynomial_ar[self._polynomial_ar_idx] = -params_ar
            else:
                polynomial_ar = self._polynomial_ar.real.astype(params.dtype)
                polynomial_ar[self._polynomial_ar_idx] = -params_ar
                self._polynomial_ar = polynomial_ar

        if self.k_ma > 0:
            if self._polynomial_ma.dtype == params.dtype:
                self._polynomial_ma[self._polynomial_ma_idx] = params_ma
            else:
                polynomial_ma = self._polynomial_ma.real.astype(params.dtype)
                polynomial_ma[self._polynomial_ma_idx] = params_ma
                self._polynomial_ma = polynomial_ma

        if self.k_seasonal_ar > 0:
            idx = self._polynomial_seasonal_ar_idx
            if self._polynomial_seasonal_ar.dtype == params.dtype:
                self._polynomial_seasonal_ar[idx] = -params_seasonal_ar
            else:
                polynomial_seasonal_ar = (
                    self._polynomial_seasonal_ar.real.astype(params.dtype)
                )
                polynomial_seasonal_ar[idx] = -params_seasonal_ar
                self._polynomial_seasonal_ar = polynomial_seasonal_ar

        if self.k_seasonal_ma > 0:
            idx = self._polynomial_seasonal_ma_idx
            if self._polynomial_seasonal_ma.dtype == params.dtype:
                self._polynomial_seasonal_ma[idx] = params_seasonal_ma
            else:
                polynomial_seasonal_ma = (
                    self._polynomial_seasonal_ma.real.astype(params.dtype)
                )
                polynomial_seasonal_ma[idx] = params_seasonal_ma
                self._polynomial_seasonal_ma = polynomial_seasonal_ma

        # Get the reduced form lag polynomial terms by multiplying the regular
        # and seasonal lag polynomials
        # Note: that although the numpy np.polymul examples assume that they
        # are ordered from highest degree to lowest, whereas our are from
        # lowest to highest, it does not matter.
        if self.k_seasonal_ar > 0:
            reduced_polynomial_ar = -np.polymul(
                self._polynomial_ar, self._polynomial_seasonal_ar
            )
        else:
            reduced_polynomial_ar = -self._polynomial_ar
        if self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self._polynomial_ma, self._polynomial_seasonal_ma
            )
        else:
            reduced_polynomial_ma = self._polynomial_ma

        # Observation intercept
        # Exogenous data with MLE estimation of parameters enters through a
        # time-varying observation intercept (is equivalent to simply
        # subtracting it out of the endogenous variable first)
        if self.mle_regression:
            self.ssm['obs_intercept'] = np.dot(self.exog, params_exog)[None, :]

        # State intercept (Harvey) or additional observation intercept
        # (Hamilton)
        # SARIMA trend enters through the a time-varying state intercept,
        # associated with the first row of the stationary component of the
        # state vector (i.e. the first element of the state vector following
        # any differencing elements)
        if self._k_trend > 0:
            data = np.dot(self._trend_data, params_trend).astype(params.dtype)
            self.ssm['state_intercept', self._k_states_diff, :] = data

        # Observation covariance matrix
        if self.measurement_error:
            self.ssm['obs_cov', 0, 0] = params_measurement_variance

        # Transition matrix
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            self.ssm[self.transition_ar_params_idx] = reduced_polynomial_ar[1:]
        elif not self.ssm.transition.dtype == params.dtype:
            # This is required if the transition matrix is not really in use
            # (e.g. for an MA(q) process) so that it's dtype never changes as
            # the parameters' dtype changes. This changes the dtype manually.
            self.ssm['transition'] = self.ssm['transition'].real.astype(
                params.dtype)

        # Selection matrix (Harvey) or Design matrix (Hamilton)
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            self.ssm[self.selection_ma_params_idx] = (
                reduced_polynomial_ma[1:]
            )

        # State covariance matrix
        if self.k_posdef > 0:
            self['state_cov', 0, 0] = params_variance
            if self.state_regression and self.time_varying_regression:
                self.ssm[self._exog_variance_idx] = params_exog_variance

        return params

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Used primarily to enforce stationarity of the autoregressive lag
        polynomial, invertibility of the moving average lag polynomial, and
        positive variance parameters.

        Parameters
        ----------
        unconstrained : array_like
            Unconstrained parameters used by the optimizer.

        Returns
        -------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self._k_trend > 0:
            end += self._k_trend
            constrained[start:end] = unconstrained[start:end]
            start += self._k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self._k_exog
            constrained[start:end] = unconstrained[start:end]
            start += self._k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            constrained[start:end] = unconstrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            constrained[start:end] = unconstrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            constrained[start:end] = unconstrained[start:end]
            start += self.k_seasonal_ma_params

        # Transform the standard deviation parameters to be positive
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            constrained[start:end] = unconstrained[start:end]**2
            start += self._k_exog
        if self.measurement_error:
            constrained[start] = unconstrained[start]**2
            start += 1
            end += 1
        if self.state_error:
            constrained[start] = unconstrained[start]**2
            # start += 1
            # end += 1

        return constrained


    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Used primarily to reverse enforcement of stationarity of the
        autoregressive lag polynomial and invertibility of the moving average
        lag polynomial.

        Parameters
        ----------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Returns
        -------
        constrained : array_like
            Unconstrained parameters used by the optimizer.

        Notes
        -----
        If the lag polynomial has non-consecutive powers (so that the
        coefficient is zero on some element of the polynomial), then the
        constraint function is not onto the entire space of invertible
        polynomials, although it only excludes a very small portion very close
        to the invertibility boundary.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, constrained.dtype)

        start = end = 0

        # Retain the trend parameters
        if self._k_trend > 0:
            end += self._k_trend
            unconstrained[start:end] = constrained[start:end]
            start += self._k_trend

        # Retain any MLE regression coefficients
        if self.mle_regression:
            end += self._k_exog
            unconstrained[start:end] = constrained[start:end]
            start += self._k_exog

        # Transform the AR parameters (phi) to be stationary
        if self.k_ar_params > 0:
            end += self.k_ar_params
            unconstrained[start:end] = constrained[start:end]
            start += self.k_ar_params

        # Transform the MA parameters (theta) to be invertible
        if self.k_ma_params > 0:
            end += self.k_ma_params
            unconstrained[start:end] = constrained[start:end]
            start += self.k_ma_params

        # Transform the seasonal AR parameters (\tilde phi) to be stationary
        if self.k_seasonal_ar > 0:
            end += self.k_seasonal_ar_params
            unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ar_params

        # Transform the seasonal MA parameters (\tilde theta) to be invertible
        if self.k_seasonal_ma_params > 0:
            end += self.k_seasonal_ma_params
            unconstrained[start:end] = constrained[start:end]
            start += self.k_seasonal_ma_params

        # Untransform the standard deviation
        if self.state_regression and self.time_varying_regression:
            end += self._k_exog
            unconstrained[start:end] = constrained[start:end]**0.5
            start += self._k_exog
        if self.measurement_error:
            unconstrained[start] = constrained[start]**0.5
            start += 1
            end += 1
        if self.state_error:
            unconstrained[start] = constrained[start]**0.5
            # start += 1
            # end += 1

        return unconstrained
    @property
    def initial_transition(self):
        """Initial transition matrix"""
        transition = np.zeros((self.k_states, self.k_states))

        # Exogenous regressors component
        if self.state_regression:
            start = -self._k_exog
            # T_\beta
            transition[start:, start:] = np.eye(self._k_exog)

            # Autoregressive component
            start = -(self._k_exog + self._k_order)
            end = -self._k_exog if self._k_exog > 0 else None
        else:
            # Autoregressive component
            start = -self._k_order
            end = None

        # T_c
        if self._k_order > 0:
            transition[start:end, start:end] = companion_matrix(self._k_order)

        # Seasonal differencing component
        # T^*
        if self._k_seasonal_diff > 0:
            seasonal_companion = companion_matrix(self.seasonal_periods).T
            seasonal_companion[0, -1] = 1
            for d in range(self._k_seasonal_diff):
                start = self._k_diff + d * self.seasonal_periods
                end = self._k_diff + (d + 1) * self.seasonal_periods

                # T_c^*
                transition[start:end, start:end] = seasonal_companion

                # i
                if d < self._k_seasonal_diff - 1:
                    transition[start, end + self.seasonal_periods - 1] = 1

                # \iota
                transition[start, self._k_states_diff] = 1

        # Differencing component
        if self._k_diff > 0:
            idx = np.triu_indices(self._k_diff)
            # T^**
            transition[idx] = 1
            # [0 1]
            if self.seasonal_periods > 0:
                start = self._k_diff
                end = self._k_states_diff
                transition[:self._k_diff, start:end] = (
                    ([0] * (self.seasonal_periods - 1) + [1]) *
                    self._k_seasonal_diff)
            # [1 0]
            column = self._k_states_diff
            transition[:self._k_diff, column] = 1

        return transition

    @property
    def initial_selection(self):
        """Initial selection matrix"""
        if not (self.state_regression and self.time_varying_regression):
            if self.k_posdef > 0:
                selection = np.r_[
                    [0] * (self._k_states_diff),
                    [1] * (self._k_order > 0), [0] * (self._k_order - 1),
                    [0] * ((1 - self.mle_regression) * self._k_exog)
                ][:, None]

                if len(selection) == 0:
                    selection = np.zeros((self.k_states, self.k_posdef))
            else:
                selection = np.zeros((self.k_states, 0))
        else:
            selection = np.zeros((self.k_states, self.k_posdef))
            # Typical state variance
            if self._k_order > 0:
                selection[0, 0] = 1
            # Time-varying regression coefficient variances
            for i in range(self._k_exog, 0, -1):
                selection[-i, -i] = 1
        return selection

    @property
    def initial_state_intercept(self):
        """Initial state intercept vector"""
        # TODO make this self._k_trend > 1 and adjust the update to take
        # into account that if the trend is a constant, it is not time-varying
        if self._k_trend > 0:
            state_intercept = np.zeros((self.k_states, self.nobs))
        else:
            state_intercept = np.zeros((self.k_states,))
        return state_intercept

    @property
    def model_orders(self):
        """
        The orders of each of the polynomials in the model.
        """
        return {
            'trend': self._k_trend,
            'exog': self._k_exog,
            'ar': self.k_ar,
            'ma': self.k_ma,
            'seasonal_ar': self.k_seasonal_ar,
            'seasonal_ma': self.k_seasonal_ma,
            'reduced_ar': self.k_ar + self.k_seasonal_ar,
            'reduced_ma': self.k_ma + self.k_seasonal_ma,
            'exog_variance': self._k_exog if (
                self.state_regression and self.time_varying_regression) else 0,
            'measurement_variance': int(self.measurement_error),
            'variance': int(self.state_error),
        }

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.r_[
            [1] * self._k_diff,
            ([0] * (self.seasonal_periods - 1) + [1]) * self._k_seasonal_diff,
            [1] * self.state_error, [0] * (self._k_order - 1)
        ]

        if len(design) == 0:
            design = np.r_[0]

        # If we have exogenous regressors included as part of the state vector
        # then the exogenous data is incorporated as a time-varying component
        # of the design matrix
        if self.state_regression:
            if self._k_order > 0:
                design = np.c_[
                    np.reshape(
                        np.repeat(design, self.nobs),
                        (design.shape[0], self.nobs)
                    ).T,
                    self.exog
                ].T[None, :, :]
            else:
                design = self.exog.T[None, :, :]
        return design

    @property
    def model_names(self):
        """
        The plain text names of all possible model parameters.
        """
        return self._get_model_names(latex=False)

    def _get_model_names(self, latex=False):
        names = {
            'trend': None,
            'exog': None,
            'ar': None,
            'ma': None,
            'seasonal_ar': None,
            'seasonal_ma': None,
            'reduced_ar': None,
            'reduced_ma': None,
            'exog_variance': None,
            'measurement_variance': None,
            'variance': None,
        }

        # Trend
        if self._k_trend > 0:
            trend_template = 't_%d' if latex else 'trend.%d'
            names['trend'] = []
            for i in self.polynomial_trend.nonzero()[0]:
                if i == 0:
                    names['trend'].append('intercept')
                elif i == 1:
                    names['trend'].append('drift')
                else:
                    names['trend'].append(trend_template % i)

        # Exogenous coefficients
        if self._k_exog > 0:
            names['exog'] = self.exog_names

        # Autoregressive
        if self.k_ar > 0:
            ar_template = '$\\phi_%d$' if latex else 'ar.L%d'
            names['ar'] = []
            for i in self.polynomial_ar.nonzero()[0][1:]:
                names['ar'].append(ar_template % i)

        # Moving Average
        if self.k_ma > 0:
            ma_template = '$\\theta_%d$' if latex else 'ma.L%d'
            names['ma'] = []
            for i in self.polynomial_ma.nonzero()[0][1:]:
                names['ma'].append(ma_template % i)

        # Seasonal Autoregressive
        if self.k_seasonal_ar > 0:
            seasonal_ar_template = (
                '$\\tilde \\phi_%d$' if latex else 'ar.S.L%d'
            )
            names['seasonal_ar'] = []
            for i in self.polynomial_seasonal_ar.nonzero()[0][1:]:
                names['seasonal_ar'].append(seasonal_ar_template % i)

        # Seasonal Moving Average
        if self.k_seasonal_ma > 0:
            seasonal_ma_template = (
                '$\\tilde \\theta_%d$' if latex else 'ma.S.L%d'
            )
            names['seasonal_ma'] = []
            for i in self.polynomial_seasonal_ma.nonzero()[0][1:]:
                names['seasonal_ma'].append(seasonal_ma_template % i)

        # Reduced Form Autoregressive
        if self.k_ar > 0 or self.k_seasonal_ar > 0:
            reduced_polynomial_ar = reduced_polynomial_ar = -np.polymul(
                self.polynomial_ar, self.polynomial_seasonal_ar
            )
            ar_template = '$\\Phi_%d$' if latex else 'ar.R.L%d'
            names['reduced_ar'] = []
            for i in reduced_polynomial_ar.nonzero()[0][1:]:
                names['reduced_ar'].append(ar_template % i)

        # Reduced Form Moving Average
        if self.k_ma > 0 or self.k_seasonal_ma > 0:
            reduced_polynomial_ma = np.polymul(
                self.polynomial_ma, self.polynomial_seasonal_ma
            )
            ma_template = '$\\Theta_%d$' if latex else 'ma.R.L%d'
            names['reduced_ma'] = []
            for i in reduced_polynomial_ma.nonzero()[0][1:]:
                names['reduced_ma'].append(ma_template % i)

        # Exogenous variances
        if self.state_regression and self.time_varying_regression:
            exog_var_template = ('$\\sigma_\\text{%s}^2$' if latex
                                     else 'var.%s')
            names['exog_variance'] = [
                exog_var_template % exog_name for exog_name in self.exog_names
            ]

        # Measurement error variance
        if self.measurement_error:
            meas_var_tpl = (
                    '$\\sigma_\\eta^2$' if latex else 'var.measurement_error')
            names['measurement_variance'] = [meas_var_tpl]

        # State variance
        if self.state_error:
            var_tpl = '$\\sigma_\\zeta^2$' if latex else 'sigma2'
            names['variance'] = [var_tpl]

        return names

    @property
    def param_terms(self):
        """
        List of parameters actually included in the model, in sorted order.

        TODO Make this an dict with slice or indices as the values.
        """
        model_orders = self.model_orders
        # Get basic list from model orders
        params_complete = [
            'trend', 'exog', 'ar', 'ma', 'seasonal_ar', 'seasonal_ma',
            'exog_variance', 'measurement_variance', 'variance'
        ]
        params = [
            order for order in params_complete
            if model_orders[order] > 0
        ]
        # k_exog may be positive without associated parameters if it is in the
        # state vector
        if 'exog' in params and not self.mle_regression:
            params.remove('exog')

        return params

    @property
    def param_names(self):
        """
        List of human-readable parameter names (for parameters actually
        included in the model).
        """
        params_sort_order = self.param_terms
        model_names = self.model_names
        return [
            name for param in params_sort_order for name in model_names[param]
        ]

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation
        """

        # Perform differencing if necessary (i.e. if simple differencing is
        # false so that the state-space model will use the entire dataset)
        trend_data = self._trend_data
        if self._k_diff > 0 or self._k_seasonal_diff > 0:
            endog = diff(self.endog, self._k_diff,
                         self._k_seasonal_diff, self.seasonal_periods)
            if self.exog is not None:
                exog = diff(self.exog, self._k_diff,
                            self._k_seasonal_diff, self.seasonal_periods)
            else:
                exog = None
            trend_data = trend_data[:endog.shape[0], :]
        else:
            endog = self.endog.copy()
            exog = self.exog.copy() if self.exog is not None else None
        endog = endog.squeeze()

        # Regression effects via OLS
        params_exog = []
        if self._k_exog > 0:
            params_exog = np.linalg.pinv(exog).dot(endog)
            endog = endog - np.dot(exog, params_exog)
        if self.state_regression:
            params_exog = []

        # Non-seasonal ARMA component and trend
        (params_trend, params_ar, params_ma,
         params_variance) = self._conditional_sum_squares(
            endog, self.k_ar, self.polynomial_ar, self.k_ma,
            self.polynomial_ma, self._k_trend, trend_data,
            warning_description='ARMA and trend')

        # Seasonal Parameters
        _, params_seasonal_ar, params_seasonal_ma, params_seasonal_variance = (
            self._conditional_sum_squares(
                endog, self.k_seasonal_ar, self.polynomial_seasonal_ar,
                self.k_seasonal_ma, self.polynomial_seasonal_ma,
                warning_description='seasonal ARMA'))

        # Variances
        params_exog_variance = []
        if self.state_regression and self.time_varying_regression:
            # TODO how to set the initial variance parameters?
            params_exog_variance = [1] * self._k_exog
        if (self.state_error and type(params_variance) is list and
                len(params_variance) == 0):
            if not (type(params_seasonal_variance) is list and
                    len(params_seasonal_variance) == 0):
                params_variance = params_seasonal_variance
            elif self._k_exog > 0:
                params_variance = np.inner(endog, endog)
            else:
                params_variance = np.inner(endog, endog) / self.nobs
        params_measurement_variance = 1 if self.measurement_error else []

        # Combine all parameters
        return np.r_[
            params_trend,
            params_exog,
            params_ar,
            params_ma,
            params_seasonal_ar,
            params_seasonal_ma,
            params_exog_variance,
            params_measurement_variance,
            params_variance
        ]

    @staticmethod
    def _conditional_sum_squares(endog, k_ar, polynomial_ar, k_ma,
                                 polynomial_ma, k_trend=0, trend_data=None,
                                 warning_description=None):
        k = 2 * k_ma
        r = max(k + k_ma, k_ar)

        k_params_ar = 0 if k_ar == 0 else len(polynomial_ar.nonzero()[0]) - 1
        k_params_ma = 0 if k_ma == 0 else len(polynomial_ma.nonzero()[0]) - 1

        residuals = None
        if k_ar + k_ma + k_trend > 0:
            try:
                # If we have MA terms, get residuals from an AR(k) model to use
                # as data for conditional sum of squares estimates of the MA
                # parameters
                if k_ma > 0:
                    Y = endog[k:]
                    X = lagmat(endog, k, trim='both')
                    params_ar = np.linalg.pinv(X).dot(Y)
                    residuals = Y - np.dot(X, params_ar)

                # Run an ARMA(p,q) model using the just computed residuals as
                # data
                Y = endog[r:]

                X = np.empty((Y.shape[0], 0))
                if k_trend > 0:
                    if trend_data is None:
                        raise ValueError('Trend data must be provided if'
                                         ' `k_trend` > 0.')
                    X = np.c_[X, trend_data[:(-r if r > 0 else None), :]]
                if k_ar > 0:
                    cols = polynomial_ar.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(endog, k_ar)[r:, cols]]
                if k_ma > 0:
                    cols = polynomial_ma.nonzero()[0][1:] - 1
                    X = np.c_[X, lagmat(residuals, k_ma)[r-k:, cols]]

                # Get the array of [ar_params, ma_params]
                params = np.linalg.pinv(X).dot(Y)
                residuals = Y - np.dot(X, params)
            except ValueError:
                if warning_description is not None:
                    warning_description = ' for %s' % warning_description
                else:
                    warning_description = ''
                warn('Too few observations to estimate starting parameters%s.'
                     ' All parameters except for variances will be set to'
                     ' zeros.' % warning_description)
                # Typically this will be raised if there are not enough
                # observations for the `lagmat` calls.
                params = np.zeros(k_trend + k_ar + k_ma, dtype=endog.dtype)
                if len(endog) == 0:
                    # This case usually happens when there are not even enough
                    # observations for a complete set of differencing
                    # operations (no hope of fitting, just set starting
                    # variance to 1)
                    residuals = np.ones(k_params_ma * 2 + 1, dtype=endog.dtype)
                else:
                    residuals = np.r_[
                        np.zeros(k_params_ma * 2, dtype=endog.dtype),
                        endog - np.mean(endog)]

        # Default output
        params_trend = []
        params_ar = []
        params_ma = []
        params_variance = []

        # Get the params
        offset = 0
        if k_trend > 0:
            params_trend = params[offset:k_trend + offset]
            offset += k_trend
        if k_ar > 0:
            params_ar = params[offset:k_params_ar + offset]
            offset += k_params_ar
        if k_ma > 0:
            params_ma = params[offset:k_params_ma + offset]
            offset += k_params_ma
        if residuals is not None:
            if len(residuals) > max(1, k_params_ma):
                params_variance = (residuals[k_params_ma:] ** 2).mean()
            else:
                params_variance = np.var(endog)

        return (params_trend, params_ar, params_ma,
                params_variance)