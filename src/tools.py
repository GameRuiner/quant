import numpy as np
from pandas import DataFrame
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    int_like,
    string_like,
)
from typing import Literal
from statsmodels.tools.typing import NDArray


def diff(series, k_diff=1, k_seasonal_diff=None, seasonal_periods=1):
    r"""
    Difference a series simply and/or seasonally along the zero-th axis.

    Given a series (denoted :math:`y_t`), performs the differencing operation

    .. math::

        \Delta^d \Delta_s^D y_t

    where :math:`d =` `diff`, :math:`s =` `seasonal_periods`,
    :math:`D =` `seasonal\_diff`, and :math:`\Delta` is the difference
    operator.

    Parameters
    ----------
    series : array_like
        The series to be differenced.
    k_diff : int, optional
        The number of simple differences to perform. Default is 1.
    k_seasonal_diff : int or None, optional
        The number of seasonal differences to perform. Default is no seasonal
        differencing.
    seasonal_periods : int, optional
        The seasonal lag. Default is 1. Unused if there is no seasonal
        differencing.

    Returns
    -------
    differenced : ndarray
        The differenced array.
    """
    differenced = series

    # Seasonal differencing
    if k_seasonal_diff is not None:
        while k_seasonal_diff > 0:
            differenced = (differenced[seasonal_periods:] - differenced[:-seasonal_periods])
            k_seasonal_diff -= 1

    # Simple differencing
    differenced = np.diff(differenced, k_diff, axis=0)
    return differenced

def prepare_trend_data(polynomial_trend, k_trend, nobs, offset=1):
    # Cache the arrays for calculating the intercept from the trend
    # components
    time_trend = np.arange(offset, nobs + offset)
    trend_data = np.zeros((nobs, k_trend))
    i = 0
    for k in polynomial_trend.nonzero()[0]:
        if k == 0:
            trend_data[:, i] = np.ones(nobs,)
        else:
            trend_data[:, i] = time_trend**k
        i += 1

    return trend_data


def prepare_exog(exog):
    k_exog = 0
    if exog is not None:
        exog = np.asarray(exog)

        # Make sure we have 2-dimensional array
        if exog.ndim == 1:
            exog = exog[:, None]

        k_exog = exog.shape[1]
    return k_exog, exog

def prepare_trend_spec(trend):
    # Trend
    if trend is None or trend == 'n':
        polynomial_trend = np.ones(0)
    elif trend == 'c':
        polynomial_trend = np.r_[1]
    elif trend == 't':
        polynomial_trend = np.r_[0, 1]
    elif trend == 'ct':
        polynomial_trend = np.r_[1, 1]
    elif trend == 'ctt':
        # TODO deprecate ctt?
        polynomial_trend = np.r_[1, 1, 1]
    else:
        trend = np.array(trend)
        if trend.ndim > 0:
            polynomial_trend = (trend > 0).astype(int)
        else:
            raise ValueError(
                "Valid trend inputs are 'c' (constant), 't' (linear trend in "
                "time), 'ct' (both), 'ctt' (both with trend squared) or an "
                "interable defining a polynomial, e.g., [1, 1, 0, 1] is `a + "
                f"b*t + ct**3`. Received {trend}"
            )

    # Note: k_trend is not the degree of the trend polynomial, because e.g.
    # k_trend = 1 corresponds to the degree zero polynomial (with only a
    # constant term).
    k_trend = int(np.sum(polynomial_trend))

    return polynomial_trend, k_trend

def lagmat(x,
           maxlag: int,
           trim: Literal["forward", "backward", "both", "none"]='forward',
           original: Literal["ex", "sep", "in"]="ex",
           use_pandas: bool=False
           )-> NDArray | DataFrame | tuple[NDArray, NDArray] | tuple[DataFrame, DataFrame]:
    """
    Create 2d array of lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend
    index values.

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])
    """
    maxlag = int_like(maxlag, "maxlag")
    use_pandas = bool_like(use_pandas, "use_pandas")
    trim = string_like(
        trim,
        "trim",
        optional=True,
        options=("forward", "backward", "both", "none"),
    )
    original = string_like(original, "original", options=("ex", "sep", "in"))

    # TODO:  allow list of lags additional to maxlag
    orig = x
    x = array_like(x, "x", ndim=2, dtype=None)
    is_pandas = False
    trim = "none" if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ("none", "backward"):
        raise ValueError(
            "trim cannot be 'none' or 'backward' when used on "
            "Series or DataFrames"
        )

    dropidx = 0
    nobs, nvar = x.shape
    if original in ["ex", "sep"]:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[
        maxlag - k: nobs + maxlag - k,
        nvar * (maxlag - k): nvar * (maxlag - k + 1),
        ] = x

    if trim in ("none", "forward"):
        startobs = 0
    elif trim in ("backward", "both"):
        startobs = maxlag
    else:
        raise ValueError("trim option not valid")

    if trim in ("none", "backward"):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x = orig
        if isinstance(x, DataFrame):
            x_columns = [str(c) for c in x.columns]
            if len(set(x_columns)) != x.shape[1]:
                raise ValueError(
                    "Columns names must be distinct after conversion to string "
                    "(if not already strings)."
                )
        else:
            x_columns = [str(x.name)]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + ".L." + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ("sep", "ex"):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, axis=1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == "sep":
            leads = lm[startobs:stopobs, :dropidx]

    if original == "sep":
        return lags, leads
    else:
        return lags