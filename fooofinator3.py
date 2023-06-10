import warnings

import numpy as np
from fooof import FOOOF
from fooof.core.errors import NoDataError, FitError, InconsistentDataError
from fooof.sim.gen import gen_periodic
from scipy.optimize import curve_fit
from statsmodels.compat import scipy
import scipy
from scipy import optimize

def expo_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component with a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = 10^offset * (1/(knee + x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """

    ys = np.zeros_like(xs)

    offset, knee, exp, shift = params

    ys = ys + offset - np.log10(knee + (xs+shift)**exp)

    return ys


def expo_nk_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component without a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, exp) that define Lorentzian function:
        y = 10^off * (1/(x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function, without a knee.
    """

    ys = np.zeros_like(xs)

    offset, exp, shift = params

    ys = ys + offset - np.log10((xs+shift)**exp)

    return ys

def get_ap_func(aperiodic_mode):
    """Select and return specified function for aperiodic component.

    Parameters
    ----------
    aperiodic_mode : {'fixed', 'knee'}
        Which aperiodic fitting function to return.

    Returns
    -------
    ap_func : function
        Function for the aperiodic component.

    Raises
    ------
    ValueError
        If the specified aperiodic mode label is not understood.
    """

    if aperiodic_mode == 'fixed':
        ap_func = expo_nk_function
    elif aperiodic_mode == 'knee':
        ap_func = expo_function
    else:
        raise ValueError("Requested aperiodic mode not understood.")

    return ap_func

def infer_ap_func(aperiodic_params):
    """Infers which aperiodic function was used, from parameters.

    Parameters
    ----------
    aperiodic_params : list of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee'}
        Which kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    InconsistentDataError
        If the given parameters are inconsistent with any available aperiodic function.
    """

    if len(aperiodic_params) == 3:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) == 4:
        aperiodic_mode = 'knee'
    else:
        raise InconsistentDataError("The given aperiodic parameters are "
                                    "inconsistent with available options.")

    return aperiodic_mode


def gen_aperiodic(freqs, aperiodic_params, aperiodic_mode=None):
    """Generate aperiodic values.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector to create aperiodic component for.
    aperiodic_params : list of float
        Parameters that define the aperiodic component.
    aperiodic_mode : {'fixed', 'knee'}, optional
        Which kind of aperiodic component to generate.
        If not provided, is inferred from the parameters.

    Returns
    -------
    ap_vals : 1d array
        Aperiodic values, in log10 spacing.
    """

    if not aperiodic_mode:
        aperiodic_mode = infer_ap_func(aperiodic_params)

    ap_func = get_ap_func(aperiodic_mode)

    ap_vals = ap_func(freqs, *aperiodic_params)

    return ap_vals


class FOOOFinator(FOOOF):

    def __init__(self, peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0,
                 peak_threshold=2.0, aperiodic_mode='fixed', verbose=True):
        super().__init__(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, min_peak_height=min_peak_height,
                         peak_threshold=peak_threshold, aperiodic_mode=aperiodic_mode, verbose=verbose)
        self._bw_std_edge = 0
        self._ap_bounds = ((-np.inf, -np.inf, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf))
        self._ap_guess = (None, 0, None, 0)

    def _full_fit(self, freqs, power_spectrum, aperiodic_params, gaussian_params):
        # Fit log PSD
        fit_target = power_spectrum

        # Compute error for given params
        def err_func(params):

            spec = gen_periodic(freqs, np.ndarray.flatten(gaussian_params))+gen_aperiodic(freqs,params,self.aperiodic_mode)

            # Check for NaNs or overlapping Gaussians
            # if np.any(np.isnan(spec)) or check_gaussian_overlap(params):
            if np.any(np.isnan(spec)):
                return 1000000

            err = np.sqrt(np.sum(np.power(spec - fit_target, 2)))

            return err

        # Fit
        #xopt = scipy.optimize.minimize(err_func, aperiodic_params, method='SLSQP', options={'disp': False})
        xopt = scipy.optimize.minimize(err_func, aperiodic_params, options={'disp': False})
        return xopt.x

    def _simple_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.
        """

        # Get the guess parameters and/or calculate from the data, as needed
        #   Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [power_spectrum[0] if not self._ap_guess[0] else self._ap_guess[0]]
        kne_guess = [self._ap_guess[1]] if self.aperiodic_mode == 'knee' else []
        exp_guess = [np.abs(self.power_spectrum[-1] - self.power_spectrum[0] /
                            np.log10(self.freqs[-1]) - np.log10(self.freqs[0]))
                     if not self._ap_guess[2] else self._ap_guess[2]]
        shift_guess = [self._ap_guess[3]]

        bounds=self._ap_bounds
        if self.aperiodic_mode=='fixed':
            bounds=((self._ap_bounds[0][0],self._ap_bounds[0][1],self._ap_bounds[0][3]),
                    (self._ap_bounds[1][0],self._ap_bounds[1][1],self._ap_bounds[1][3]))
        # Collect together guess parameters
        guess = np.squeeze(np.array([off_guess + kne_guess + exp_guess + shift_guess]))

        # Ignore warnings that are raised in curve_fit
        #   A runtime warning can occur while exploring parameters in curve fitting
        #     This doesn't effect outcome - it won't settle on an answer that does this
        #   It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs, power_spectrum, p0=guess,
                                                maxfev=self._maxfev, bounds=bounds)
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding parameters in "
                           "the simple aperiodic component fit.")

        return aperiodic_params

    def _robust_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum robustly, ignoring outliers.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.

        Raises
        ------
        FitError
            If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_ap_fit(freqs, power_spectrum)
        initial_fit = gen_aperiodic(freqs, popt)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = power_spectrum - initial_fit

        # Flatten outliers, defined as any points that drop below 0
        flatspec[flatspec < 0] = 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._ap_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = power_spectrum[perc_mask]

        bounds = self._ap_bounds
        if self.aperiodic_mode == 'fixed':
            bounds = ((self._ap_bounds[0][0], self._ap_bounds[0][1], self._ap_bounds[0][3]),
                      (self._ap_bounds[1][0], self._ap_bounds[1][1], self._ap_bounds[1][3]))

        # Second aperiodic fit - using results of first fit as guess parameters
        #  See note in _simple_ap_fit about warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs_ignore, spectrum_ignore, p0=popt,
                                                maxfev=self._maxfev, bounds=bounds)
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding "
                           "parameters in the robust aperiodic fit.")
        except TypeError:
            raise FitError("Model fitting failed due to sub-sampling in the robust aperiodic fit.")

        return aperiodic_params

    def fit(self, freqs=None, power_spectrum=None, freq_range=None, n_jobs=-1):
        """Fit the full power spectrum as a combination of periodic and aperiodic components using the median
        of aperidioc fits done over a range of frequencies.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to. If not provided, keeps the entire range.
        n_jobs : Number of parallel process to run. -1 (default) uses all available processors

        Raises
        ------
        NoDataError
            If no data is available to fit.
        FitError
            If model fitting fails to fit. Only raised in debug mode.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        # If freqs & power_spectrum provided together, add data to object.
        if freqs is not None and power_spectrum is not None:
            self.add_data(freqs, power_spectrum, freq_range)
        # If power spectrum provided alone, add to object, and use existing frequency data
        #   Note: be careful passing in power_spectrum data like this:
        #     It assumes the power_spectrum is already logged, with correct freq_range
        elif isinstance(power_spectrum, np.ndarray):
            self.power_spectrum = power_spectrum

        # Check that data is available
        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")

        # Check and warn about width limits (if in verbose mode)
        if self.verbose:
            self._check_width_limits()

        # In rare cases, the model fails to fit, and so uses try / except
        try:

            # Initial aperiodic fit
            self.aperiodic_params_ = self._robust_ap_fit(self.freqs, self.power_spectrum)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_, self.aperiodic_mode)

            last_err=np.inf
            self.error_=10000
            while last_err-self.error_>1e-10:
                print(last_err-self.error_)

                # Flatten the power spectrum using fit aperiodic fit
                self._spectrum_flat = self.power_spectrum - self._ap_fit

                # Find peaks, and fit them with gaussians
                self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

                # Calculate the peak fit
                #   Note: if no peaks are found, this creates a flat (all zero) peak fit
                self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))

                self.aperiodic_params_ = self._full_fit(self.freqs, self.power_spectrum, self.aperiodic_params_,
                                                        self.gaussian_params_)
                self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_, self.aperiodic_mode)

                # Create full power_spectrum model fit
                self.fooofed_spectrum_ = self._peak_fit + self._ap_fit

                # Create peak-removed (but not flattened) power spectrum
                self._spectrum_peak_rm = self.power_spectrum - self._peak_fit

                # Convert gaussian definitions to peak parameters
                self.peak_params_ = self._create_peak_params(self.gaussian_params_)

                last_err = self.error_
                self._calc_error()


            print(last_err - self.error_)


            # Calculate R^2 and error of the model fit
            self._calc_r_squared()
            self._calc_error()

        except FitError:

            # If in debug mode, re-raise the error
            if self._debug:
                raise

            # Clear any interim model results that may have run
            #   Partial model results shouldn't be interpreted in light of overall failure
            self._reset_data_results(clear_results=True)

            # Print out status
            if self.verbose:
                print("Model fitting was unsuccessful.")


if __name__=='__main__':
    fname = 'psd_20190917_L_M1_1.npz'
    # fname = 'psd_20180528_R_M1_2.npz'
    psd_data = np.load(fname, allow_pickle=True)
    freqs = psd_data['freqs']
    psd = psd_data['psd'][0, :]
    f = FOOOFinator(aperiodic_mode='fixed')
    f.fit(freqs, psd)
    f.plot(plot_peaks='shade', peak_kwargs={'color': 'green'})