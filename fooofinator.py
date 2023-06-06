import numpy as np
from fooof import FOOOF
from fooof.core.errors import NoDataError, FitError
from fooof.sim.gen import gen_aperiodic, gen_periodic
from joblib import Parallel, delayed


class FOOOFinator(FOOOF):

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

            start_params = np.arange(self.freq_range[0], self.freq_range[1] - 5, 1)

            def run_fooof(i):
                start = start_params[i]
                f = FOOOF(aperiodic_mode='fixed')
                f.fit(freqs, power_spectrum, freq_range=[start, self.freq_range[1]])
                if f.has_model:
                    ap_params = f.get_params('aperiodic_params')
                    return gen_aperiodic(freqs, ap_params)
                else:
                    return np.zeros(power_spectrum.shape) * np.nan

            aperiodic = Parallel(
                n_jobs=n_jobs
            )(delayed(run_fooof)(i) for i in range(len(start_params)))
            self._ap_fit=np.nanmedian(np.array(aperiodic), axis=0)

            self.aperiodic_params_ = self._simple_ap_fit(freqs, self._ap_fit)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Find peaks, and fit them with gaussians
            self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

            # Calculate the peak fit
            #   Note: if no peaks are found, this creates a flat (all zero) peak fit
            self._peak_fit = gen_periodic(freqs, np.ndarray.flatten(self.gaussian_params_))

            # Create peak-removed (but not flattened) power spectrum
            self._spectrum_peak_rm = self.power_spectrum - self._peak_fit

            # Create full power_spectrum model fit
            self.fooofed_spectrum_ = self._peak_fit + self._ap_fit

            # Convert gaussian definitions to peak parameters
            self.peak_params_ = self._create_peak_params(self.gaussian_params_)

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