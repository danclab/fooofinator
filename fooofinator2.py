from functools import partial
from multiprocessing import cpu_count, Pool
from statistics import NormalDist

import scipy
import numpy as np
from fooof import FOOOF, FOOOFGroup
from fooof.core.errors import NoDataError, FitError, NoModelError
from fooof.core.info import get_indices
from fooof.core.io import save_fg, load_jsonlines
from fooof.core.items import OBJ_DESC
from fooof.core.modutils import copy_doc_func_to_method
from fooof.core.reports import save_report_fg
from fooof.core.strings import gen_results_fg_str
from fooof.core.utils import check_inds
from fooof.objs.group import _progress, _par_fit
from fooof.plts.fg import plot_fg
from fooof.sim.gen import gen_aperiodic, gen_periodic
from joblib import Parallel, delayed
from scipy.stats import norm


def total_overlap(params):
    total_overlap = 0
    n_gauss = int((len(params) - 3) / 3)
    for i in range(n_gauss):
        f1 = params[3 + i * 3]  # Mean of Gaussian 1
        sd1 = params[3 + i * 3 + 1]  # Standard deviation of Gaussian 1

        for j in range(i + 1, n_gauss):
            f2 = params[3 + j * 3]  # Mean of Gaussian 2
            sd2 = params[3 + j * 3 + 1]  # Standard deviation of Gaussian 2

            total_overlap = total_overlap + NormalDist(mu=f1, sigma=sd1).overlap(NormalDist(mu=f2, sigma=sd2))

    return total_overlap  # No overlap exceeds the threshold


# Generate parameterized spectrum
def gen_parameterized_spectrum(freqs, params):
    # Aperiodic component
    offset = params[0]
    slope = params[1]
    shift = params[2]
    spec = offset + slope * np.log10(1. / (freqs + shift))

    # Number of peaks
    n_gauss = int((len(params) - 3) / 3)
    for i in range(n_gauss):
        # Peak is a Gaussian centered on frequency f with
        # standard deviation sd
        f = params[3 + i * 3]
        sd = params[3 + i * 3 + 1]
        w = params[3 + i * 3 + 2]
        peak_dist = norm(f, sd)
        peak = peak_dist.pdf(freqs)
        if np.max(peak[:]) > 0:
            peak = peak / np.max(peak[:])

        spec = spec + w * peak
    return spec


def fit_psd(freqs, psd, max_n_peaks=5, alpha=.3, n_runs=50, n_jobs=-1, peak_width_limits=(.5, 12.0),
            min_peak_height=0.1, method='SLSQP'):
    # Fit log PSD
    fit_target = np.log10(psd)

    # Compute error for given params
    def err_func(params):

        spec = gen_parameterized_spectrum(freqs, params)

        # Check for NaNs or overlapping Gaussians
        # if np.any(np.isnan(spec)) or check_gaussian_overlap(params):
        if np.any(np.isnan(spec)):
            return 1000000

        err = np.sqrt(np.sum(np.power(spec - fit_target, 2)))

        # Cost includes aperiodic offset and peak weights
        cost = total_overlap(params)
        if len(params) > 3:
            cost = cost + np.sum(params[5::3])

        return err + alpha * cost

    best_params = None
    all_aics = []
    prev_params = None
    for n in range(max_n_peaks + 1):

        print('Fitting {} peaks'.format(n))
        # Shift, peak frequencies, width, and weights are bounded
        bounds = [(None, None), (None, None), (1e-6, None)]
        for i in range(n):
            bounds.extend([(freqs[0], freqs[-1]), peak_width_limits, (min_peak_height, None)])

        # Run fit with random initial conditions
        def run_fit():
            if prev_params is None:
                init_params = [np.random.uniform(low=-3, high=-2),
                               np.random.uniform(low=0, high=5),
                               np.random.uniform(low=0, high=5)]

                # Try to use peaks as initial params
                (peaks, peak_properties) = scipy.signal.find_peaks(fit_target, prominence=.1)
                for i in range(n):
                    if i < len(peaks):
                        init_params.append(freqs[peaks[i]])
                    else:
                        init_params.append(np.random.uniform(low=freqs[0], high=freqs[-1]))
                    init_params.append(np.random.uniform(low=peak_width_limits[0], high=peak_width_limits[1]))
                    init_params.append(np.random.uniform(low=min_peak_height, high=10))
            else:
                init_params = copy.copy(prev_params)
                (peaks, peak_properties) = scipy.signal.find_peaks(fit_target)
                if n - 1 < len(peaks):
                    init_params.append(freqs[peaks[n - 1]])
                else:
                    init_params.append(np.random.uniform(low=freqs[0], high=freqs[-1]))
                init_params.append(np.random.uniform(low=peak_width_limits[0], high=peak_width_limits[1]))
                init_params.append(np.random.uniform(low=min_peak_height, high=10))

            # Fit
            xopt = scipy.optimize.minimize(err_func, init_params, method=method, bounds=bounds, options={'disp': False})
            return xopt

        xopts = Parallel(n_jobs=n_jobs)(delayed(run_fit)() for i in range(n_runs))
        all_err = [xopt.fun for xopt in xopts]
        all_params = [xopt.x for xopt in xopts]

        # Find best parameters over tested initial conditions
        best_idx = np.argmin(all_err)
        local_best_params = all_params[best_idx]

        parameterized = gen_parameterized_spectrum(freqs, local_best_params)

        err = np.mean(np.power(parameterized - fit_target, 2))

        aic = len(psd) * np.log(err) + 2 * (len(local_best_params) + 1)

        prev_params = local_best_params.tolist()
        print('{} AIC={}'.format(n, aic))
        all_aics.append(aic)
        if n > 0 and aic > all_aics[n - 1]:
            break
        best_params = local_best_params
    return best_params, all_aics


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

        orig_spectrum=power_spectrum
        # If freqs & power_spectrum provided together, add data to object.
        if freqs is not None and power_spectrum is not None:
            self.add_data(freqs, power_spectrum, freq_range)
        # If power spectrum provided alone, add to object, and use existing frequency data
        #   Note: be careful passing in power_spectrum data like this:
        #     It assumes the power_spectrum is already logged, with correct freq_range
        elif isinstance(power_spectrum, np.ndarray):
            self.power_spectrum = power_spectrum
            orig_spectrum=np.power(10,power_spectrum)

        # Check that data is available
        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")

        # Check and warn about width limits (if in verbose mode)
        if self.verbose:
            self._check_width_limits()

        # In rare cases, the model fails to fit, and so uses try / except
        try:
            best_params, all_aics = fit_psd(freqs, orig_spectrum, max_n_peaks=6, alpha=0.3)

            offset = best_params[0]
            slope = best_params[1]
            shift = best_params[2]
            aperiodic = offset + slope * np.log10(1. / (freqs + shift))
            self._ap_fit=aperiodic

            self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._ap_fit)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Find peaks, and fit them with gaussians
            self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

            # Calculate the peak fit
            #   Note: if no peaks are found, this creates a flat (all zero) peak fit
            self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))

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


class FOOOFinatorGroup(FOOOFinator):
    def __init__(self, *args, **kwargs):
        """Initialize object with desired settings."""

        FOOOFinator.__init__(self, *args, **kwargs)

        self.power_spectra = None

        self._reset_group_results()


    def __len__(self):
        """Define the length of the object as the number of model fit results available."""

        return len(self.group_results)


    def __iter__(self):
        """Allow for iterating across the object by stepping across model fit results."""

        for result in self.group_results:
            yield result


    def __getitem__(self, index):
        """Allow for indexing into the object to select model fit results."""

        return self.group_results[index]


    @property
    def has_data(self):
        """Indicator for if the object contains data."""

        return True if np.any(self.power_spectra) else False


    @property
    def has_model(self):
        """Indicator for if the object contains model fits."""

        return True if self.group_results else False


    @property
    def n_peaks_(self):
        """How many peaks were fit for each model."""

        return [f_res.peak_params.shape[0] for f_res in self] if self.has_model else None


    @property
    def n_null_(self):
        """How many model fits are null."""

        return sum([1 for f_res in self.group_results if np.isnan(f_res.aperiodic_params[0])]) \
            if self.has_model else None


    @property
    def null_inds_(self):
        """The indices for model fits that are null."""

        return [ind for ind, f_res in enumerate(self.group_results) \
            if np.isnan(f_res.aperiodic_params[0])] \
            if self.has_model else None


    def _reset_data_results(self, clear_freqs=False, clear_spectrum=False,
                            clear_results=False, clear_spectra=False):
        """Set, or reset, data & results attributes to empty.

        Parameters
        ----------
        clear_freqs : bool, optional, default: False
            Whether to clear frequency attributes.
        clear_spectrum : bool, optional, default: False
            Whether to clear power spectrum attribute.
        clear_results : bool, optional, default: False
            Whether to clear model results attributes.
        clear_spectra : bool, optional, default: False
            Whether to clear power spectra attribute.
        """

        super()._reset_data_results(clear_freqs, clear_spectrum, clear_results)
        if clear_spectra:
            self.power_spectra = None


    def _reset_group_results(self, length=0):
        """Set, or reset, results to be empty.

        Parameters
        ----------
        length : int, optional, default: 0
            Length of list of empty lists to initialize. If 0, creates a single empty list.
        """

        self.group_results = [[]] * length


    def add_data(self, freqs, power_spectra, freq_range=None):
        """Add data (frequencies and power spectrum values) to the current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectra, in linear space.
        power_spectra : 2d array, shape=[n_power_spectra, n_freqs]
            Matrix of power values, in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectra to. If not provided, keeps the entire range.

        Notes
        -----
        If called on an object with existing data and/or results
        these will be cleared by this method call.
        """

        # If any data is already present, then clear data & results
        #   This is to ensure object consistency of all data & results
        if np.any(self.freqs):
            self._reset_data_results(True, True, True, True)
            self._reset_group_results()

        self.freqs, self.power_spectra, self.freq_range, self.freq_res = \
            self._prepare_data(freqs, power_spectra, freq_range, 2, self.verbose)


    def report(self, freqs=None, power_spectra=None, freq_range=None, n_jobs=1, progress=None):
        """Fit a group of power spectra and display a report, with a plot and printed results.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power_spectra, in linear space.
        power_spectra : 2d array, shape: [n_power_spectra, n_freqs], optional
            Matrix of power spectrum values, in linear space.
        freq_range : list of [float, float], optional
            Desired frequency range to run FOOOF on. If not provided, fits the entire given range.
        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel.
            1 is no parallelization. -1 uses all available cores.
        progress : {None, 'tqdm', 'tqdm.notebook'}, optional
            Which kind of progress bar to use. If None, no progress bar is used.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        self.fit(freqs, power_spectra, freq_range, n_jobs=n_jobs, progress=progress)
        self.plot()
        self.print_results(False)


    def fit(self, freqs=None, power_spectra=None, freq_range=None, n_jobs=1, progress=None):
        """Fit a group of power spectra.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power_spectra, in linear space.
        power_spectra : 2d array, shape: [n_power_spectra, n_freqs], optional
            Matrix of power spectrum values, in linear space.
        freq_range : list of [float, float], optional
            Desired frequency range to run FOOOF on. If not provided, fits the entire given range.
        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel.
            1 is no parallelization. -1 uses all available cores.
        progress : {None, 'tqdm', 'tqdm.notebook'}, optional
            Which kind of progress bar to use. If None, no progress bar is used.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        # If freqs & power spectra provided together, add data to object
        if freqs is not None and power_spectra is not None:
            self.add_data(freqs, power_spectra, freq_range)

        # If 'verbose', print out a marker of what is being run
        if self.verbose and not progress:
            print('Running FOOOFGroup across {} power spectra.'.format(len(self.power_spectra)))

        # Run linearly
        if n_jobs == 1:
            self._reset_group_results(len(self.power_spectra))
            for ind, power_spectrum in \
                _progress(enumerate(self.power_spectra), progress, len(self)):
                self._fit(power_spectrum=power_spectrum)
                self.group_results[ind] = self._get_results()

        # Run in parallel
        else:
            self._reset_group_results()
            n_jobs = cpu_count() if n_jobs == -1 else n_jobs
            with Pool(processes=n_jobs) as pool:
                self.group_results = list(_progress(pool.imap(partial(_par_fit, fg=self),
                                                              self.power_spectra),
                                                    progress, len(self.power_spectra)))

        # Clear the individual power spectrum and fit results of the current fit
        self._reset_data_results(clear_spectrum=True, clear_results=True)


    def drop(self, inds):
        """Drop one or more model fit results from the object.

        Parameters
        ----------
        inds : int or array_like of int or array_like of bool
            Indices to drop model fit results for.
            If a boolean mask, True indicates indices to drop.

        Notes
        -----
        This method sets the model fits as null, and preserves the shape of the model fits.
        """

        for ind in check_inds(inds):
            fm = self.get_fooof(ind)
            fm._reset_data_results(clear_results=True)
            self.group_results[ind] = fm.get_results()


    def get_results(self):
        """Return the results run across a group of power spectra."""

        return self.group_results


    def get_params(self, name, col=None):
        """Return model fit parameters for specified feature(s).

        Parameters
        ----------
        name : {'aperiodic_params', 'peak_params', 'gaussian_params', 'error', 'r_squared'}
            Name of the data field to extract across the group.
        col : {'CF', 'PW', 'BW', 'offset', 'knee', 'exponent'} or int, optional
            Column name / index to extract from selected data, if requested.
            Only used for name of {'aperiodic_params', 'peak_params', 'gaussian_params'}.

        Returns
        -------
        out : ndarray
            Requested data.

        Raises
        ------
        NoModelError
            If there are no model fit results available.
        ValueError
            If the input for the `col` input is not understood.

        Notes
        -----
        For further description of the data you can extract, check the FOOOFResults documentation.
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available, can not proceed.")

        # Allow for shortcut alias, without adding `_params`
        if name in ['aperiodic', 'peak', 'gaussian']:
            name = name + '_params'

        # If col specified as string, get mapping back to integer
        if isinstance(col, str):
            col = get_indices(self.aperiodic_mode)[col]
        elif isinstance(col, int):
            if col not in [0, 1, 2]:
                raise ValueError("Input value for `col` not valid.")

        # Pull out the requested data field from the group data
        # As a special case, peak_params are pulled out in a way that appends
        #  an extra column, indicating which FOOOF run each peak comes from
        if name in ('peak_params', 'gaussian_params'):
            out = np.array([np.insert(getattr(data, name), 3, index, axis=1)
                            for index, data in enumerate(self.group_results)])
            # This updates index to grab selected column, and the last column
            #  This last column is the 'index' column (FOOOF object source)
            if col is not None:
                col = [col, -1]
        else:
            out = np.array([getattr(data, name) for data in self.group_results])

        # Some data can end up as a list of separate arrays
        #   If so, concatenate it all into one 2d array
        if isinstance(out[0], np.ndarray):
            out = np.concatenate([arr.reshape(1, len(arr)) \
                if arr.ndim == 1 else arr for arr in out], 0)

        # Select out a specific column, if requested
        if col is not None:
            out = out[:, col]

        return out


    @copy_doc_func_to_method(plot_fg)
    def plot(self, save_fig=False, file_name=None, file_path=None):

        plot_fg(self, save_fig, file_name, file_path)


    @copy_doc_func_to_method(save_report_fg)
    def save_report(self, file_name, file_path=None):

        save_report_fg(self, file_name, file_path)


    @copy_doc_func_to_method(save_fg)
    def save(self, file_name, file_path=None, append=False,
             save_results=False, save_settings=False, save_data=False):

        save_fg(self, file_name, file_path, append, save_results, save_settings, save_data)


    def load(self, file_name, file_path=None):
        """Load FOOOFGroup data from file.

        Parameters
        ----------
        file_name : str
            File to load data from.
        file_path : str, optional
            Path to directory to load from. If None, loads from current directory.
        """

        # Clear results so as not to have possible prior results interfere
        self._reset_group_results()

        power_spectra = []
        for ind, data in enumerate(load_jsonlines(file_name, file_path)):

            self._add_from_dict(data)

            # If settings are loaded, check and update based on the first line
            if ind == 0:
                self._check_loaded_settings(data)

            # If power spectra data is part of loaded data, collect to add to object
            if 'power_spectrum' in data.keys():
                power_spectra.append(data['power_spectrum'])

            # If results part of current data added, check and update object results
            if set(OBJ_DESC['results']).issubset(set(data.keys())):
                self._check_loaded_results(data)
                self.group_results.append(self._get_results())

        # Reconstruct frequency vector, if information is available to do so
        if self.freq_range:
            self._regenerate_freqs()

        # Add power spectra data, if they were loaded
        if power_spectra:
            self.power_spectra = np.array(power_spectra)

        # Reset peripheral data from last loaded result, keeping freqs info
        self._reset_data_results(clear_spectrum=True, clear_results=True)


    def get_fooof(self, ind, regenerate=True):
        """Get a FOOOF object for a specified model fit.

        Parameters
        ----------
        ind : int
            The index of the FOOOFResults in FOOOFGroup.group_results to load.
        regenerate : bool, optional, default: False
            Whether to regenerate the model fits from the given fit parameters.

        Returns
        -------
        fm : FOOOF
            The FOOOFResults data loaded into a FOOOF object.
        """

        # Initialize a FOOOF object, with same settings as current FOOOFGroup
        fm = FOOOFinator(*self.get_settings(), verbose=self.verbose)

        # Add data for specified single power spectrum, if available
        #   The power spectrum is inverted back to linear, as it is re-logged when added to FOOOF
        if self.has_data:
            fm.add_data(self.freqs, np.power(10, self.power_spectra[ind]))
        # If no power spectrum data available, copy over data information & regenerate freqs
        else:
            fm.add_meta_data(self.get_meta_data())

        # Add results for specified power spectrum, regenerating full fit if requested
        fm.add_results(self.group_results[ind])
        if regenerate:
            fm._regenerate_model()

        return fm


    def get_group(self, inds):
        """Get a FOOOFGroup object with the specified sub-selection of model fits.

        Parameters
        ----------
        inds : array_like of int or array_like of bool
            Indices to extract from the object.
            If a boolean mask, True indicates indices to select.

        Returns
        -------
        fg : FOOOFGroup
            The requested selection of results data loaded into a new FOOOFGroup object.
        """

        # Check and convert indices encoding to list of int
        inds = check_inds(inds)

        # Initialize a new FOOOFGroup object, with same settings as current FOOOFGroup
        fg = FOOOFinatorGroup(*self.get_settings(), verbose=self.verbose)

        # Add data for specified power spectra, if available
        #   The power spectra are inverted back to linear, as they are re-logged when added to FOOOF
        if self.has_data:
            fg.add_data(self.freqs, np.power(10, self.power_spectra[inds, :]))
        # If no power spectrum data available, copy over data information & regenerate freqs
        else:
            fg.add_meta_data(self.get_meta_data())

        # Add results for specified power spectra
        fg.group_results = [self.group_results[ind] for ind in inds]

        return fg


    def print_results(self, concise=False):
        """Print out FOOOFGroup results.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_results_fg_str(self, concise))


    def _fit(self, *args, **kwargs):
        """Create an alias to FOOOF.fit for FOOOFGroup object, for internal use."""

        super().fit(*args, **kwargs)


    def _get_results(self):
        """Create an alias to FOOOF.get_results for FOOOFGroup object, for internal use."""

        return super().get_results()

    def _check_width_limits(self):
        """Check and warn about bandwidth limits / frequency resolution interaction."""

        # Only check & warn on first power spectrum
        #   This is to avoid spamming stdout for every spectrum in the group
        if self.power_spectra[0, 0] == self.power_spectrum[0]:
            super()._check_width_limits()
