import pymc
import numpy
import pandas as pd


class PhysicalLogDModel(object):
    """
    Physical description of the experiment, including conversion from MRM to concentration.
    """

    def __init__(self, df, stock_concentration=10, stock_volume=10, chx_dilution_factor=0.1, chx_volume=500, pbs_volume=500, dilution=False):
        """

        Parameters
        ----------
        df - pd.DataFrame
            Dataframe from preprocessing
        stock_concentration - float
            concentration of dmso stock in mM
        stock_volume - float
            volume of dmso stock used, in uL (initial guess)
        chx_dilution_factor - float
            factor of dilution for cyclohexane into octanol as prep for ms step
        chx_volume - float
            volume of the chx phase in uL
        pbs_volume - float
            volume of the buffer phase in uL
        dilution - bool
            Placeholder for possible dilution parameter
        Notes
        -----
        For internal concistency, specify all volumes in uL, all concentrations in mM

        Returns
        -------

        """
        # TODO sampl_67 is the same internal standard, adjust concentrations/volumes accordingly

        self.model = dict()
        self.chx_volume = chx_volume
        self.pbs_volume = pbs_volume
        self.stock_concentration = stock_concentration
        sigma_guess = 0.1

        # Steady hand metrics for Bas' right hand
        # Note: Assume no left hand pipetting operations
        # self.model["Log Sigma volume"] = pymc.Uniform("Log Sigma volume", -4., 4., numpy.log(dispense_sigma_guess))

        # Measurement error
        self.model["logsigma_ms_chx"] = pymc.Uniform("logsigma_ms_chx", -4., 4., numpy.log(sigma_guess))
        self.model["logsigma_ms_pbs"] = pymc.Uniform("logsigma_ms_pbs", -4., 4., numpy.log(sigma_guess))

        # Every compound
        for compound, compound_group in df.groupby("Sample Name"):
            # Get initial guesses
            mean_counts = dict()
            mean_concentrations = dict()

            for phase, phase_group in compound_group.groupby("Solvent"):
                phase = phase.lower()
                mean_counts[phase] = phase_group["Area/Volume"].mean()

            mean_logd = numpy.log10(mean_counts['chx']/mean_counts['pbs'])
            mean_concentrations["pbs"] = self._pbs_conc(self.chx_volume, mean_logd, self.pbs_volume, stock_volume, self.stock_concentration)
            mean_concentrations["chx"] = self._chx_conc(self.chx_volume, mean_logd, self.pbs_volume, stock_volume, self.stock_concentration)
            avg_log_mrm_factor = numpy.log((mean_counts["chx"]+ mean_counts["pbs"])/(mean_concentrations["chx"] + mean_concentrations["pbs"]))

            logd_name = "LogD_{}".format(compound)
            fragmentation_param_name = "log_MRM_{}".format(compound)

            self.model[logd_name] = pymc.Uniform(logd_name, lower=-10, upper=10, value=mean_logd)
            self.model[fragmentation_param_name] = pymc.Uniform(fragmentation_param_name, lower=0.0, upper=numpy.log(1.e8/0.2), value=avg_log_mrm_factor)

            for (batch, repeat), repeat_group in compound_group.groupby(["Set", "Repeat"]):

                # One pipetting operation per repeat experiment
                # mu = self._mu_lognorm(stock_volume, self.model["Log Sigma volume"])
                # tau = self._tau_lognorm(stock_volume, self.model["Log Sigma volume"])

                # TODO remove artificial sigma constraint. (made up)
                mu_vol = self._mu_lognorm(stock_volume, pymc.log(0.1*stock_volume))
                tau_vol = self._tau_lognorm(stock_volume, pymc.log(0.1*stock_volume))
                vol_parameter_name = 'vol_{0}_{1}-{2}'.format(compound, batch, repeat)
                self.model[vol_parameter_name] = pymc.Lognormal(vol_parameter_name, mu=mu_vol, tau=tau_vol, value=stock_volume)

                for phase, phase_group in repeat_group.groupby("Solvent"):
                    phase = phase.lower()
                    if phase == "pbs":
                        pbs_concentration = pymc.exp(self.model[fragmentation_param_name]) * self._pbs_conc(self.chx_volume,
                                                                                                         self.model[logd_name],
                                                                                                         self.pbs_volume,
                                                                                                         self.model[vol_parameter_name],
                                                                                                         self.stock_concentration)
                        mu = self._mu_lognorm(pbs_concentration, pymc.log(pbs_concentration) + self.model["logsigma_ms_pbs"])
                        tau = self._tau_lognorm(pbs_concentration, pymc.log(pbs_concentration) + self.model["logsigma_ms_pbs"])
                        conc_name = "pbs_{0}_{1}-{2}".format(compound, batch, repeat)
                        self.model[conc_name] = pymc.Lambda(conc_name, lambda x=pbs_concentration: x)
                    elif phase == "chx":
                        chx_concentration = pymc.exp(self.model[fragmentation_param_name]) * self._chx_conc(self.chx_volume,
                                                                                                         self.model[logd_name],
                                                                                                         self.pbs_volume,
                                                                                                         self.model[vol_parameter_name],
                                                                                                         self.stock_concentration)

                        mu = self._mu_lognorm(chx_concentration, pymc.log(chx_concentration) + self.model["logsigma_ms_chx"])
                        tau = self._tau_lognorm(chx_concentration, pymc.log(chx_concentration) + self.model["logsigma_ms_chx"])
                        conc_name = "chx_{0}_{1}-{2}".format(compound, batch, repeat)
                        self.model[conc_name] = pymc.Lambda(conc_name, lambda x=chx_concentration: x)

                    else:
                        raise ValueError("Unknown phase: {}".format(phase))

                    # likelihood of each observation
                    for replicate, repl_group in repeat_group.groupby("Replicate"):
                        replicate_parameter_name = 'C{0}_{1}_{2}-{3}_{4}'.format(compound, phase, batch, repeat, replicate)
                        # Extract the observed concentration
                        self.model[replicate_parameter_name] = pymc.Lognormal(replicate_parameter_name, mu=mu, tau=tau, observed=True, value=repl_group["Area/Volume"])

    def _chx_conc(self, v_chx, logd, v_pbs, v_stock, c_stock):
        return (v_stock * c_stock) / (pow(10.0, -logd) * v_pbs + v_chx)

    def _pbs_conc(self, v_chx, logd, v_pbs, v_stock, c_stock):
        return (v_stock * c_stock) / (pow(10.0, logd) * v_chx + v_pbs)



    def _mu_lognorm(self, mu, logsigma):
        """
        Transform a gaussian mu to the lognormal mu
        Parameters
        ----------
        mu - float
            the mean of a gaussian variable
        logsigma - float
            sigma of a gaussian variable

        Returns
        -------
        float  mu
        """
        sigma = pymc.exp(logsigma)
        return pymc.log(mu**2 / pymc.sqrt(sigma**2 + mu**2))

    def _tau_lognorm(self, mu, logsigma):

        """
        Get lognormal tau from gaussian parameters

        Parameters
        ----------
        mu - float
            the mean of a gaussian variable
        logsigma - float
            sigma of a gaussian variable


        Returns
        -------

        """
        sigma = pymc.exp(logsigma)
        return pymc.sqrt(pymc.log(1.0 + (sigma/mu)**2))**(-2)
