# Import relevant modules
import pymc
import numpy
import pandas as pd


class LogDModel(object):

    def __init__(self, df):
        """

        Parameters
        ----------
        df - pandas dataframe

        Returns
        -------

        """
        assert type(df) == pd.DataFrame

        self.logd = dict()
        sigma_guess = 0.2
        logsigma_chx = pymc.Uniform("Sigma cyclohexane", -4., 4., numpy.log(sigma_guess))
        logsigma_pbs = pymc.Uniform("Sigma buffer", -4., 4., numpy.log(sigma_guess))
        logsigma_ms_chx = pymc.Uniform("Sigma MS cyclohexane", -4., 4., numpy.log(sigma_guess))
        logsigma_ms_pbs = pymc.Uniform("Sigma MS buffer", -4., 4., numpy.log(sigma_guess))
        self.model = dict(logsigma_chx=logsigma_chx, logsigma_pbs=logsigma_pbs, logsigma_ms_chx=logsigma_ms_chx, logsigma_ms_pbs=logsigma_ms_pbs)

        # Every compound
        for compound, compound_group in df.groupby("Sample Name"):

            # Concentration in each solvent phase
            for phase, phase_group in compound_group.groupby("Solvent"):
                phase = phase.lower()
                parameter_name = 'log10_{0}_{1}'.format(compound, phase)
                mean_concentration = phase_group["Area/Volume"].mean()
                # logsig = numpy.log(phase_group["Area/Volume"].std())

                min_concentration = 1/2.0
                max_concentration = 1.e8
                # The log10 of the concentration is modelled with a uniform prior
                self.model[parameter_name] = pymc.Uniform(parameter_name, lower=numpy.log10(min_concentration), upper=numpy.log10(max_concentration), value=numpy.log10(mean_concentration))

                # Corresponds to independent repeats
                for (batch, repeat), repeat_group in phase_group.groupby(["Set", "Repeat"]):
                    repeat_parameter_name = '{0}_{1}_{2}-{3}'.format(compound, phase, batch, repeat)
                    mu = pymc.Lambda(repeat_parameter_name + "-MU", lambda mu=pow(10.0, self.model[parameter_name]), ls=pow(10.0,self.model[parameter_name])*pymc.exp(self.model["logsigma_{}".format(phase)]): self._mu_lognorm(mu, ls))
                    tau = pymc.Lambda(repeat_parameter_name + "-TAU", lambda mu=pow(10.0,self.model[parameter_name]), ls=pow(10.0,self.model[parameter_name])*pymc.exp(self.model["logsigma_{}".format(phase)]): self._tau_lognorm(mu, ls))
                    # True concentration of independent repeats
                    self.model[repeat_parameter_name] = pymc.Lognormal(repeat_parameter_name, mu=mu, tau=tau, value=mean_concentration)

                    # likelihood of each observation
                    for replicate, repl_group in repeat_group.groupby("Replicate"):
            
                        replicate_parameter_name = '{0}_{1}_{2}-{3}_{4}'.format(compound, phase, batch, repeat, replicate)
                        # Extract the observed concentration

                        assert len(repl_group) == 1 # failsafe
                        value = repl_group["Area/Volume"]
                        mu = pymc.Lambda(replicate_parameter_name + "-MU", lambda mu=self.model[repeat_parameter_name], ls=self.model[repeat_parameter_name]*pymc.exp(self.model["logsigma_ms_{}".format(phase)]): self._mu_lognorm(mu, ls))
                        tau = pymc.Lambda(replicate_parameter_name + "-TAU", lambda mu=self.model[repeat_parameter_name], ls=self.model[repeat_parameter_name]*pymc.exp(self.model["logsigma_ms_{}".format(phase)]): self._tau_lognorm(mu, ls))
                        # Observed concentration from replicate experiment
                        self.model[replicate_parameter_name] = pymc.Lognormal(replicate_parameter_name, mu=mu, tau=tau, value=value, observed=True) #, value=1.0)

            self.logd[compound] = pymc.Lambda("LogD_{}".format(compound), lambda c=self.model["log10_{}_chx".format(compound)], p=self.model["log10_{}_pbs".format(compound)]: c-p)

    def _mu_lognorm(self, mu, sigma):
        """
        Transform a gaussian mu to the lognormal mu
        Parameters
        ----------
        mu - float
            the mean of a gaussian variable
        sigma - float
            sigma of a gaussian variable

        Returns
        -------
        float  mu
        """
        # sigma = numpy.exp(logsigma)
        return numpy.log(mu**2 / numpy.sqrt(sigma**2 + mu**2))

    def _tau_lognorm(self, mu, sigma):

        """
        Get lognormal tau from gaussian parameters

        Parameters
        ----------
        mu - float
            the mean of a gaussian variable
        sigma - float
            sigma of a gaussian variable


        Returns
        -------

        """
        # sigma = numpy.exp(logsigma)
        return numpy.sqrt(numpy.log(1.0 + (sigma/mu)**2))**(-2)


