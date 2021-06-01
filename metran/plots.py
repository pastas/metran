"""This module contains the Plot helper class for Metran."""
import matplotlib.pyplot as plt
import numpy as np
from pandas import Timestamp
from pastas.plots import _get_height_ratios


class MetranPlot:
    """Plots available directly from the Metran Class."""

    def __init__(self, mt):
        self.mt = mt

    def scree_plot(self):
        """Draw a scree plot of the eigenvalues.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            plot axis handle
        """

        n_ev = np.arange(self.mt.eigval.shape[0]) + 1
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(n_ev, self.mt.eigval, marker="o", ms=7, mfc="none", c="C3")
        ax.bar(n_ev, self.mt.eigval, facecolor="none",
               edgecolor="C0", linewidth=2)
        ax.grid(b=True)
        ax.set_xticks(n_ev)
        ax.set_ylabel("eigenvalue")
        ax.set_xlabel("eigenvalue number")
        fig.tight_layout()
        return ax

    def state_means(self, tmin=None, tmax=None, adjust_height=True):
        """Plot all specific and common smoothed state means.

        Parameters
        ----------
        tmin : str or pd.Timestamp, optional
            start time, by default None
        tmax : str or pd.Timestamp, optional
            end time, by default None
        adjust_height : bool, optional
            scale y-axis of plots relative to one another, by default True

        Returns
        -------
        axes : list of matplotlib.pyplot.Axes
            list of axes handles
        """

        # Get all smoothed state means
        states = self.mt.get_state_means()

        if tmin is None:
            tmin = states.index[0]
        if tmax is None:
            tmax = states.index[-1]

        ylims = []

        if adjust_height:
            for s in states:
                hs = states.loc[tmin:tmax, s]
                if hs.empty:
                    if s.empty:
                        ylims.append((0.0, 0.0))
                    else:
                        ylims.append((s.min(), hs.max()))
                else:
                    ylims.append((hs.min(), hs.max()))
            hrs = _get_height_ratios(ylims)
        else:
            hrs = [1] * (states.columns.size)

        fig = plt.figure(figsize=(10, states.columns.size * 2))
        gs = fig.add_gridspec(
            ncols=1, nrows=states.columns.size, height_ratios=hrs)

        for i, col in enumerate(states.columns):
            if i == 0:
                iax = fig.add_subplot(gs[i])
                ax0 = iax
            else:
                iax = fig.add_subplot(gs[i], sharex=ax0)

            if col.startswith("cdf"):
                c = "C3"
                lbl = f"common dynamic factor {col[3:]}"
            else:
                c = "C0"
                lbl = f"specific dynamic factor {col.replace('_sdf', '')}"

            states.loc[:, col].plot(ax=iax, label=lbl, color=c)
            iax.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)
            iax.grid(b=True)

            if adjust_height:
                iax.set_ylim(ylims[i])

        iax.set_xlabel("")
        fig.tight_layout()
        return fig.axes

    def simulation(self, name, alpha=0.05, tmin=None, tmax=None, ax=None):
        """Plot simulation for single oseries.

        Parameters
        ----------
        name : str
            name of the oseries
        alpha : float, optional
            confidence interval statistic, by default 0.05 (95% confidence
            interval), if None no confidence interval is shown.
        tmin : str or pd.Timestamp, optional
            start time, by default None
        tmax : str or pd.Timestamp, optional
            end time, by default None
        ax : matplotlib.pyplot.Axis
            axes to plot simulation on, if None (default) create a new figure

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            plot axis handle
        """

        sim = self.mt.get_simulation(name, alpha=alpha)
        obs = self.mt.get_observations(
            standardized=False, masked=self.mt.kf.mask).loc[:, name]

        if tmin is None:
            tmin = sim.index[0]
        else:
            tmin = Timestamp(tmin)
        if tmax is None:
            tmax = sim.index[-1]
        else:
            tmax = Timestamp(tmax)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(sim.index, sim["mean"], label=f"simulation {name}")
        ax.plot(obs.index, obs, marker=".", ms=3, color="k",
                ls='none', label="observations")
        if alpha is not None:
            ax.fill_between(sim.index, sim["lower"], sim["upper"], color="gray",
                            alpha=0.5, label="95%-confidence interval")
        ax.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)
        ax.grid(b=True)
        ax.set_xlim(tmin, tmax)

        if ax is None:
            fig.tight_layout()

        return ax

    def simulations(self, alpha=0.05, tmin=None, tmax=None):
        """Plot simulations for all oseries.

        Parameters
        ----------
        name : str
            name of the oseries
        alpha : float, optional
            confidence interval statistic, by default 0.05 (95% confidence
            interval), if None no confidence interval is shown.
        tmin : str or pd.Timestamp, optional
            start time, by default None
        tmax : str or pd.Timestamp, optional
            end time, by default None
        ax : matplotlib.pyplot.Axis
            axes to plot simulation on, if None (default) create a new figure

        Returns
        -------
        axes : list of matplotlib.pyplot.Axes
            list of axes handles
        """
        nrows = len(self.mt.snames)
        fig, axes = plt.subplots(nrows, 1, sharex=True,
                                 sharey=True, figsize=(10, nrows * 2))

        for i, name in enumerate(self.mt.snames):
            self.simulation(name, alpha=alpha, tmin=tmin, tmax=tmax,
                            ax=axes.flat[i])
        fig.tight_layout()
        return axes

    def decomposition(self, name, tmin=None, tmax=None, ax=None,
                      split=False, adjust_height=True, **kwargs):
        """Plot decomposition into specific and common dynamic components.

        Parameters
        ----------
        name : str
            name of oseries
        tmin : str or pd.Timestamp, optional
            start time, by default None
        tmax : str or pd.Timestamp, optional
            end time, by default None
        ax : matplotlib.pyplot.Axis
            axes to plot decomposition on
        split : bool, optional
            plot specific and common dynamic factors on different axes,
            only if ax is None
        adjust_height : bool, optional
            scale y-limits of axes relative to one another, by default True,
            only used when ax is None and split=True


        Returns
        -------
        axes : list of matplotlib.pyplot.Axes
            list of axes handles
        """
        decomposition = self.mt.decompose_simulation(name, **kwargs)
        if tmin is None:
            tmin = decomposition.index[0]
        if tmax is None:
            tmax = decomposition.index[-1]

        # logic for height ratios if splitting decomposition into subplots
        if ax is None:
            ylims = []
            if adjust_height and split:
                for s in decomposition:
                    hs = decomposition.loc[tmin:tmax, s]
                    if hs.empty:
                        if s.empty:
                            ylims.append((0.0, 0.0))
                        else:
                            ylims.append((s.min(), hs.max()))
                    else:
                        ylims.append((hs.min(), hs.max()))
                hrs = _get_height_ratios(ylims)
            elif split:
                hrs = [1] * (decomposition.columns.size)
            else:
                hrs = [1]

            if split:
                fig = plt.figure(figsize=(10, 6))
                nrows = decomposition.columns.size
            else:
                fig = plt.figure(figsize=(10, 4))
                nrows = 1
            gs = fig.add_gridspec(
                ncols=1, nrows=nrows, height_ratios=hrs)

        cdfcount = 0  # color counter for common dynamic factors

        for i, col in enumerate(decomposition.columns):

            if i == 0:
                if ax is None:
                    iax = fig.add_subplot(gs[i])
                    ax0 = iax
                else:
                    iax = ax
            elif i > 0 and split and ax is None:
                iax = fig.add_subplot(gs[i], sharex=ax0)

            # line color
            if col.startswith("cdf"):
                c = f"C{3 + cdfcount % 10}"
                cdfcount += 1
                zorder = 2
            else:
                c = "C0"
                zorder = 3

            # plot
            s = decomposition[col]
            iax.plot(s.index, s, label=f"{col} {name}", color=c, zorder=zorder)

            # grid and legend
            iax.grid(b=True)
            iax.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)

            # set ylimits
            if adjust_height and split and ax is None:
                iax.set_ylim(ylims[i])

        if ax is None:
            fig.tight_layout()

        return iax.figure.axes

    def decompositions(self, tmin=None, tmax=None, **kwargs):
        """Plot all decompositions into specific and common dynamic components.

        Parameters
        ----------
        name : str
            name of oseries
        tmin : str or pd.Timestamp, optional
            start time, by default None
        tmax : str or pd.Timestamp, optional
            end time, by default None

        Returns
        -------
        axes : list of matplotlib.pyplot.Axes
            list of axes handles
        """
        nrows = len(self.mt.snames)
        fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True,
                                 figsize=(10, nrows * 2))

        for i, name in enumerate(self.mt.snames):
            self.decomposition(name, tmin=tmin, tmax=tmax, ax=axes.flat[i],
                               split=False, adjust_height=False, **kwargs)

        fig.tight_layout()
        return axes
