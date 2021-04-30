import matplotlib.pyplot as plt
import numpy as np
from pandas import Timestamp
from pastas.plots import _get_height_ratios


class MetranPlot:
    """Plots available directly from the Metran Class."""

    def __init__(self, mt):
        self.mt = mt

    def scree_plot(self):

        n_ev = np.arange(self.mt.eigval.shape[0])
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(n_ev, self.mt.eigval, marker="o", ms=7, mfc="none", c="C3")
        ax.bar(n_ev, self.mt.eigval, facecolor="none",
               edgecolor="C0", linewidth=2)
        ax.grid(b=True)
        ax.set_ylabel("eigenvalue")
        ax.set_xlabel("eigenvalue number")
        fig.tight_layout()
        return ax

    def states(self, tmin=None, tmax=None, adjust_height=True):

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

        fig = plt.figure(figsize=(10, 10))
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
                lbl = f"specific dynamic factor {col[3:]}"

            states.loc[:, col].plot(ax=iax, label=lbl, color=c)
            iax.legend(loc="upper right")
            iax.grid(b=True)

            if adjust_height:
                iax.set_ylim(ylims[i])

        iax.set_xlabel("")
        fig.tight_layout()
        return fig.axes

    def simulation(self, name, alpha=0.05, tmin=None, tmax=None, ):

        sim = self.mt.get_simulation(name, alpha=alpha)
        obs = self.mt.get_observations(
            standardized=False, masked=False).loc[:, name]

        if tmin is None:
            tmin = sim.index[0]
        else:
            tmin = Timestamp(tmin)
        if tmax is None:
            tmax = sim.index[-1]
        else:
            tmax = Timestamp(tmax)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(sim.index, sim["mean"], label=f"simulation {name}")
        ax.plot(obs.index, obs, marker=".", ms=3, color="k",
                ls='none', label="observations")
        if alpha is not None:
            ax.fill_between(sim.index, sim["lower"], sim["upper"], color="gray",
                            alpha=0.5, label="95%-confidence interval")
        ax.legend(loc="best", numpoints=3, ncol=3)
        ax.grid(b=True)
        ax.set_ylabel("head (m+ref)")
        ax.set_xlim(tmin, tmax)
        fig.tight_layout()

        return ax

    def decomposition(self, name, tmin=None, tmax=None, adjust_height=True,
                      **kwargs):

        decomposition = self.mt.decompose_simulation(name, **kwargs)
        if tmin is None:
            tmin = decomposition.index[0]
        if tmax is None:
            tmax = decomposition.index[-1]

        ylims = []

        if adjust_height:
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
        else:
            hrs = [1] * (decomposition.columns.size)

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(
            ncols=1, nrows=decomposition.columns.size, height_ratios=hrs)

        for i, col in enumerate(decomposition.columns):
            if i == 0:
                iax = fig.add_subplot(gs[i])
                ax0 = iax
            else:
                iax = fig.add_subplot(gs[i], sharex=ax0)

            if col.startswith("cdf"):
                c = "C3"
            else:
                c = "C0"
            s = decomposition[col]
            iax.plot(s.index, s, label=col, color=c)
            iax.grid(b=True)
            iax.legend(loc="upper right")

            if adjust_height:
                iax.set_ylim(ylims[i])
        fig.tight_layout()
        return fig.axes
