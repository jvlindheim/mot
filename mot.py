"""
Approximative algorithms for computing sparse Wasserstein-2 multi-marginal optimal transport plans and corresponding free support barycenters.

Johannes von Lindheim, 2021
https://github.com/jvlindheim/mot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ot import emd

"""
Reference algorithm.
"""

def bary_ref(posns, masses, weights=None, shuffle=False, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the reference algorithm.
    Convenience method combining execution of the reference algorithm and construction
    of the correcponding barycenter.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    shuffle=False: If set to True, the indices of all input mass arrays and their corresponding points
        are randomly shuffled.
    precision=7: Support point positions of barycenter are rounded to a grid. This parameter
        indicates the number of decimals to which the positions are rounded.
        
    Returns
    -------
    bary_pos: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    
    posns, masses, n, nis, d = prepare_data(posns, masses, shuffle=shuffle)
    if weights is None:
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"

    bary_tuples, bary_masses = tuples_reference(posns, masses, shuffle=False)
    bary_pos = bary_from_ref_tuples(bary_tuples, bary_masses, posns, masses, weights, precision=precision)
    
    # if the barycenter has multiple Diracs at the same location, merge them back into one
    # in order to check equality between floats without floating point precision issues, round
    bary_pos = np.round(bary_pos, decimals=precision)
    bary_pos, inverse = np.unique(bary_pos, axis=0, return_inverse=True)
    bary_masses = np.bincount(inverse, bary_masses)
    return bary_pos, bary_masses

def tuples_reference(posns, masses, min_bary_mass=1e-10, shuffle=False):
    """
    Approximative multi-marginal optimal transport (MOT) algorithm
    that constructs the MOT using 2-marginal transport plans to a reference measure.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    min_bary_mass: All points with less than this mass will be discarded.
    shuffle=False: If set to True, the indices of all input mass arrays and their corresponding points
        are randomly shuffled.
        
    Returns
    -------
    bary_tuples: (M, N)-shaped array indicating the MOT support tuples.
        Each row contains an index of a support point from each measure.
    bary_masses: MOT masses corresponding to the support tuples.
    """

    posns, masses, n, nis, d = prepare_data(posns, masses, shuffle=shuffle)
    assert np.allclose(np.array([masses_j.sum() for masses_j in masses]), np.ones(n)), "all given measures need to sum to one"
    
    bary_tuples = np.arange(nis[0])[:, None]
    bary_cdf = masses[0].cumsum()
    
    for i, (pos_i, mass_i) in enumerate(zip(posns[1:], masses[1:])):
        
        # compute W2 optimal transport between measure_1 and measure i
        # this could be parallelized
        c_i = cdist(posns[0], pos_i, metric='sqeuclidean')
        pi_i = emd(masses[0], mass_i, c_i)
        assert pi_i.sum() > 0.5, "incorrect result from ot.emd, sum = {0}".format(pi_i.sum())

        # unite transport plans in a splitting-free manner
        cdfs_conc = np.concatenate((bary_cdf, pi_i[pi_i > 0].cumsum()))[:-1]
        args = np.argsort(cdfs_conc)
        bary_inds = np.insert(args[:-1] < len(bary_tuples), 0, 0).cumsum()
        measure_inds = np.insert(args[:-1] >= len(bary_tuples), 0, 0).cumsum()
        bary_tuples = np.concatenate((bary_tuples[bary_inds], pi_i.nonzero()[1][measure_inds][:, None]), axis=1)
        bary_cdf = cdfs_conc[args]
        bary_masses = np.diff(np.insert(bary_cdf, 0, 0))
        
        # remove tuples with zero mass
        bary_tuples = bary_tuples[bary_masses > min_bary_mass]
        bary_cdf = bary_cdf[bary_masses > min_bary_mass]
        
    bary_masses = bary_masses[bary_masses > min_bary_mass]
    return bary_tuples, bary_masses / bary_masses.sum()

def bary_from_ref_tuples(bary_tuples, bary_masses, posns, masses, weights, precision=7):
    """
    From the MOT tuples computed by the reference algorithm, 
    compute the corresponding barycenter.
    
    Parameters
    ----------
    bary_tuples: Solution as given from tuples_reference function:
        (M, N)-shaped array indicating the MOT support tuples.
        Each row contains an index of a support point from each measure.
    bary_masses: MOT masses corresponding to the support tuples.
    posns: Measure support positions list/array of length n of the original measures.
        Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the originally given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
    precision=7: Support point positions of barycenter are rounded to a grid. This parameter
        indicates the number of decimals to which the positions are rounded.
    
    Returns
    -------
    bary_pos: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.    
    """
    
    posns, masses, n, nis, d = prepare_data(posns, masses)
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    bary_pos = np.array([w * pos[bary_tuples[:, i]] for i, (w, pos) in enumerate(zip(weights, posns))]).sum(axis=0)
    return bary_pos

"""
Greedy algorithm.
"""

def bary_greedy(posns, masses, weights=None, shuffle=False, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the greedy algorithm.
    Convenience method combining execution of the greedy algorithm and construction
    of the correcponding barycenter.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
    shuffle=False: If set to True, the indices of all input mass arrays and their corresponding points
        are randomly shuffled.
    precision=7: Support point positions of barycenter are rounded to a grid. This parameter
        indicates the number of decimals to which the positions are rounded.
        
    Returns
    -------
    bary_pos: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    
    posns, masses, n, nis, d = prepare_data(posns, masses, shuffle=shuffle)
    assert np.allclose(np.array([masses_j.sum() for masses_j in masses]), np.ones(n)), "all given measures need to sum to one"

    if weights is None:
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    plans, bary_masses = plans_greedy(posns, masses, weights=weights)
    return bary_from_greedy_plans(plans, bary_masses, posns, masses, weights, precision=precision)

def plans_greedy(posns, masses, weights=None):
    """
    Approximative multi-marginal optimal transport (MOT) algorithm
    that constructs the MOT iteratively, computing plans from a current barycenter
    to the next given measure.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.

    Returns
    -------
    plans: array of transport plans from approximated barycenter to original measure.
    bary_masses: MOT masses corresponding to the support tuples.
    """
    posns, masses, n, nis, d = prepare_data(posns, masses)
    assert np.allclose(np.array([masses_j.sum() for masses_j in masses]), np.ones(n)), "all given measures need to sum to one"
    if weights is None:
        weights = np.ones(n)/n
    else:
        assert len(weights) == n and np.abs(weights.sum()-1).sum() < 1e-10, "number of entries of weights needs to be equal to number of given measures and sum up to 1"

    # init barycenter to the zero point (for W2, any point will do)
    bary_masses = np.array([1.0], dtype=np.float64)
    plans = [np.array([[1.0]])]

    for i, (pos_i, mass_i, ni) in enumerate(zip(posns, masses, nis)):

        # compute the transport between the mot support and current measure
        # costs are sums of indices of existing tuple-entries to indices of i-th measure
        # repeat cost-rows by indexing by their nonzero-cols
        plans_nzcols = [plan.nonzero()[1] for plan in plans]
        tuple_pos = posns[:i] if i > 0 else [np.zeros((1, d), dtype=np.float64)]
        costs = [lambda_j*cdist(pos_j, pos_i, metric='sqeuclidean')[nz_j] for lambda_j, pos_j, nz_j in zip(weights, tuple_pos, plans_nzcols)]

        if i > 0:
            cost = np.sum(costs, axis=0)
            pi_i = emd(bary_masses, mass_i / mass_i.sum(), cost)
            assert pi_i.sum() > 0.5, "incorrect result from ot.emd, sum = {0}".format(pi_i.sum())
        else:
            pi_i = mass_i[None, :]

        # remove zero-rows
        greater_zero_rows = (pi_i.sum(axis=1) > 0.0)
        plans = [plan[greater_zero_rows] for plan in plans]
        bary_masses = bary_masses[greater_zero_rows]
        pi_i = pi_i[greater_zero_rows]

        # if the kantorovich plan wants to split a bary-Dirac (send mass to k>1 different targets),
        # split that mass into k Diracs and send each of them to a unique target accordingly
        nzrows, nzcols = pi_i.nonzero()
        split_rows = (1-np.diff(nzrows, prepend=-1)).astype(bool) # length: new support
        insert_inds = nzrows[split_rows]*pi_i.shape[1] + nzcols[split_rows] # length: new supp - old supp
        # to this end, insert a number of zeros worth one row in the plan whenever there is more
        # than one non-zero entry in that row
        pi_i = np.insert(pi_i.flatten(), np.repeat(insert_inds, ni), 0).reshape(-1, ni)
        bary_masses = pi_i.sum(axis=1) # masses of Diracs correspond to current marginal
        pi_i = (pi_i > 0) # just remember the permutation matrix corresponding to the origin-points
        if i == 0:
            plans[0] = pi_i
        else:
            plans.append(pi_i)
        for j in range(i):
            plans[j] = plans[j][nzrows]

    return plans, bary_masses

def bary_from_greedy_plans(plans, bary_masses, posns, masses, weights, precision=7):
    """
    From the MOT tuples computed by the reference algorithm, 
    compute the corresponding barycenter.
    
    Parameters
    ----------
    plans: array of "permutation plans" from approximated barycenter to original measure.
        That is, the transport plans, where the entries are only ones.
    bary_masses: MOT masses corresponding to the support tuples.
    posns: Measure support positions list/array of length n of the original measures.
        Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the originally given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
    precision=7: Support point positions of barycenter are rounded to a grid. This parameter
        indicates the number of decimals to which the positions are rounded.
        
    Returns
    -------
    bary_pos: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    
    posns, masses, n, nis, d = prepare_data(posns, masses)

    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    bary_pos = np.sum([w_i * plan.dot(pos) for (w_i, plan, pos) in zip(weights, plans, posns)], axis=0)
            
    # if the barycenter has multiple Diracs at the same location, merge them back into one
    # in order to check equality between floats without floating point precision issues, round
    bary_pos = np.round(bary_pos, decimals=precision)
    bary_pos, inverse = np.unique(bary_pos, axis=0, return_inverse=True)
    bary_masses = np.bincount(inverse, bary_masses)

    return bary_pos, bary_masses

"""
Helper functions.
"""

def prepare_data(posns, masses, min_mass=1e-10, shuffle=False):
    '''
    Given a support positions and masses array, determine (and make security checks for) the number of measures, number
    of support points array and dimension. Also modify posns to array, if given only one array for all measures.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
    shuffle=False: If set to True, the indices of all input mass arrays and their corresponding points
        are randomly shuffled.
        
    Returns
    -------
    posns: Measure support positions list of length n.
    masses: Masses/weights of the given measure support points.
    n: Number of determined measures.
    nis: Number of determined support points for each measure.
    d: Dimension of the support points.    
    '''    
    # if given a list, we assume that we are given multiple measures and the length of the list is
    # the number of measures
    if isinstance(posns, list):
        n = len(posns)
        assert len(masses) == n, "masses needs have same length as pos (equal number of measures)"
    # if given a 2d-array and a list of of mass arrays, we assume that we are given a number of measures,
    # which are all supported on the same posns-array, so the number of measures n is len(masses)
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and (isinstance(masses, list) \
                                                                or (isinstance(masses, np.ndarray) and masses.ndim == 2)):
        n = len(masses)
        posns = [posns]*n
    # if given a 3d-array and a 2d array of masses, we assume that we are given a number of measures that
    # all have the same number of points
    elif isinstance(posns, np.ndarray) and posns.ndim == 3:
        n = posns.shape[0]
        assert masses.shape[0] == n, "masses needs have same length as pos (equal number of measures)"
        assert posns.shape[1] == masses.shape[1], "number of points and number of mass entries need to be the same"

    # if given a 2d-array and a 1d-array of masses, assume that we are given only one measure
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and isinstance(masses, np.ndarray) and masses.ndim == 1:
        n = 1
        assert len(posns) == len(masses), "if only one measure is given, length of posns and masses need to match"
        posns = [posns]
        masses = masses[None, :]
    else:
        raise ValueError("cannot see what the number of measures is for given parameters 'posns' and 'masses'")
    assert n >= 1, "at least one measure needs to be given"
    assert all([pos.ndim == 2 for pos in posns]), "position arrays need to be two-dimensional"
    posns = [pos[mass > min_mass] for (pos, mass) in zip(posns, masses)] # throw out points with mass <= min_mass
    masses = [mass[mass > min_mass] for mass in masses]
    nis = np.array([pos.shape[0] for pos in posns]) # number of support points for all measures
    d = posns[0].shape[1]
    perms = [np.random.permutation(ni) for ni in nis] if shuffle else [np.arange(ni) for ni in nis]
    posns = [pos[perm] for (pos, perm) in zip(posns, perms)]
    masses = [mass[perm] for (mass, perm) in zip(masses, perms)]
    return posns, masses, n, nis, d

def scatter_distr(posns, masses, n_plots_per_row=2, scale=4, invert=False, disk_size=6000/5, xmarkers=False, color='gray',
           alpha=0.5, xmarker_posns=None, axis_off=False, margin_fac=0.2, dpi=300,
           subtitles='', figax=None, savepath=None):
    '''
    Scatter plot function for either one or multiple discrete probability distributions.

    Parameters
    ----------
    posns: Measure support positions list/array of length 1 or n.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    n_plots_per_row=2: In case multiple measures are given, a figure with this number of
        subplots per column is generated.
    scale=4: Size of the figure is proportional to this parameter.
    invert=False: Whether to invert the y-axis.
    disk_size=6000/5: Size of the plotted disks per support point is proportional
        to their weight and this parameter.
    xmarkers=False: Whether to plot an x in the center of each disk (support point).
    color='gray': Color of each disk (support point).
    alpha=0.5: Transparency of each disk (support point).
    xmarker_posns=None: An additional set of x-markers can be plotted if this
        parameter is given an array of 2d positions.
    axis_off=False: Whether to turn off the coordinate system of the subplots.
    margin_fac=0.2: How much margin to leave around the minimum and maximum x/y-values
        of the support points.
    dpi=300: Resolution of figure (important for export).
    subtitles='': Array of subtitles to each subplot.
    figax=None: Tuple of the form (fig, ax, k, l). This can be used, if this function
        is only supposed to plot in the given axis array 'ax' at index k, l that
        has already been constructed.
        If None is given, a new fig and axis array are created.
    savepath=None: Saves figure to this given path.
    '''
    posns, masses, n, nis, d = prepare_data(posns, masses)
    n_plots = n
    if figax is None:
        n_rows = np.ceil(n_plots / n_plots_per_row).astype(int)
        n_cols = min(n_plots_per_row, n_plots)
        figsize = (scale*n_cols, scale*n_rows)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    else:
        fig, ax, k, l = figax

    if isinstance(subtitles, list) and len(subtitles) == 1:
        subtitles = [subtitles[0]]*n_plots
    elif isinstance(subtitles, str):
        subtitles = [subtitles]*n_plots
    else:
        assert len(subtitles) == n_plots, "length of subtitles array needs to be equal to number of plots"

    xmin, xmax, ymin, ymax = min([pos[:, 0].min() for pos in posns]), max([pos[:, 0].max() for pos in posns]), min([pos[:, 1].min() for pos in posns]), max([pos[:, 1].max() for pos in posns])

    if figax is None:
        row_inds, col_inds = range(n_rows), range(n_cols)
    else:
        row_inds, col_inds = [k], [l]
    for i in row_inds:
        for j in col_inds:
            idx = i*n_plots_per_row + j if figax is None else 0
            if idx >= n_plots or axis_off:
                ax[i, j].axis('off')
                if idx >= n_plots:
                    continue
            pos = posns[idx]
            mass = masses[idx]

            # set plot dimensions
            xmargin = margin_fac*(xmax-xmin)
            ymargin = margin_fac*(ymax-ymin)
            ax[i, j].set_xlim([xmin-xmargin, xmax+xmargin])
            ax[i, j].set_ylim([ymin-ymargin, ymax+ymargin])
            ax[i, j].set_aspect('equal')
            ax[i, j].set_title(subtitles[idx])

            # plot
            if xmarkers:
                ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='x', c='red')
            if xmarker_posns is not None:
                ax[i, j].scatter(xmarker_posns[:, 0], xmarker_posns[:, 1], marker='x', c='red')
            ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='o', s=mass*disk_size*scale, c=color, alpha=alpha)
            if invert:
                ax[i, j].set_ylim(ax[i, j].get_ylim()[::-1])

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, pad_inches=0, bbox_inches='tight')
