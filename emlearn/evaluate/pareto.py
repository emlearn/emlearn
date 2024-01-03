
"""
Pareto-optimal evaluation
=========================
"""

import numpy

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points (smaller is better)
    
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    
    From https://stackoverflow.com/a/40239615/1967571
    Fairly fast for many datapoints, less fast for many costs, somewhat readable
    """
    is_efficient = numpy.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = numpy.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def find_pareto_front(df,
    cost_metric : str = 'mean_test_compute',
    performance_metric : str = 'mean_test_accuracy',
    higher_is_better : bool = True,
    min_performance=None):    
    """
    Find the Pareto front

    :param cost_metric: Column with model compute cost. Lower cost always better
    :param performance_metric: Column with model predictive performance.
    :param higher_is_better: Whether higher or lower is better for @performance_metric
    :param min_performance: Cut datapoints with worse performance than this

    :returns: The rows that make up the Pareto front
    """

    # prevnt mutating input    
    df = df.copy()

    # flip to negative-is-better
    if higher_is_better:
        df['mean_test_score'] = -df[performance_metric]
    else:
        df['mean_test_score'] = df[performance_metric]

    pp = is_pareto_efficient_simple(df[['mean_test_score', cost_metric]].values)
    
    if min_performance is not None:
        if higher_is_better:
            pp = (pp & (df[performance_metric] >= min_performance))
        else:
            pp = (pp & (df[performance_metric] <= min_performance))
    
    return df[pp]

def plot_pareto_front(results,
                      pareto_cut=None,
                      plot_other=True,
                      plot_pareto=True,
                      hue=None,
                      pareto_alpha=0.8,
                      other_alpha=0.3,
                      pareto_global=False,
                      s=100,
                      pareto_s=5,
                      height=8,
                      aspect=1,
                      cost_metric='mean_test_compute',
                      performance_metric='mean_test_accuracy',
                      size_metric='mean_test_size'):
    """
    Utility for plotting performance vs compute cost and size of a model.
    
    Can also compute and plot the pareto front. 
    """

    import seaborn

    pf = find_pareto_front(results, min_performance=pareto_cut)
    pf = pf.sort_values(cost_metric)

    g = seaborn.FacetGrid(results, hue=hue, height=height, aspect=aspect)

    # plot all data
    if plot_other:
        g.map_dataframe(seaborn.scatterplot,
                        x=cost_metric,
                        y=performance_metric,
                        size=size_metric,
                        legend=False,
                        s=pareto_s*s,
                        alpha=other_alpha,
                        #hue=hue,
       )

    def _plot_front(color, label):
        if pareto_global:
            sub = pf.copy()
        else:        
            sub = pf[pf[hue] == label]        

        seaborn.lineplot(
            data=sub,
            x=cost_metric,
            y=performance_metric,
            #color=color,
            label=label,
            alpha=pareto_alpha,
            legend=True,
        )

    # plot the data along Pareto front 
    if plot_pareto:
        g.map(_plot_front)

    return g



