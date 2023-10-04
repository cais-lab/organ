"""Functions and classes for organization metrics calculation."""
import numpy as np


# Deprecated
class OrganizationMetrics:
    """Utility functions to calculate organization quality and
    validity metrics.

    Organizations are defined by a tuple:
    - vector of node types,
    - matrix of edge types.
    """

    def __init__(self, org_model):
        self.org_model = org_model

        def _valid_lambda(org):
            """
            Checks validity of the organization structure.
            """
            return org is not None and \
                self.org_model.check_nodes(org[0]) and \
                self.org_model.check_relations(org[0], org[1])[0]

        self.valid_lambda = _valid_lambda

    def edge_validness_scores(self, orgs):
        """Estimates egde validness for multiple organizations."""
        def meth(x):
            return self.org_model.check_relations(x[0], x[1])[0]
        return np.array(list(map(meth, orgs)), dtype=np.float32)

    def node_validness_scores(self, orgs):
        """Estimates node validness for multiple organizations."""
        def meth(x):
            return self.org_model.check_nodes(x[0])
        return np.array(list(map(meth, orgs)), dtype=np.float32)

    def valid_scores(self, orgs):
        return np.array(list(map(self.valid_lambda, orgs)),
                        dtype=np.float32)

    def valid_filter(self, orgs):
        return list(filter(self.valid_lambda, orgs))

    def valid_total_score(self, orgs):
        return np.array(list(map(self.valid_lambda, orgs)),
                        dtype=np.float32).mean()

    def sample_organization_metric(self, orgs, norm=False):
        scores = [self.valid_lambda(org) if org is not None
                  else None
                  for org in orgs]
        scores = np.array(scores)

        return scores


class MetricsAggregator:
    """Collects and aggragates validity and quality metrics
    of the generated organization structures.

    Organizations are defined by a tuple:
    - Numpy vector of node types,
    - Numpy matrix of edge types.
    """

    def __init__(self, org_model):
        self.org_model = org_model

        def _valid_lambda(org):
            """
            Checks validity of the organization structure.
            """
            return self.org_model.validness(org)

        self.valid_lambda = _valid_lambda

    def get_scores(self, orgs):
        metric_values = [self.org_model.metrics(org) for org in orgs]
        if not metric_values:
            return {}
        metrics = metric_values[0].keys()
        return {metric: np.array(tuple(v[metric] for v in metric_values),
                                 dtype=np.float32)
                for metric in metrics}

    def valid_scores(self, orgs):
        return np.array(list(map(self.valid_lambda, orgs)),
                        dtype=np.float32)

    def valid_filter(self, orgs):
        return list(filter(self.valid_lambda, orgs))

    def valid_total_score(self, orgs):
        return np.array(list(map(self.valid_lambda, orgs)),
                        dtype=np.float32).mean()

    def sample_organization_metric(self, orgs, norm=False):
        scores = [self.valid_lambda(org) if org is not None
                  else None
                  for org in orgs]
        scores = np.array(scores)

        return scores


def all_scores(metrics_aggregator,
               orgs,
               data,
               norm=False):

    # These are one-value-for-structure scores
    m0 = {k: list(filter(lambda e: e is not None, v))
          for k, v in {
              'Sample score': metrics_aggregator.sample_organization_metric(orgs,       # noqa: E501
                                                                            norm=norm)  # noqa: E501
          }.items()}
    # These are one-value-for-batch scores (used, e.g., in batch log reporting)
    m1 = {k: np.mean(v)
          for k, v in metrics_aggregator.get_scores(orgs).items()
          }
    m1.update({'Accuracy': metrics_aggregator.valid_total_score(orgs)})

    return m0, m1


# Deprecated
def all_scores_(metrics_processor,
                orgs,
                data,
                norm=False,
                reconstruction=False):

    # These are one-value-for-structure scores
    m0 = {k: list(filter(lambda e: e is not None, v))
          for k, v in {
              'Sample score': metrics_processor.sample_organization_metric(orgs,       # noqa: E501
                                                                           norm=norm)  # noqa: E501
          }.items()}
    # These are one-value-for-batch scores (used, e.g., in batch log reporting)
    m1 = {'valid score': metrics_processor.valid_total_score(orgs) * 100,
          'node score': np.mean(metrics_processor.node_validness_scores(orgs)) * 100,  # noqa: E501
          'edge score': np.mean(metrics_processor.edge_validness_scores(orgs)) * 100   # noqa: E501
          }

    return m0, m1
