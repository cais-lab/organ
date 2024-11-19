"""
Module implementing tools intended for multilevel systems
configuration of systems, e.g., when one need to configure
upper-level structure which, in its turn, sets certain
restrictions for the lower-level, and so on.
"""
from collections.abc import Callable

import numpy as np

from organ.structure.models import Organization 


class OrganizationConfigurationConflict(Exception):
    pass

class Configurator:
    """Generates complex (multi-level) structures according to the specified
    rules and using the specified set of generators."""
    
    N_PROBES: int = 10
    
    def __init__(self, generators: dict,
                       dependencies: dict[tuple[str, str], Callable],
                       sequence: list[str]):
        """
        Constructor.

        Parameters
        ----------
        generators: dict
            Dict with generators, mapping structure name to a generator instance.
        dependencies: dict
            Dict with dependency specifications, mapping pairs of structure names
            to a function, defining input parameters of the underlying model.
        sequence: list
            List, defining the generation sequence.
        """
        self.generators = generators
        self.dependencies = dependencies or {}
        self.sequence = sequence

    def _make_condition(self, org: dict, aspect_name: str, condition: np.array =None):
        for (a1, a2), foo in self.dependencies.items():
            if a2 == aspect_name:
                return foo(org[a1], condition)
        # If there is no dependency, touching this
        # aspect, return the one requested by user
        return condition
    
    def _generate_recursive(self, current_aspect_idx: int,
                                  partial_org: dict,
                                  conditions: dict):

        # If there are no more aspects to generate,
        # return this organization
        if current_aspect_idx >= len(self.sequence):
            return partial_org

        current_aspect_name = self.sequence[current_aspect_idx]

        # Make restrictions for this aspect based on
        # user-specified dependencies
        # ... and check if there is a way to match the user-requested
        # input and the input, determined by the upper levels
        try:
            cond = self._make_condition(partial_org,
                                        current_aspect_name,
                                        conditions.get(current_aspect_name, None))
        except OrganizationConfigurationConflict:
            return None

        # Generate several organizations
        orgs = self.generators[current_aspect_name].generate_valid(self.N_PROBES, ctx=cond)
        for org in orgs:
            partial_org[current_aspect_name] = org
            complete = self._generate_recursive(current_aspect_idx + 1,
                                                partial_org,
                                                conditions)
            # If some configuration was found down the line
            # return it
            if complete is not None:
                return complete
            del partial_org[current_aspect_name]

        # If no configurations were found (and returned)
        # up to this moment, then report failure
        return None
    
    def generate(self, cond: dict =None) -> list[dict[str, Organization]]:
        """Multi-level generation.

        Parameters:
        cond : dict
            Multilevel specification. Not supported yet.

        Returns:
        A generated organization or None, if failure.
        """
        if cond is None:
            cond = {}
        
        complex_org = self._generate_recursive(0, {}, cond)

        return complex_org
