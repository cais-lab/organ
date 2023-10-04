# flake8: noqa
import os
import copy
import random
import torch

import numpy as np


class LogisticsDepartmentOrganizationStructureModel:
    """SCSP demo model class for the logistics department
    scenario.
    """

    # status: 0 - optional, 1 - mandatory, 2 - replaceble
    node_type_dict = {
        0:  {'title': 'none', 'status': 0, 'weight': 0},
        1:  {'title': 'Warehouse Management', 'status': 2, 'replacement': [2, 3]},
        2:  {'title': 'Material Stock Management', 'status': 0},
        3:  {'title': 'Product Stock Management', 'status': 0},
        4:  {'title': 'Planning', 'status': 2, 'replacement': [5, 6, 7], 'children': [8]},
        5:  {'title': 'Material Planning', 'status': 0},
        6:  {'title': 'Product Stick Planning', 'status': 0},
        7:  {'title': 'Logistics Planning', 'status': 0},
        8:  {'title': 'Analytics', 'status': 0},
        9:  {'title': 'Audit', 'status': 0},
        10: {'title': 'Transportation', 'status': 1, 'children': [11]},
        11: {'title': 'Fleet Management', 'status': 0},
    }

    top_level_nodes = [1, 4, 9, 10]

    relations_dict = [
        # 0   1   2   3   4   5   6   7   8   9   10  11
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 0
        [ 0,  0,  1,  1,  2,  2,  2,  0,  2,  2,  2,  2], # 1
        [ 0,  0,  0,  0,  2,  2,  0,  2,  2,  2,  2,  2], # 2
        [ 0,  0,  0,  0,  2,  0,  2,  2,  2,  2,  2,  0], # 3
        [ 0,  2,  2,  2,  0,  1,  1,  1,  1,  0,  2,  2], # 4
        [ 0,  2,  2,  0,  0,  0,  0,  2,  2,  0,  0,  2], # 5
        [ 0,  2,  0,  2,  0,  0,  0,  2,  2,  0,  0,  0], # 6
        [ 0,  0,  2,  2,  0,  2,  2,  0,  2,  0,  2,  2], # 7
        [ 0,  2,  2,  2,  0,  2,  2,  2,  0,  0,  2,  2], # 8
        [ 0,  2,  2,  2,  0,  0,  0,  0,  0,  0,  2,  2], # 9
        [ 0,  2,  2,  2,  2,  0,  0,  2,  2,  2,  0,  1], # 10
        [ 0,  2,  2,  0,  2,  2,  0,  2,  2,  2,  0,  0], # 11
    ]

    # Number of node types
    NODE_N_TYPES = len(node_type_dict)
    # Number of edge types
    EDGE_N_TYPES = 3
    # Max number of vertices per graph
    MAX_NODES_PER_GRAPH = 12

    # Parametrization constants
    upper_limit = 25
    min_person = 0.7
    max_person = 1.4
    min_nonzero_value = 0.1
    min_orgunit = 2
    req_orgunit = 5

    def check_org_unit_feasibility(self, nodes, load, unit_id,
                                   min_person, max_person, min_orgunit, req_orgunit,
                                   logging=False):
        """Checking the feasibility of the staff quantity of an organisational unit.

        Parameters
        ----------
        nodes : list
            Model's nodes.
        load : numeric
            Expected load of the node to be checked.
        unit_id : int
            The node to be checked.
        min_person : numeric
            Allowed minimum load for a person.
        max_person : numeric
            Allowed maximum load for a person.
        min_orgunit : numeric
            Allowed minimum load for a dedicated organisational unit.
        req_orgunit : numeric
            Minimum load for that requaires a dedicated organisational unit.
        logging : bool
            Enable/disable logging.

        Returns
        -------
        bool
            True if the validations successful, False otherwise.
        """
        if ((nodes[unit_id]==0 and load[unit_id]>max_person*req_orgunit) or
            (nodes[unit_id]>0 and load[unit_id]<min_person*min_orgunit)):
            #error
            if logging:
                print(f"Node {unit_id} {self.node_type_dict[unit_id]['title']}"
                      " doesn't meet the requirements: load = {load[unit_id]},"
                      " node = {nodes[unit_id]}.")
            return False

        return True

    def generate_key_values(self, nodes, logging=False):
        """Generation of random key values for model parameters.

        Parameters
        ----------
        nodes : list
            List of model nodes.
        logging : bool
            Enable/disable logging..

        Returns
        -------
        list
            list of generated key values.
        """

        v = np.zeros(12)

        if nodes[2] > 0:
            v[2] = np.random.uniform(self.min_person*self.min_orgunit, self.upper_limit)
        else:
            v[2] = np.random.uniform(self.min_nonzero_value, self.max_person*self.req_orgunit)

        if nodes[3] > 0:
            v[3] = np.random.uniform(self.min_person*self.min_orgunit, self.upper_limit)
        else:
            v[3] = np.random.uniform(self.min_nonzero_value, self.max_person*self.req_orgunit)

        if logging:
            print("input v=", v)
        return v

    def generate_values(self, nodes, v, logging=False):
        """Generation of augmented model parameters.

        Parameters
        ----------
        nodes : list
            List of model nodes.
        v : list
            Key values for parameter generation.
        logging : bool
            Enable/disable logging.

        Returns
        -------
        list
            list of generated model parameter values.
        """

        v2_k = 0.2 if nodes[2] > 0 else 1
        v3_k = 0.2 if nodes[3] > 0 else 1

        v[1] = v2_k * v[2] + v3_k * v[3]
        if not self.check_org_unit_feasibility(nodes, v, 1,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v[10] = v[2] * 1 + v[3] * 1

        v[11] = v[10] * 0.2
        if not self.check_org_unit_feasibility(nodes, v, 11,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v12_k = v[2] * 0.4 if nodes[2] > 0 else v[1] * 0.2
        v11_k = v[11] * 0.2 if nodes[11] > 0 else 0
        v[5] = v12_k + v11_k
        if not self.check_org_unit_feasibility(nodes, v, 5,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v[6] = v[3] * 0.4 if nodes[3] > 0 else v[1] * 0.2
        if not self.check_org_unit_feasibility(nodes, v, 6, self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v[7] = v[10] * 0.2
        if not self.check_org_unit_feasibility(nodes, v, 7, self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v[8] = v[5] * 0.2 + v[6] * 0.2 + v[7] * 0.2
        if not self.check_org_unit_feasibility(nodes, v, 8,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v5_k = 0.2 if nodes[5] > 0 else 1
        v6_k = 0.2 if nodes[6] > 0 else 1
        v7_k = 0.2 if nodes[7] > 0 else 1
        v8_k = 0.2 if nodes[8] > 0 else 1
        v[4] = v5_k * v[5] + v6_k * v[6] + v7_k * v[7] + v8_k * v[8]
        if not self.check_org_unit_feasibility(nodes, v, 4,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        v[9] = v[1] * 0.05 + v[2] * 0.05 + v[3] * 0.05 + v[10] * 0.05 + v[11] * 0.05
        if not self.check_org_unit_feasibility(nodes, v, 9,
                                               self.min_person, self.max_person,
                                               self.min_orgunit, self.req_orgunit,
                                               logging=logging):
            return np.empty(0)

        return v

    def convert_values2persons(self, nodes, load, min_person, max_person):
        """Convert load values to staff quantity.

        Parameters
        ----------
        nodes : list
            List of model nodes.
        load : list
            Load per unit.
        min_person : numeric
            Allowed minimum load for a person.
        max_person : numeric
            Allowed maximum load for a person.

        Returns
        -------
        list
            list of staff quantities.
        """
        staff = np.zeros(12)
        for i in range(len(nodes)):
            #print(i)
            if nodes[i] > 0:
                #print(load[i], load[i]/max_person, load[i]/min_person)
                staff[i] = np.random.randint(np.ceil(load[i]/max_person),
                                             np.ceil(load[i]/min_person),
                                             1)[0]
        return staff

    def pack_to_ctx(self, v):
        """Pack list of load values for all nodes to context.

        Parameters
        ----------
        v : list
            List of load values for all nodes.

        Returns
        -------
        list
            context (key parameters).
        """
        return [v[2], v[3]]
    
    def unpack_ctx(self, ctx):
        """Unpack context to list of load values for all nodes.

        Parameters
        ----------
        ctx : list
            context.

        Returns
        -------
        list
            list of load values for all nodes.
        """
        v = np.zeros(12)
        v[2] = ctx[0]
        v[3] = ctx[1]
        return v

    def check_children(self, top_level_nodes, node_list, force_nodes = False):
        """Recursive function for checking correctness of model's node structure.

        Parameters
        ----------
        top_level_nodes : list
            List of top level nodes.
        node_list : list
            List of child nodes.
        force_nodes : bool
            Flag indicating that child models are mandatory.

        Returns
        -------
        tuple
            Boolean flag of the model structure correctness, txtual description
            of the problem (empty the model structure is valid).
        """
        for key in top_level_nodes:
            if self.node_type_dict[key]['status'] == 1 or force_nodes:
                if node_list[key] == 0:
                    return False, f'Mandatory element {key} is missing.'
                if 'replacement' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['replacement'], node_list)
                    if not result:
                        return False, explanation
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
            if self.node_type_dict[key]['status'] == 2 or force_nodes:
                if node_list[key] == 0:
                    if 'replacement' in self.node_type_dict[key]:
                        result, explanation = self.check_children(
                            self.node_type_dict[key]['replacement'], node_list, True)
                        if not result:
                            return False, f'Replacable element {key} is missing. {explanation}'
                    else:
                        return False, f'Replacable element {key} is missing.'
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
            elif self.node_type_dict[key]['status'] == 0:
                if 'replacement' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['replacement'], node_list)
                    if not result:
                        return False, explanation
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
        return True, ''

    def check_relations(self, nodes, relations):
        """Checks relations validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node types for each vertex (length must be
            ==`self.MAX_NODES_PER_GRAPH`).
        relations : numpy.ndarray (n, n)
            Relation type matrix.

        Returns
        -------
        bool
            Returns `True` if all the set of edges is valid and consistent
            with the nodes.
        diff
            Boolean matrix of edge validness (`True` for valid edges).
        """
        target_relations = np.array(copy.deepcopy(self.relations_dict))
        for node_key in self.node_type_dict:
            if node_key not in nodes:
                target_relations[node_key, :] = 0
                target_relations[:, node_key] = 0

        relations_diff = np.array([relations == target_relations])
        result = relations_diff.all()
        return result, relations_diff

    def check_nodes(self, nodes) -> bool:
        """Checks node types validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node types for each vertex (length must be
            ==`self.MAX_NODES_PER_GRAPH`).

        Returns
        -------
        bool
            Returns `True` if the structure contains valid set of nodes.
        """
        return self.check_children(self.top_level_nodes, nodes)[0]

    def overlap(self, first, last, another_first, another_last)->bool:
        """Checks if two intervals intersect.

        Parameters
        ----------
        first : numeric
            Lower bound of the first interval
        last : numeric
            Upper bound of the first interval
        another_first : numeric
            Lower bound of the second interval
        another_last : numeric
            Upper bound of the second interval

        Returns
        -------
        bool
            Returns `True` if the intervals intersect.
        """
        return min(last, another_last) - max(first, another_first) >= 0

    def check_paramater_feasibility(self, nodes, staff, logging=False, ctx=None):
        """Checks parameter validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node.
        staff : List, numpy.array
            The list of parameters.
        logging : bool
            Enable/disable logging.
        ctx : list
            context.

        Returns
        -------
        bool
            Returns `True` if the parameters are valid.
        """
        def log_capacity_problem(unit_id):
            log = f"Capacity of node {unit_id} " \
                  f"{self.node_type_dict[unit_id]['title']} " \
                   "does not meet requirements: "
            log += ' '.join([str(load_min[unit_id]),
                             str(load_max[unit_id]),
                             str(v_min[unit_id]),
                             str(v_max[unit_id])])
            return log
        
        log = None

        for unit_id in self.node_type_dict:
            # Optional node (department) has too few staff members
            if (not self.node_type_dict[unit_id]['status'] == 1 and
                nodes[unit_id] > 0 and staff[unit_id] < self.min_orgunit):
                #error
                if logging:
                    log = f"Node {unit_id} " \
                          f"{self.node_type_dict[unit_id]['title']} " \
                          f"doesn't meet the requirements: " \
                          f"staff = {staff[unit_id]}"
                return False, log
            # Empty node (department) has some staff
            if nodes[unit_id] == 0 and staff[unit_id] > 0:
                #error
                if logging:
                    log = f"Non-existing Node {unit_id} " \
                          f"{self.node_type_dict[unit_id]['title']} " \
                          f"has staff: " \
                          f"staff = {staff[unit_id]}"
                return False, log

        load_min = staff * self.min_person
        load_max = staff * self.max_person + self.min_person
        max_no_unit = self.req_orgunit * self.max_person
        for unit_id in self.node_type_dict:
            if nodes[unit_id] == 0:
                load_max[unit_id] = max_no_unit

        v_min = np.zeros(12)
        v_max = np.zeros(12)

        unit_id = 1
        v_min[unit_id] = ((0.2 if nodes[2] > 0 else 1) * load_min[2] +
                         (0.2 if nodes[3] > 0 else 1) * load_min[3])
        v_max[unit_id] = ((0.2 if nodes[2] > 0 else 1) * load_max[2] +
                         (0.2 if nodes[3] > 0 else 1) * load_max[3])
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 10
        v_min[10] = load_min[2] * 1 + load_min[3] * 1
        v_max[10] = load_max[2] * 1 + load_max[3] * 1
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 11
        v_min[11] = load_min[10] * 0.2
        v_max[11] = load_max[10] * 0.2
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 5
        v_min[5] = ((load_min[2] * 0.4 if nodes[2] > 0 else load_min[1] * 0.2) +
                    (load_min[11] * 0.2 if nodes[11] > 0 else 0))
        v_max[5] = ((load_max[2] * 0.4 if nodes[2] > 0 else load_max[1] * 0.2) +
                    (load_max[11] * 0.2 if nodes[11] > 0 else 0))
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 6
        v_min[6] = load_min[3] * 0.4 if nodes[3] > 0 else load_min[1] * 0.2
        v_max[6] = load_max[3] * 0.4 if nodes[3] > 0 else load_max[1] * 0.2
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 7
        v_min[7] = load_min[10] * 0.2
        v_max[7] = load_max[10] * 0.2
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 8
        v_min[8] = load_min[5] * 0.2 + load_min[6] * 0.2 + load_min[7] * 0.2
        v_max[8] = load_max[5] * 0.2 + load_max[6] * 0.2 + load_max[7] * 0.2
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 4
        v_min[4] = (0.2 if nodes[5] > 0 else 1) * load_min[5] + \
                   (0.2 if nodes[6] > 0 else 1) * load_min[6] + \
                   (0.2 if nodes[7] > 0 else 1) * load_min[7] + \
                   (0.2 if nodes[8] > 0 else 1) * load_min[8]
        v_max[4] = (0.2 if nodes[5] > 0 else 1) * load_max[5] + \
                   (0.2 if nodes[6] > 0 else 1) * load_max[6] + \
                   (0.2 if nodes[7] > 0 else 1) * load_max[7] + \
                   (0.2 if nodes[8] > 0 else 1) * load_max[8]
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        unit_id = 9
        v_min[9] = (load_min[1] * 0.05 +
                    load_min[2] * 0.05 +
                    load_min[3] * 0.05 +
                    load_min[10] * 0.05 +
                    load_min[11] * 0.05)
        v_max[9] = (load_max[1] * 0.05 +
                    load_max[2] * 0.05 +
                    load_max[3] * 0.05 +
                    load_max[10] * 0.05 +
                    load_max[11] * 0.05)
        if nodes[unit_id] > 0 and not self.overlap(load_min[unit_id],
                                                   load_max[unit_id],
                                                   v_min[unit_id],
                                                   v_max[unit_id]):
            #error
            if logging:
                log = log_capacity_problem(unit_id)
            return False, log

        if ctx is not None:
            tmp_v = self.generate_values(nodes, self.unpack_ctx(ctx), logging=logging)
            for idx, val in enumerate(tmp_v):
                if idx == 0 or idx == 2 or idx == 3:
                    continue
                if nodes[idx]>0 and (val > v_max[idx] or val < v_min[idx]):
                    if logging:
                        log = f"Value of node {idx} {self.node_type_dict[idx]['title']} does not meet context. "
                        log += str(v_min[idx]) + ' '
                        log += str(val) + ' '
                        log += str(v_max[idx])
                    return False, log

        return True, None

    def generate_augmentation(self,
                              base_nodes,
                              base_edges,
                              base_staff,
                              logging=False,
                              max_iterations=100):
        """Generate augmentation.

        Parameters
        ----------
        base_nodes : List, numpy.array
            Nodes of the source configuration.
        base_nodes : List, numpy.array
            Edges of the source configuration.
        base_staff : List, numpy.array
            The staff quantiities of the source configuration.
        logging : bool
            Enable/disable logging.
        max_iterations : int
            maximum number of iterations until a valid 
            augmented configuration is generated.

        Returns
        -------
        tuple
            aug_nodes - augmented nodes,
            aug_edges - augmented edges,
            aug_staff - augmented staff,
            self.pack_to_ctx(v) - augmented context.
        """

        #   Augment structure
        iterations = 0
        valid_structure = False
        while iterations < max_iterations:
            iterations += 1
            aug_nodes = base_nodes.copy()
            for i in range(1,len(self.node_type_dict)):
                if random.choices([True, False], k=1, weights=[1, 8])[0]:
                    if aug_nodes[i] == 0:
                        aug_nodes[i] = i
                    else:
                        aug_nodes[i] = 0
            if self.check_children(self.top_level_nodes, aug_nodes)[0]:
                valid_structure = True
                break
        if not valid_structure:
            return  [], [], [], []
        aug_edges = np.array(copy.deepcopy(self.relations_dict))
        for node_key in self.node_type_dict:
            if node_key not in aug_nodes:
                aug_edges[node_key, :] = 0
                aug_edges[:, node_key] = 0

        #   Augment parameters
        iterations = 0
        valid_params = False
        while iterations < max_iterations:
            iterations += 1
            v = self.generate_key_values(aug_nodes, logging)
            load = self.generate_values(aug_nodes, v, logging)
            if len(load) > 0:
                valid_params = True
                if logging:
                    print(load)
                aug_staff = self.convert_values2persons(aug_nodes,
                                                        load,
                                                        self.min_person,
                                                        self.max_person)

                break
        if not valid_params:
            return  [], [], [], []
        return aug_nodes, aug_edges, aug_staff, self.pack_to_ctx(v)

    def check_uniqueness(self,
                         ground_truth_nodes,
                         ground_truth_edges,
                         ground_truth_staff,
                         ground_truth_ctx,
                         nodes,
                         edges,
                         staff,
                         ctx):
        """Checks structure uniqueness compared to the training set.

        Parameters
        ----------
        ground_truth_nodes : List, numpy.array
            Nodes of the training set configurations.
        ground_truth_edges : List, numpy.array
            Edges of the training set configurations.
        ground_truth_staff : List, numpy.array
            Staff quantities of the training set configurations.
        ground_truth_ctx : List, numpy.array
            Contexts of the training set configurations.
        nodes : List, numpy.array
            Nodes of the checked configuration.
        edges : List, numpy.array
            Edges of the checked configuration.
        staff : List, numpy.array
            Staff quantiities of the checked configuration.
        ctx: List, numpy.array
            Context of the checked configuration.

        Returns
        -------
        bool
            True if the configuration is unique,
            False otherwise.
        """
        for i in range(ground_truth_nodes.shape[0]):
            if  ((ground_truth_nodes[i]==nodes).all() and
                 (ground_truth_edges[i]==edges).all() and
                 (ground_truth_staff[i]==staff).all() and
                 (ground_truth_ctx[i]==ctx).all()):
                return False
        return True


class ManagementStructureModel:
    """SCSP demo model class, describing administration and sales scenario.
    """

    # status: 0 - optional, 1 - mandatory, 2 - replaceble
    node_type_dict = {
        0:  {'title': 'none', 'status': 0, 'weight': 0},
        1:  {'title': 'Administration', 'status': 2, 'replacement': [2, 3], 'children': [4]},
        2:  {'title': 'Headquarters', 'status': 0},
        3:  {'title': 'Management', 'status': 0},
        4:  {'title': 'Marketing department', 'status': 0},
        5:  {'title': 'Warehouse', 'status': 0},
        6:  {'title': 'Logistics', 'status': 0},
        7:  {'title': 'ERP System', 'status': 1},
        8:  {'title': 'Shop', 'status': 1},
    }

    top_level_nodes = [1, 5, 6, 7, 8]

    relations_dict = [
        # 0   1   2   3   4   5   6   7   8   
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # 0
        [ 0,  0,  1,  1,  1,  0,  0,  2,  0], # 1
        [ 0,  0,  0,  0,  0,  0,  0,  2,  0], # 2
        [ 0,  0,  0,  0,  0,  0,  0,  2,  6], # 3
        [ 0,  0,  0,  0,  0,  0,  0,  2,  0], # 4
        [ 0,  0,  0,  0,  0,  0,  4,  3,  0], # 5
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # 6
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # 7
        [ 0,  0,  0,  0,  0,  0,  5,  3,  0], # 8
    ]

    # Number of node types
    NODE_N_TYPES = len(node_type_dict)
    # Number of edge types
    EDGE_N_TYPES = 7
    # Max number of vertices per graph
    MAX_NODES_PER_GRAPH = 9

    # Parametrization constants
    node_param_0_min_allowed = [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    node_param_0_min_required = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1200.0, 0.0, 0.0]
    param_0_max = 2000.0

    def __init__(self):
        pass

    def generate_key_values(self, nodes, logging=False):
        """Generation of random key values for model parameters.

        Parameters
        ----------
        nodes : list
            List of model nodes.
        logging : bool
            Enable/disable logging..

        Returns
        -------
        list
            list of generated key values.
        """

        v = np.zeros(2)

        tmp_param_min = 0
        tmp_param_max = self.param_0_max
        for i, node in enumerate(nodes):
            if node > 0:
                tmp_param_min = max(self.node_param_0_min_allowed[i], tmp_param_min)
            elif self.node_param_0_min_required[i] > 0:
                tmp_param_max = min(self.node_param_0_min_required[i], tmp_param_max)
        
        v[0] = np.random.uniform(tmp_param_min, tmp_param_max)
        
        v[1] = int(nodes[4] > 0) 

        if logging:
            print("input v=", v)
        return v

    def generate_values(self, nodes, v, logging=False):
        """Generation of augmented model parameters.

        Parameters
        ----------
        nodes : list
            List of model nodes.
        v : list
            Key values for parameter generation.
        logging : bool
            Enable/disable logging.

        Returns
        -------
        list
            list of generated model parameter values.
        """

        params = np.zeros([9, 2])
        params[8, 0] = v[0]
        
        pm = max(int(nodes[1]>0) * params[8, 0] / 200, int(nodes[1]>0)) + \
             max(int(nodes[2]>0) * params[8, 0] / 200, int(nodes[2]>0)) + \
             max(int(nodes[3]>0) * params[8, 0] / 200, int(nodes[3]>0)) + \
             int(nodes[4]>0) * 5 + \
             max(int(nodes[5]>0) * params[8, 0] / 100, int(nodes[5]>0)) + \
             max(int(nodes[6]>0) * params[8, 0] / 500, int(nodes[6]>0)) + \
             max(int(nodes[8]>0) * params[8, 0] / 100, int(nodes[3]>0))
        params[7, 1] = round(pm)
        return params

    def check_children(self, top_level_nodes, node_list, force_nodes = False):
        """Recursive function for checking correctness of model's node structure.

        Parameters
        ----------
        top_level_nodes : list
            List of top level nodes.
        node_list : list
            List of child nodes.
        force_nodes : bool
            Flag indicating that child models are mandatory.

        Returns
        -------
        tuple
            Boolean flag of the model structure correctness, txtual description
            of the problem (empty the model structure is valid).
        """
        for key in top_level_nodes:
            if self.node_type_dict[key]['status'] == 1 or force_nodes:
                if node_list[key] == 0:
                    return False, f'Mandatory element {key} is missing.'
                if 'replacement' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['replacement'], node_list)
                    if not result:
                        return False, explanation
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
            if self.node_type_dict[key]['status'] == 2 or force_nodes:
                if node_list[key] == 0:
                    if 'replacement' in self.node_type_dict[key]:
                        flag = False
                        for child_node in self.node_type_dict[key]['replacement']:
                            result, explanation = self.check_children(
                                [child_node], node_list, True)
                            if result:
                                flag = True
                        if not flag:
                            return False, f'No replacements for {key}.'
                    else:
                        return False, f'Replacable element {key} is missing.'
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
            elif self.node_type_dict[key]['status'] == 0:
                if 'replacement' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['replacement'], node_list)
                    if not result:
                        return False, explanation
                if 'children' in self.node_type_dict[key]:
                    result, explanation = self.check_children(
                        self.node_type_dict[key]['children'], node_list)
                    if not result:
                        return False, explanation
        return True, ''

    def check_relations(self, nodes, relations):
        """Checks relations validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node types for each vertex (length must be
            ==`self.MAX_NODES_PER_GRAPH`).
        relations : numpy.ndarray (n, n)
            Relation type matrix.

        Returns
        -------
        bool
            Returns `True` if all the set of edges is valid and consistent
            with the nodes.
        diff
            Boolean matrix of edge validness (`True` for valid edges).
        """
        target_relations = np.array(copy.deepcopy(self.relations_dict))
        for node_key in self.node_type_dict:
            if node_key not in nodes:
                target_relations[node_key, :] = 0
                target_relations[:, node_key] = 0
                
        if nodes[1] == 1:
            target_relations[2, 7] = 0
            target_relations[3, 7] = 0
            target_relations[4, 7] = 0

        relations_diff = np.array([relations == target_relations])
        result = relations_diff.all()
        return result, relations_diff

    def check_nodes(self, nodes) -> bool:
        """Checks node types validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node types for each vertex (length must be
            ==`self.MAX_NODES_PER_GRAPH`).

        Returns
        -------
        bool
            Returns `True` if the structure contains valid set of nodes.
        """
        return self.check_children(self.top_level_nodes, nodes)[0]

    def check_paramater_feasibility(self, nodes, staff, ctx):
        """Checks parameter validity.

        Parameters
        ----------
        nodes : List, numpy.array
            The list of node.
        staff : List, numpy.array
            The list of parameters.
        logging : bool
            Enable/disable logging.
        ctx : list
            context.

        Returns
        -------
        bool
            Returns `True` if the parameters are valid.
        """

        log = "Valid configuration"

        if ctx[1] != int(nodes[4] > 0): 
            log = f"Structure does not meet the context. "
            log += f"{self.node_type_dict[4]['title']} = {nodes[4]}. "
            log += f"Context[1] = {ctx[1]}"
        
        pm = max(int(nodes[1]>0) * staff[8, 0] / 200, int(nodes[1]>0)) + \
             max(int(nodes[2]>0) * staff[8, 0] / 200, int(nodes[2]>0)) + \
             max(int(nodes[3]>0) * staff[8, 0] / 200, int(nodes[3]>0)) + \
             int(nodes[4]>0) * 5 + \
             max(int(nodes[5]>0) * staff[8, 0] / 100, int(nodes[5]>0)) + \
             max(int(nodes[6]>0) * staff[8, 0] / 500, int(nodes[6]>0)) + \
             max(int(nodes[8]>0) * staff[8, 0] / 100, int(nodes[3]>0))
        if round(staff[7, 1]) != round(pm):
            log = f"Value 1 of node 7 {self.node_type_dict[7]['title']} does not meet the configuration. "
            log += f"Value = {staff[7, 1]} "
            log += f"Required value = {pm} "
            return False, log
        
        return True, None

    def generate_augmentation(self,
                              base_nodes,
                              base_edges,
                              base_staff,
                              logging=False,
                              max_iterations=100):
        """Generate augmentation.

        Parameters
        ----------
        base_nodes : List, numpy.array
            Nodes of the source configuration.
        base_nodes : List, numpy.array
            Edges of the source configuration.
        base_staff : List, numpy.array
            The staff quantiities of the source configuration.
        logging : bool
            Enable/disable logging.
        max_iterations : int
            maximum number of iterations until a valid 
            augmented configuration is generated.

        Returns
        -------
        tuple
            aug_nodes - augmented nodes,
            aug_edges - augmented edges,
            aug_staff - augmented staff,
            ctx - augmented context.
        """

        #   Augment structure
        iterations = 0
        valid_structure = False
        while iterations < max_iterations:
            iterations += 1
            aug_nodes = base_nodes.copy()
            for i in range(1,len(self.node_type_dict)):
                if random.choices([True, False], k=1, weights=[1, 8])[0]:
                    if aug_nodes[i] == 0:
                        aug_nodes[i] = i
                    else:
                        aug_nodes[i] = 0
            if self.check_children(self.top_level_nodes, aug_nodes)[0]:
                valid_structure = True
                break
        if not valid_structure:
            return  [], [], [], []
        aug_edges = np.array(copy.deepcopy(self.relations_dict))
        for node_key in self.node_type_dict:
            if node_key not in aug_nodes:
                aug_edges[node_key, :] = 0
                aug_edges[:, node_key] = 0
                
        if aug_nodes[1] == 1:
            aug_edges[2, 7] = 0
            aug_edges[3, 7] = 0
            aug_edges[4, 7] = 0

        #   Augment parameters
        ctx = self.generate_key_values(aug_nodes, logging)
        aug_params = self.generate_values(aug_nodes, ctx, logging)
        return aug_nodes, aug_edges, aug_params, ctx

    def check_uniqueness(self,
                         ground_truth_nodes,
                         ground_truth_edges,
                         ground_truth_staff,
                         ground_truth_ctx,
                         nodes,
                         edges,
                         staff,
                         ctx):
        """Checks structure uniqueness compared to the training set.

        Parameters
        ----------
        ground_truth_nodes : List, numpy.array
            Nodes of the training set configurations.
        ground_truth_edges : List, numpy.array
            Edges of the training set configurations.
        ground_truth_staff : List, numpy.array
            Staff quantities of the training set configurations.
        ground_truth_ctx : List, numpy.array
            Contexts of the training set configurations.
        nodes : List, numpy.array
            Nodes of the checked configuration.
        edges : List, numpy.array
            Edges of the checked configuration.
        staff : List, numpy.array
            Staff quantiities of the checked configuration.
        ctx: List, numpy.array
            Context of the checked configuration.

        Returns
        -------
        bool
            True if the configuration is unique,
            False otherwise.
        """
        for i in range(ground_truth_nodes.shape[0]):
            if  ((ground_truth_nodes[i]==nodes).all() and
                 (ground_truth_edges[i]==edges).all() and
                 (ground_truth_staff[i]==staff).all() and
                 (ground_truth_ctx[i]==ctx).all()):
                return False
        return True


class LogisticsDepartmentModel(
    LogisticsDepartmentOrganizationStructureModel):
    """An adapter, defining methods to use organization structure
    model for logistics scenario during OrGAN training."""

    def __init__(self):
        super(LogisticsDepartmentModel, self).__init__()

    def validness(self, org) -> bool:
        """Checks structure validness.

        Parameters
        ----------
        org : Configuration
            Organization structure configuration.

        Returns
        -------
        bool
            True if the configuration is valid,
            False otherwise.
        """
        return self.check_nodes(org.nodes) and \
            self.check_relations(org.nodes, org.edges)[0] and \
            self.check_paramater_feasibility(org.nodes,
                                             org.node_features.ravel(),
                                             ctx=org.condition)[0]

    def metrics(self, org) -> dict:
        """Returns a dict with relevant metric values.

        Parameters
        ----------
        org : Configuration
            Organization structure configuration.

        Returns
        -------
        dict
            metrics and functions for their evaluation.
        """
        return {
            'node score': self.check_nodes(org.nodes),
            'edge score': self.check_relations(org.nodes, org.edges)[0],
            'staff score': self.check_paramater_feasibility(org.nodes,
                org.node_features.ravel(),
                ctx=org.condition)[0]
        }


class ManagementModel(ManagementStructureModel):
    """An adapter, defining methods to use organization structure
    model for the administration and sales scenario during OrGAN
    training.
    """

    def __init__(self):
        super(ManagementStructureModel, self).__init__()

    def validness(self, org) -> bool:
        """Checks structure validness.

        Parameters
        ----------
        org : Configuration
            Organization structure configuration.

        Returns
        -------
        bool
            True if the configuration is valid,
            False otherwise.
        """
        return self.check_nodes(org.nodes) and \
            self.check_relations(org.nodes, org.edges)[0] and \
            self.check_paramater_feasibility(org.nodes,
                                             org.node_features,
                                             ctx=org.condition)[0]

    def metrics(self, org) -> dict:
        """Returns a dict with relevant metric values.

        Parameters
        ----------
        org : Configuration
            Organization structure configuration.

        Returns
        -------
        dict
            metrics and functions for their evaluation.
        """
        return {
            'node score': self.check_nodes(org.nodes),
            'edge score': self.check_relations(org.nodes, org.edges)[0],
            'staff score': self.check_paramater_feasibility(org.nodes,
                org.node_features,
                ctx=org.condition)[0]
        }

    def soft_constraints(self, nodes, edges, params, ctx):
        """Soft constraints for this scenario.

        The function describes some relationships between
        node parameters, context, and organization structure
        to simplify the training of a generator.

        Parameters
        ----------
        nodes : torch.tensor
            Nodes description in an 'internal' format:
            (batch, nodes, node_types). Value is the probability that
            a node of the specific type is located in a certain position.
            Zero-type corresponds to the absense of a node. Non-zero
            values can be only on the matrix diagonal or zeroth column.
        edges : torch.tensor
            Edges representation in an 'internal' format:
            (batch, nodes, nodes, edge_types).
        params : torch.tensor
            Node features: (batch, nodes, features_per_node).
        ctx : torch.tensor
            Generation context: (batch, context_features).

        Returns
        -------
        torch.tensor
            Value tensor (0-dimensional). Non-negative loss for violation
            of the constraints.
        """
        marketing_cnt = torch.mean(torch.abs(nodes[:, 4, 4] - ctx[:, 1]))
        
        node_8_cnt = torch.mean(torch.abs(params[:, 8, 0] - ctx[:, 0])) / self.param_0_max
        
        node_existence = torch.sum(nodes[:, :, 1:], axis=-1)

        pm_nodes = torch.stack([
            ctx[:, 0] * 0,
            ctx[:, 0] / 200,
            ctx[:, 0] / 200,
            ctx[:, 0] / 200,
            ctx[:, 0] / ctx[:, 0] * 5,
            ctx[:, 0] / 100,
            ctx[:, 0] / 500,
            ctx[:, 0] * 0,
            ctx[:, 0] / 100], dim=-1)
        pm = torch.sum(node_existence * pm_nodes, axis=-1)
        node_7_cnt = torch.mean(torch.abs(pm-params[:, 7, 1]))

        node_2_cnt = torch.mean(torch.nn.functional.relu((1000 - ctx[:, 0]) / 1000) *
                                nodes[:, 2, 2])
        node_6_cnt = torch.mean(torch.nn.functional.relu((ctx[:, 0] - 1200) / 1200) *
                                (1-nodes[:, 6, 6]))

        return torch.mean(torch.stack([marketing_cnt,
                                       node_8_cnt,
                                       node_7_cnt,
                                       node_2_cnt,
                                       node_6_cnt]))
