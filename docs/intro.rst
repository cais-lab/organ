Introduction
============

Installation
------------

The recommended way of using the library is to copy the contents of the ``organ``
folder of the repository to your project. Make sure that the environment
you're running satisfies the requirements of OrGAN. In order to do so, you may
want to run:

.. code-block:: shell

  $ pip install -r requirements.txt

OrGAN repository root folder contains a number of scripts (first of all, :file:`main.py`).
If you only want to train an organization structure generator feeding your
organization samples, then, most likely, you won't need any coding and the scripts
may be all you need. In this case, just clone the repository, create a
virtual environment and install dependencies:

.. code-block:: shell

  $ git clone https://gitlab.actcognitive.org/itmo-sai-code/organ.git organ
  $ cd organ
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt

After that, you can provide a dataset and run the training script.

Quick Start
-----------

There are two ways one can use the OrGAN library. Those users who don't have
any specific requirements for the generated organization structures may find
that the provided scripts (e.g., :file:`main.py`) are enough for them. Using
command-line options, one can control some basic requirements to the
generated organizations as well as neural network architectures and training
process. This quick start gives some flavour of this scenario.

We assume that the OrGAN was successfully installed according to the instructions
in the `Installation`_.

The whole idea of OrGAN is that it can generate structures *similar* to the
provided ones (presumably, created by experts). It means that in order to use
the library one have to prepare an organization structure dataset. In order to
simplify demonstration we include two demo datasets (for logistics department
scenario and for sales and administration scenario). However, the datasets
of organization structures are typically very small, therefore, before training
the model one shoud augment the training dataset:

.. code-block:: shell

  $ python augment_dataset.py demo_logistics 1000 demo_data/logistics data 

This script will create 1000 organization samples in the :file:`data` directory,
the dataset format is discussed in `Data`_.

After that, one should start training by running :file:`main.py`: 

.. code-block:: shell

  $ python main.py --rules demo_logistics

The script will periodically print the summary of validness and quality of
the generated structures.


Usage Scenarios
===============

Script-Only
-----------

This scenario doesn't require programming (at least, to train OrGAN), however,
it is limited to using only some generic pre-defined constrains on
organizations and it is not possible to define custom constraints. 

Basically, the scenario is following:

#. Prepare the dataset of real examples of organization structures 
   according to the data format specification (see `Data`_).

#. If the number of samples is less than several thousands, augment the
   dataset using :file:`augment_dataset.py` scipt.

#. Train the OrGAN. Assuming that the dataset was placed into the
   :file:`data` folder, training can be done by running the following
   script:

    .. code-block:: shell

      $ python main.py --rules=generic

Note, that the training script :file:`main.py` is run with the argument
'--rules=generic'. It means, that there are only generic requirements for the
organization stuctures (e.g., that a link is possible only between existing
units). You may also use other options of :file:`main.py` to control the
process of training and output (see :command:`python main.py --help`
for the list of options).

You may find that generic rules in most cases are in fact too generic and
the generated structures conforming to them are still not very useful.
In this case, you can define arbitrary constraints (and metrics) for your
organization structures using `Program-Level`_ scenario, but it requires a
bit of coding.

Program-Level
-------------

All the specific requirements to organization structures are connected
with the concept of an organization structure model. Such model is
represented by a Python class, defining following methods:

- `validness(org) -> bool`, which checks the organization structure
  for validness,
- `metrics(org) -> dict`, returning a dict with relevant organization
  metric values, and
- optional `soft_constraints(nodes, edges, features, condition) -> tensor`,
  which can implement some differentiable contraints.

Having defined such class one can pass the class name as an argument
for the training script:

.. code-block:: shell

  $ python main.py --rules=module.CustomOrgStructureModelClass

Let's illustrate the process of implementing a simplistic custom
organization structure model. We can create a Python module
:file:`hornsnhooves.py` in the repository root folder and
define `HNHStructureModel` class in it. As it was noted earlier,
this class has to implement several methods.

Let's start from the most important one - `validness(org)`. This method
can implement any checks of the organization structure (e.g., you
can easily invoke some external system, do some simulation, or whatever
seems resonable to ensure that the organization `org` is valid for you.
Note, however, that this method is called for each organization (existing
and generated), so you'll want to make it as efficient as possible.

The `org` parameter of this method is (currently) just a tuple - a pair of
two matrices - one describing organization nodes (corresponds to node type
in each position, zero if a position is empy), and the other one
describing connections between nodes (edge type for each pair of node
positions, zero if no edge).

.. note:: 

    This will probably change in future releases.

For example, let's require that valid structures must contain at least 3
elements and must also contain either node of type 1, or type 2 (or both
of them). Then, vaildness definition is following:

.. code-block:: python

   def validness(self, org):
       nodes, edges = org
       return (nodes != 0).sum() >= 3 and \
           (nodes[1] != 0 or nodes[2] != 0)

Then, we must define a set of metrics for organization structures. These
metrics will be printed during the training process. Typically, such metrics
characterize the validness and quality of the structure, so one
of the metrics for `HNHStructureModel` might directly correspond to
validness and another just show the organization size:

.. code-block:: python

   def metrics(self, org):
       return {'validness': self.validness(org),
               'size': (org[0] != 0).sum()}

Finally, we can add a number of differentiable soft constraints on the
organization structures. The generator will be penalized for violating
these constraints, so they are likely to be fulfilled (eventially), but
it is not strictly guaranteed. As these constraints have to be differentiable,
the interface of the `soft_constraints()` function is a bit different,
it should accept an organization representation in an internal form
(a couple of pytorch tensors), it can also use only differentiable functions
(typically, from pytorch library). OrGAN defines a set of helper functions
to assist in implementing `soft_constraints()` - 
see `organ.structure.constraints` module for the complete list. Let's,
for example, require that the edges would be symmetric and an egde
connect only existing nodes:

.. code-block:: python

   def soft_constraints(self, nodes, edges, *ignored):
       return 0.1 * organ.structure.constraints.edge_consistent(nodes, edges) + \
              0.1 * organ.structure.constraints.edge_symmetric(edges)

.. note:: 

    Typically you'll want that the set of valid structures would be
    a subset of the structures conforming the soft constraints.
    It is not the case in this example, but in practice should be.

In general, this function should return a non-negative penalty for the structure
described by `nodes` and `edges` tensors. The penalty should be zero
for a structure conforming to the requirements.

TiNGLe
======

For the most demanding users and use cases, OrGAN includes a
Tiny Neural Graph Library (TiNGLe), providing a set of abstractions and
tools to program (convolutional) graph neural networks and use
them as custom approximators and discriminators for OrGAN.

The TiNGLe supports graphs having several types of nodes and edges,
and uses graph representation most convenient for the
generation process, representing graph connectivity by an
adjacency matrix (and the presense of an edge is not binary,
it can be on the continuum from zero to one, which is important for
gradient flow). Conceptually, the library follows message passing
framework for graph neural networks and is based on the ideas,
described in https://distill.pub/2021/gnn-intro/. More precidely,
a graph is represented using the following components:

- global representation (one vector, describing graph as a whole);
- nodes representation. In TiNGLe it is assumed, that a node can have
  a type, besides, it can also have some set of features, so:

  - node types (batch x nodes x node_types);
  - node features (batch x nodes x N_F);

- edges representation. Edges can also be of multiple types,
  however, between a pair of nodes it is not possible to have
  more than one edge:

  - edge types (batch x edge_types x nodes x nodes).
    In this representation, 0 means that there is no edge of the
    respective type, and 1 - that there is. However, other values
    are also possible - they are interpreted as a "power" of connection
    and are used during the propagation through (or from) the
    respective edge.
  - edge representation (one for all types of edges)
    (batch x nodes x nodes x V_F).

The library defines two types of tools:

#. Functions to implement collection and aggragation steps for various kinds of
   message passing.
#. Classes and "orchestration" tools to compose the architecture of a
   complete graph neural network.

As a result, one can build graph neural networks in the following way:

.. code :: python

   gn = torch.nn.ModuleList([
       VV(merge='replace', apply_to_types=True),
       GNNBlock(nodes_module=torch.nn.Linear(2, 4)),
       VV(merge='replace'),
       GNNBlock(nodes_module=torch.nn.Linear(4, 2)),
   ])

For more information about TiNGLe and its API, please, refer to the
library documentation.

Data
====

Currently, OrGAN uses binary datasets. There are, however, plans to provide
conversion tools from popular graph exchange formats to this binary format
and we would welcome such contributions.

Each organization structure is described by a graph. Its nodes correspond to
organization units, and edges correspond to connections between units.

The dataset consists of a set of files with fixed names residing in one
folder:

- :file:`data_nodes.npy` - an integer (n, f) NumPy matrix with nodes description, 
- :file:`data_edges.npy` - an integer (n, f, f) NumPy matrix with edge description,
- :file:`data_staff.npy` - a float (n, f) NumPy matrix with node features,
- :file:`data_cond.npy` - a float (n, f) NumPy matrix with condition value
  (organization context to use as input for the generation),
- :file:`data_meta.pkl` - a pickle file, containing a dict with the dataset
  description.

All :file:`.npy` files are in a standard NumPy binary format. In the description
above, `n` is the number of samples in the dataset, `f` is the number of types of
nodes (and, at the same time, maximal number of nodes, because we assume
that an organization structure can contain at most one unit of some type).

Due to historical reasons, *i*-th position of the nodes description can contain only
*i* or 0. Zero at the *i*-th position means that there are no node with type
*i* in the respective graph.

Similarly, each value of the multidimensional array in :file:`data_edges.npy`
encodes the type of link between two nodes, where 0 means there is no such link. Note,
that according to this format there cannot be more than one link (e.g., links
of different types) between a couple of nodes.

In the features file, for each node there is one value (e.g., corresponding
to the scale of this unit).

Dict with dataset description contains a number of keys:

- for `X` in ('train', 'validation, or 'test'):

   - `X_idx` - a list with subset indices,
   - `X_count` - number of samples in `X_idx`,
   - `X_counter` - must be zero,

- `node_num_types` - number of node types (including 0-type), must be (`f` - 1),
- `edge_num_types` - number of edge types (including 0-type),
- `vertexes` - must be equal to `node_num_types`,
- `features_per_node` - number of feartures per node,
- `condition_dim` - number of features representing the generation
  context (goal organization parameters).
