Monte Carlo uncertainty and RSE
================================

The ASE result is a Monte Carlo estimate. HASEonGPU reports its sampling
uncertainty separately for every Tet4 cell and can use that uncertainty to
adapt the number of forward rays. The relevant solver setting is
``relativeStandardErrorThreshold``:

.. code-block:: python

   aseSolver = MonteCarloASESolver(minRays=100_000, maxRays=1_600_000,
                                    relativeStandardErrorThreshold=0.05)

``0.05`` requests a one-standard-error sampling uncertainty no larger than 5%
of the estimated ASE flux in every cell. It is a numerical accuracy target,
not a material property and not a five-percent bound on the total modeling
error.

How RSE is calculated
---------------------

For cell :math:`j`, let :math:`X_{rj}` be the score deposited by globally
launched forward history :math:`r`. A history that does not visit the cell has
:math:`X_{rj}=0`; those zero-score histories remain part of the statistics.
For :math:`N` launched histories, the backend accumulates

.. math::

   S_{1j} = \sum_{r=1}^{N} X_{rj},
   \qquad
   S_{2j} = \sum_{r=1}^{N} X_{rj}^{2}.

It then evaluates exactly

.. math::

   \operatorname{RSE}_j
   = \sqrt{\max\!\left(0,
       \frac{N S_{2j}/S_{1j}^{2}-1}{N}\right)}.

The physical source normalization and the cell-volume scaling cancel in this
ratio. ``relative_standard_error`` is therefore dimensionless. The absolute
``standard_error`` is

.. math::

   \operatorname{SE}_j
   = \operatorname{RSE}_j\,\left|\Phi_j\right|,

and has the same physical unit as ``phiAse``.

For example, scores ``[1, 3, 0, 0]`` give :math:`S_1=4`, :math:`S_2=10`,
:math:`N=4`, and an RSE of approximately ``0.612``. Omitting the two
zero-score histories would understate the uncertainty of the cell estimate.

How the threshold controls a run
--------------------------------

Adaptive execution first launches ``minRays`` histories. If at least one cell
has a non-finite RSE or an RSE above the threshold, it adds histories in
geometrically increasing global batches. At most ``adaptiveSteps`` increases
span the interval from ``minRays`` to ``maxRays``. Accumulators are retained,
so later checks use every history launched so far rather than replacing the
earlier estimate.

The run stops when either:

* every cell has a finite RSE at or below the threshold; or
* the ``maxRays`` budget has been evaluated.

Reaching ``maxRays`` is not itself proof that the threshold was met. Always
inspect the reported RSE when convergence matters:

.. code-block:: python

   state = simulation.step().getLastState()
   print(np.nanmax(state.volumeRelativeStandardError))

Setting ``minRays == maxRays`` requests one fixed budget, so the threshold is
reported but cannot trigger another batch. ``forwardRayCount`` likewise
selects a fixed global history count and disables adaptive refinement:

.. code-block:: python

   aseSolver = MonteCarloASESolver(forwardRayCount=1_000_000,
                                    relativeStandardErrorThreshold=0.05)

Undefined and invalid estimates
-------------------------------

If a cell's accumulated score is exactly zero, its ASE estimate and absolute
standard error are zero but its relative error is undefined; HASEonGPU reports
that RSE as ``NaN``. Fewer than two histories, non-finite accumulators, or a
dropped traversal produce an invalid RSE sentinel. Undefined and invalid RSEs
do not satisfy the adaptive stopping criterion, so such a run continues to its
maximum budget.

Choosing a threshold
--------------------

``0.1`` allows 10% estimated relative sampling uncertainty and ``0.05`` allows
5%. A practical study can begin with a looser pilot threshold and tighten it
until the derived observables of interest are stable. In the usual Monte Carlo
regime RSE decreases approximately as :math:`1/\sqrt{N}`; halving a threshold
can therefore require roughly four times as many histories.

The all-cell stopping rule makes weakly illuminated or rarely visited cells the
likely cost driver. A low RSE also says nothing about mesh convergence,
time-step error, uncertain material data, or missing physical effects. Check
those errors independently rather than interpreting the RSE threshold as a
complete accuracy guarantee.

Result fields
-------------

After a completed step, the main cell-wise diagnostics are:

* ``state.volumePhiAse`` -- ASE flux estimate;
* ``state.volumeStandardError`` -- absolute sampling standard error;
* ``state.volumeRelativeStandardError`` -- dimensionless RSE;
* ``state.volumeTotalRays`` -- ray visits deposited in each cell, not the
  global history count :math:`N` used in the RSE formula.

A short convergence check can report both the worst defined RSE and cells that
did not obtain a finite relative estimate:

.. code-block:: python

   rse = state.volumeRelativeStandardError
   print(np.nanmax(rse), np.count_nonzero(~np.isfinite(rse)))
