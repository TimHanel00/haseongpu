Interpret forward ASE uncertainty
=================================

Forward ASE results include a relative standard error (RSE) for every cell.
Use this value to decide whether the sampled ray histories provide sufficient
precision for your simulation.

Set an RSE target
-----------------

Configure the target and the available ray range with ``PhiASE``:

.. code-block:: python

   phi_ase = PhiASE(
       minRays=100_000,
       maxRays=1_000_000,
       adaptiveSteps=4,
       numIndependentRayPopulations=8,
       relativeStandardErrorThreshold=0.05,
   )

The solver adds ray histories until every cell reaches the target or
``maxRays`` is reached. A threshold of ``0.05`` requests an estimated
one-standard-error uncertainty of 5% relative to the cell value.

Read the result
---------------

After the run, inspect ``result.relativeStandardError`` together with
``result.phiAse`` and ``result.totalRays``. Cells that remain above the target
at ``maxRays`` need a larger ray budget if lower sampling uncertainty is
required.

How the RSE is estimated
------------------------

HASEonGPU divides the global ray budget into ``numIndependentRayPopulations``
independent ray populations (default 8). Each
population samples the complete source and wavelength distributions and produces a
complete cell-field estimate. The variation between these estimates supplies
the reported RSE.

Each population contribution
is normalized by its own history count, and the reported field is the equal-weight
mean of these complete-source estimates. RSE uses the sample variance of those
same estimates divided by the number of populations. Worker count does not
select the number of statistical replicates.

Every emitting domain needs at least one ray in every population. A budget that
cannot satisfy this is rejected; the configured population count is not silently
reduced. Adaptive evaluations can be delayed until all sources can be represented.
A single population
cannot estimate uncertainty and reports the maximum-error sentinel. A zero mean
has undefined RSE (NaN). Source strata and wavelength strata use independent
randomization to avoid coupling those sampling dimensions.

Each domain has its own unnormalized source CDF starting from zero. This avoids
losing weak sources through subtraction of cumulative totals from stronger
domains. A domain with zero inversion has exactly zero spontaneous source
strength: automatic allocation assigns it no primary rays, but transmitted rays
still score its field. Zero cell contributions from a population remain part of
the estimate, not missing observations.

SRM logical batches contain at most 65,536 primary rays from one source domain
and population. Each batch owns its reservoir throughout transport, irrespective
of its worker. Raw batch scores are summed within the population before
normalization; batches are not additional RSE samples. Changing this fixed cap
can affect variance, while changing worker ownership only affects execution
and floating-point reduction order.

Limitations
-----------

RSE measures Monte Carlo sampling uncertainty. It does not measure mesh
discretization error, uncertainty in material data, finite boundary-pass
truncation, or model error. Direct transport resamples boundary candidates per
complete population; SRM resamples within fixed logical batches. The randomness
of either operation contributes to the complete population estimate and its RSE.
Eight populations provide only seven variance degrees of freedom; reported RSE
is an estimate, not a bound or a guaranteed accuracy level.
