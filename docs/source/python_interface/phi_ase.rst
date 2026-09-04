PhiASE
======

``PhiASE`` configures the forward, source-driven ASE estimator. ``Simulation``
supplies its gain domains and the authoritative spectra from each component's
``Material``. ``PhiASE`` owns numerical sampling, reflection, compute,
transport, and parallel controls; it does not own geometry, material, or the
evolving excitation state.
``propagationMode="forward"`` is the only supported mode.

.. code-block:: python

   phi_ase = PhiASE(
       minRays=100_000,
       maxRays=1_000_000,
       adaptiveSteps=4,
       relativeStandardErrorThreshold=0.05,
       enableDiagnostics=True,
       useReflections=True,
       reflectionMode="direct",
       backend="Host_Cpu_CpuSerial",
       openpmdBackend="auto",
       rngSeed=1234,
       ase_steps=150,
   )

Normal applications pass it to ``Simulation``. The direct ``run`` entry point
accepts the same physical ``GainMedium`` graph. Pass ``opticalComponents`` when
the trace also traverses passive components. Spectra are obtained from the
``Material`` referenced by each ``OpticalComponent``; ``PhiASE`` does not
create a second transport copy:

.. code-block:: python

   phi_ase.run(
       gainMedium=medium,
       opticalComponents=components,
       initialExcitation=0.25,
   )
   result = phi_ase.getResults()
   phi = np.asarray(result.phiAse)

``getResults`` raises ``RuntimeError`` before a successful run.
The result includes ``phiAse``, ``standardError``,
``relativeStandardError``, ``totalRays``, and ``dndtAse`` plus boundary-pass
termination information when reflection or inter-component transmission is
active. A time-stepped
``Simulation`` exposes the same raw object as ``TimeStepState.aseResult``.

Sampling controls
-----------------

``ase_steps``
   Initial outer simulation steps that include ASE. ``None`` and zero disable
   ASE in ``Simulation``. Direct one-state ``PhiASE.run`` calls are unaffected.

``minRays`` and ``maxRays``
   Initial and maximum global history counts. Adaptive execution adds
   geometrically growing batches until every cell reaches the requested RSE or
   the maximum is reached.

``adaptiveSteps``
   Maximum geometric count increases between the two ray limits.

``forwardRayCount``
   Fixed global history count. Setting it disables adaptive count selection.

``relativeStandardErrorThreshold``
   Target one-sigma uncertainty relative to each cell's estimated mean. ``0.05``
   requests 5%. It measures sampling uncertainty, not discretization or model
   error.

``enableDiagnostics``
   Enable per-cell ray visits in ``totalRays`` and failed-ray counts used to
   validate the trace. Use diagnostics for a first run on a new or refined
   mesh, then disable them for performance after confirming that no rays are
   dropped. Diagnostics use 128 threads per block and retain the counters and
   atomics needed to report failures. The performance specialization uses 512
   threads per block and compiles those diagnostics out; ``totalRays`` and the
   dropped-ray array then retain their normal cell-shaped layouts but contain
   zeros. ``trackRayVisits`` remains accepted as a deprecated constructor and
   transport input alias, but new output contains only ``enableDiagnostics``.

``rngSeed``
   Unsigned seed for reproducible ASE histories. If omitted, each invocation
   draws a process-local seed.

``monochromatic``
   Use the first absorption and emission samples instead of integrating the
   spectrum.

Each direct history samples a spectral bin, a source cell with probability
proportional to its spontaneous-source strength, a uniform point in that Tet4
cell, and an isotropic direction. The global count is divided among optical
components from their current total source strengths. Setting
``OpticalComponent.aseRays`` reserves that component's exact final primary-ray
count; unspecified positive-source components share the remainder. It then
deposits a gain-weighted track-length score in every traversed cell. Spectral
bins and source cells are stratified within each domain and statistical batch. See
:ref:`forward-ase-model` for normalization and uncertainty equations.

Forward traversal has no fixed cell-crossing limit. A valid ray continues until
it reaches a physical boundary or a cell policy terminates it, so increasing
mesh resolution cannot make a ray fail merely because it requires more
crossings. With diagnostics enabled, invalid geometric transitions and
non-finite contributions are counted as dropped rays.

Domain boundaries
-----------------

``useReflections`` enables specular ASE reflection on domain-assigned
``SurfaceOptics``. Direct and reflected rays travel to a physical mesh boundary;
there is no configurable forward ray-length cutoff.

``reflectionMode``
   Selects the boundary-history representation. ``"direct"`` (the default)
   stores up to two compact exact-intersection children per boundary hit and
   performs systematic particle combing. ``"srm"`` instead offers both children
   to a bounded weighted sample per mesh face. Both policies route transmitted
   histories between adjacent optical components and keep their large buffers,
   scans, and selections on the accelerator.

``surfaceReservoirSize``
   Number of statistically retained ray records per boundary face when
   ``reflectionMode="srm"``. Reflected and transmitted weight is accumulated
   independently of this bounded record count.

``srmPositionMode``
   Selects where retained SRM records are relaunched. ``"exact"`` retains each
   sampled boundary intersection, while ``"centroid"`` allocates no device
   position buffers and relaunches every record at the centroid of its owning
   face. Directions, weights, wavelengths, and selection keys remain bounded by
   faces times ``surfaceReservoirSize``. This setting affects only
   ``reflectionMode="srm"``; direct mode always uses exact intersections.

``boundaryMaxPasses``
   Hard limit for direct or SRM boundary passes. ``None`` chooses a limit from
   the number of domains and the reflection iteration setting.

``reflectionMaxIterations``
   Legacy reflection-pass setting used when deriving the automatic boundary
   cap. For ``reflectionMode="srm"``, the positive integer environment
   override is ``HASE_SRM_MAX_ITERATIONS``.

``reflectionTolerance``
   Stop when remaining reflected source weight, relative to the direct pass,
   falls below this fraction.

The runtime reports ``boundaryStatus``, ``boundaryPasses``,
``boundaryRemainingFraction``, ``boundaryMaxPasses``, and
``boundaryDivergenceStreak``. It also reports ``boundaryTailStatus``,
``boundaryGamma``, ``boundaryGammaStandardError``, ``boundaryTailFactor``, and
``boundaryTailClosure`` for analytical completion of a truncated reflected
series. Terminal status can be ``converged``, ``stable``,
``diverged``, or ``maxPasses``; ``disabled`` means neither reflections nor
inter-component routing required boundary passes.
``HASE_SRM_DIVERGENCE_STREAK`` controls how many consecutive growing SRM passes
report divergence.

For a truncated series, HASE fits the recent reflected population as
:math:`W_p \simeq W_0\Gamma^p`. When the multiplier is confidently below one,
stationary over a longer window, and consistent with the weight removed by the
last pass, the remaining Neumann series is added as the final pass contribution
times :math:`\Gamma/(1-\Gamma)`. ``boundaryTailStatus="applied"`` identifies
that completion. ``"refused"`` means the frozen-inversion field has not
established a finite stationary continuation; no analytical tail is added.

With ``useReflections`` enabled, each eligible interface hit creates both
histories: the reflected child has weight ``R W`` and the transmitted child has
weight ``(1-R) W``. Total internal reflection creates only a reflected child
with weight ``W``. Disabling reflections discards the reflected contribution.
Particle combing then restores the configured per-domain population: discarded
histories transfer their represented weight to selected histories, and
histories may be duplicated when a domain has too few candidates. The model
uses configured constant reflectivity and Snell refraction; it does not
calculate Fresnel or polarization-dependent coefficients. See
:ref:`ase-surface-reflections`.

Compute and transport
---------------------

``backend`` selects an Alpaka compute backend reported by
``AlpakaBackends.all()``. ``openpmdBackend`` independently selects the transport
format/engine. See :doc:`../backendSelection` for selection syntax and
:doc:`../openpmdTransport` for available storage backends and provider
compatibility.

``parallelMode="single"`` runs one process. ``parallelMode="mpi"`` asks the
frontend to launch through MPI; ``nPerNode`` selects ranks per node and
``numDevices`` limits devices available on each node. See :doc:`../mpi`.

``minSampleRange`` and ``maxSampleRange`` optionally restrict the inclusive
flattened cell range. Normal full-volume runs leave them unset.

YAML and CLI helpers
--------------------

``fromYaml`` accepts schema-v3 PhiASE settings under ``simulation.phi_ase``:

.. code-block:: yaml

   schema_version: 3
   simulation:
     phi_ase:
       min_rays: 100000
       max_rays: 1000000
       relative_standard_error_threshold: 0.05
       enable_diagnostics: true
       adaptive_steps: 4
       ase_steps: 150
       use_reflections: true
       reflection_mode: srm
       surface_reservoir_size: 256
       srm_position_mode: centroid
       reflection_max_iterations: 40
       boundary_max_passes: 256
       backend: Host_Cpu_CpuSerial
       openpmd_backend: auto
       parallel_mode: single
       rng_seed: 1234

.. code-block:: python

   phi_ase = PhiASE.fromYaml(
       "config/hase-phiase.yaml",
       maxRays=2_000_000,
   )

Keyword arguments override file values. Material spectra are attached later by
``Simulation``. ``addArguments`` and ``fromArgs`` add the same controls to an
``argparse`` command. Boolean pairs allow either a constructor default or a
loaded YAML value to be overridden explicitly:

.. code-block:: console

   --use-reflections | --no-reflections
   --enable-diagnostics | --disable-diagnostics
   --monochromatic | --polychromatic
   --write-vtk | --no-write-vtk

Explicit device IDs use ``--devices ID [ID ...]``. Inclusive sample bounds use
``--min-sample-range`` and ``--max-sample-range``; ``--ase-steps`` controls the
outer ASE-active step count.
The deprecated ``track_ray_visits`` YAML key and
``--track-ray-visits``/``--no-track-ray-visits`` CLI spellings remain accepted
as input aliases.
