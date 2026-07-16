:doc:`<- Back to overview <index>`

Getting Started
===============

This page is a compact installation guide for a source checkout of HASEonGPU.
For modeling concepts, see :doc:`Theory and Model <theoryAndModel>`.  For the
main user workflow, continue with :doc:`Python Interface Guide <pythonInterface>`
after installation.

1. Clone the Repository
-----------------------

.. code-block:: bash

   git clone https://github.com/computationalradiationphysics/haseongpu.git
   cd haseongpu

2. Install Prerequisites
------------------------

Required tools are:

* ``Python >= 3.10`` with ``pip``
* ``cmake`` and ``ninja``
* a C++20 compiler, tested with ``gcc >= 12`` and ``clang >= 17``
* an openPMD-api provider for the storage backend you want to use, or the
  bundled provider selected by ``hase-configure``

Optional dependencies depend on the run mode:

* CUDA Toolkit ``>= 12.5`` for CUDA GPU builds, or ROCm/HIP ``>= 6.2.4`` for
  HIP GPU builds
* OpenMPI for MPI runs
* ParaView for VTK visualization
* ``matplotlib`` for helper plotting scripts

Windows support is experimental; see :doc:`windows`.

3. Create a Python Environment
------------------------------

Use a virtual environment unless your site already provides a managed Python
module or Conda environment:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -U pip

If you use Conda, Spack, or environment modules for openPMD-api, activate/load
that environment before configuring HASEonGPU so Python and CMake see the same
provider.

4. Run the Guided Configurator
------------------------------

From the source checkout, run:

.. code-block:: bash

   python3 utils/configure_hase.py

After HASEonGPU is installed, the same helper is available as:

.. code-block:: bash

   hase-configure

The configurator asks only for choices that affect installation or runtime
selection:

* openPMD provider: auto, bundled, or system
* ADIOS2/HDF5 handling for the selected provider
* runtime openPMD backend: ``adios-sst``, ``adios``, or ``hdf5``
* Alpaka compute backend
* single-process or MPI mode
* native CPU optimizations for the local machine
* whether to run the printed install command immediately

The script writes a small PhiASE YAML run-control file to
``config/hase-phiase.yaml`` by default, prints the exact install command, and
finishes with guidance for the selected openPMD backend, MPI setting, and
available compute backends.  The generated YAML contains the default ray range,
relative-standard-error threshold, and compute settings; geometry, spectra,
pump settings, and material state are still constructed in Python.

For the bundled provider, the generated package configuration records the
matching build-local ``openpmd_api`` path and native runtime root. Start a fresh
Python process after rebuilding; do not install an unrelated PyPI
``openpmd-api`` wheel into that environment. The installed frontend then uses
the matching local provider automatically.

Useful non-interactive options include ``--autoinstall``, ``--reinstall``,
``--use-ccache``, ``--runtime-dir``, ``--provider``, ``--openpmd-backend``, and
``--output``. The default install creates or reuses the shared native runtime
under ``build/``. ``--runtime-dir`` selects a different shared native build
directory when needed. Run
``python3 utils/configure_hase.py --help`` for the complete list.

5. Install
----------

The configurator prints a command of this form:

.. code-block:: bash

   CMAKE_ARGS="<selected CMake options>" python3 -m pip install -v .

Run the printed command if you did not let the configurator install
immediately. ``python3 -m pip install -v .`` uses a lightweight
``build/{wheel_tag}`` tree for wheel staging and creates or incrementally builds
the native runtime under ``build/``. A normal ``cmake -S . -B build`` build and
the Python installation therefore share ``hase-cpp``, openPMD, ADIOS2/HDF5,
and Alpaka. Only the Python frontend is exported into site-packages.
``CMAKE_ARGS`` is how you pass build options such as the openPMD provider, MPI
mode, Alpaka choices, and native CPU optimization setting.

If pip reports an externally managed Python environment, prefer a virtual
environment.  Use ``--break-system-packages`` with the configurator only when
you intentionally install into such an environment.

6. Verify and Continue
----------------------

Check that the package imports:

.. code-block:: bash

   python3 -c "import HASEonGPU; print(HASEonGPU.__version__)"

For the recommended user workflow, continue with
:doc:`Python Interface Guide <pythonInterface>`.  Use
:doc:`Binary Interface <binaryInterface>` only when running ``hase-cpp``
directly, and :doc:`CMake Build Options <compilation>` when you need manual
CMake configuration.
