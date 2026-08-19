Adding and Removing Transported Fields
======================================

Frontend primitives describe their transported fields in
``_transportDescription()``. The corresponding backend primitive reads those
fields in ``fromTransport(reader, prefix)``. Field names use lower camel case
on both sides.

Adding a scalar field
---------------------

The following example adds ``nonlinearIndex`` to ``Material``.

Add the field to the public frontend API in ``material_library/model.py``.
For the current explicit ``Material`` constructor this consists of its type
annotation, constructor parameter, validation or conversion, and assignment:

.. code-block:: python

   nonlinearIndex: float

   def __init__(self, ..., nonlinearIndex, ...):
       ...
       self.nonlinearIndex = float(nonlinearIndex)

Add the field to ``Material._transportDescription()`` in the same file:

.. code-block:: python

   fields=(
       ...
       transportField("nonlinearIndex"),
   ),

``transportField("nonlinearIndex")`` reads
``material.nonlinearIndex``. If the frontend attribute has a different name,
pass it as the second argument:

.. code-block:: python

   transportField("totalPower", "total_power"),

Add the transport name and backend member to
``include/backend/primitives/Material.hpp``:

.. code-block:: cpp

   struct FieldName
   {
       // ...
       static constexpr char const* nonlinearIndex = "nonlinearIndex";
   };

   double nonlinearIndex{};

Read the field in ``Material::fromTransport()`` in
``src/backend/primitives/Material.cpp``:

.. code-block:: cpp

   reader.assign(result.nonlinearIndex, prefix, FieldName::nonlinearIndex);

The generic graph composer, openPMD writer, and top-level parser obtain the
field through these primitive declarations and require no field-specific
change.

Optional fields
---------------

Use ``optional=True`` when ``None`` is a valid frontend value:

.. code-block:: python

   transportField("coatingThickness", optional=True),

Use ``std::optional`` for the backend member. The assignment call has the same
form as for a required field:

.. code-block:: cpp

   static constexpr char const* coatingThickness = "coatingThickness";
   std::optional<double> coatingThickness;

   reader.assign(result.coatingThickness, prefix, FieldName::coatingThickness);

An absent optional field resets the backend ``std::optional``.

Array fields
------------

Specify the stored axes in their storage order. For example,
``VolumeTopology.points`` changes the frontend point-major array to the
coordinate-major SoA representation:

.. code-block:: python

   transportField(
       "points",
       lambda owner: np.asarray(owner.points).T,
       axes=("coordinate", "point"),
   ),

The backend member uses ``transport::Array<T>``:

.. code-block:: cpp

   static constexpr char const* points = "points";
   transport::Array<double> points;

   reader.assign(result.points, prefix, FieldName::points);

``transport::Array<T>`` contains the flattened values and the transported
shape. A sequence of differently sized arrays uses ``encoding="ragged"`` on
the frontend and ``transport::RaggedArray<T>`` on the backend.

References to primitives
------------------------

Use ``reference`` when a field refers to another transported primitive. The
referenced object supplies its own ``_transportDescription()``. For example,
``Material`` refers to its cross-section table with:

.. code-block:: python

   references=(
       reference("crossSections", optional=True),
   ),

The backend member and assignment are:

.. code-block:: cpp

   static constexpr char const* crossSections = "crossSections";
   std::shared_ptr<CrossSectionTable> crossSections;

   reader.assign(result.crossSections, prefix, FieldName::crossSections);

Use ``many=True`` for a sequence of references. Its backend member is a
``std::vector<std::shared_ptr<T>>``. References to the same frontend object
resolve to the same transported node.

Removing a field
----------------

To remove ``nonlinearIndex`` from ``Material``:

#. Remove ``transportField("nonlinearIndex")`` from
   ``Material._transportDescription()``.
#. Remove the field from the frontend constructor, validation, assignment, and
   public attributes in ``material_library/model.py``.
#. Remove ``FieldName::nonlinearIndex`` and the ``nonlinearIndex`` member from
   ``include/backend/primitives/Material.hpp``.
#. Remove its ``reader.assign`` call from
   ``src/backend/primitives/Material.cpp``.
#. Remove or update tests and examples that construct, transport, or inspect
   the field.

Removing a reference follows the same procedure using the primitive's
``references`` tuple and backend pointer member.

Verification
------------

Frontend transport tests can compose the owning primitive and inspect its
``TransportNode``. Backend transport tests can write an openPMD iteration and
inspect the member produced by ``fromTransport``. Existing examples are in
``tests/python/transport/test_description.py`` and
``tests/transportReader.cpp``.

Run the focused tests extended for the field:

.. code-block:: bash

   python3 -m pytest tests/python/transport/test_description.py --tb=short
   cmake --build build/<task> --target tests_transportReader
   ctest --test-dir build/<task> \
     -R '^tests_transportReader$' \
     --output-on-failure
