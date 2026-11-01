# Fields and spaces

A field gives values meaning on a topology. Its **space** identifies where
they belong and, when it supplies a basis, how they define a function inside
cells. Its **array** supplies coefficients; its **unit** describes the quantity.

On the [two-cell dataset](01-datasets.md#a-simpler-first-order-field),
temperature has twelve shared point values and material has two cell values.
The choice belongs to each field, not to the mesh as a whole.

## Choose where values live

| Quantity | Common space | Interpretation |
|---|---|---|
| Shared point temperature | `H1(data)` | First-order reconstruction from shared corners |
| Material or cell average represented as constant | `Constant(data)` | Independent constant function in each cell |
| Discontinuous nodal solution | `L2(data, order=1)` | Each cell has its own corner coefficients |
| Face pressure or label | `Values(data, "faces")` | Attribute; no cell-interior basis |
| Normal velocity or integrated flux | `Values(data, "faces", oriented=True)` | Signed along the stored normal |
| Point, cell or edge attribute | `Values(data, on)` | Association without a declared basis |

## Association and reconstruction

Association says which entities own values. Reconstruction says what a
value between them means. Point samples do not uniquely specify that choice.
H1 order 1 chooses linear interpolation on triangles/tetrahedra, bilinear
on quads and trilinear on hexes. It is useful for visualization of structured
samples regardless of the simulation's discretization method.

A point sample at a cell center and a cell average are different quantities.
Constant can give either a piecewise-constant reconstruction but does not
convert one to the other. Face association alone does not identify normal
velocity versus flux already multiplied by area. The
[FV workflow](18-finite-volume.md) makes the distinction at each operation.

`Values` declares no basis. Some current algorithms add an interpretation:
the shared algorithm helper reads point Values as H1 order 1 and cell Values
as Constant. Check the operation's semantics; association alone does not
suppress interpolation everywhere.

## Sharing and independent cell values

| Space on the two-cell mesh | Coefficients | Shared face |
|---|---:|---|
| `H1(data)` | 12 | Four shared face-corner coefficients |
| `Constant(data)` | 2 | One independent value on each side |
| `L2(data, order=1)` | 16 | Four independent corner coefficients on each side |

H1 can represent continuous temperature. L2 permits different values at the
same interface location, as DG needs. Constant preserves a material jump.
Connectivity and geometry remain the same in all three cases.

![Cell constants, shared point values and independent cell-local values](../assets/data-spaces.png)

The middle ramp is introduced by averaging and interpolation; it is not a
transition resolved by the original two material values.
[Create and Convert Fields](create-fields.md#a-material-interface-exposes-the-difference)
checks the operations producing this figure.

## Basis, order and value width

A basis evaluates a field from coefficients. Order controls the local
approximation, sharing controls how cells meet, and value width distinguishes
scalar and vector coefficients. Storage layout is another independent choice.

H1 order 2 adds field nodes on edges and, for supported shapes, faces and
interiors. These are not new corner vertices or subdivided cells. The two
hexes need 45 shared quadratic coefficients while retaining twelve vertices.
A vector has one vector per coefficient, not additional mesh points.

The [optional function-space chapter](00-function-spaces.md) develops the
basis formula, regularity, DOF counts and higher-order geometry.
[Cell Types and Bases](16-cells-and-bases.md) records supported combinations.
Attaching a field does not guarantee every operation supports that combination.

## Face values and orientation

Normal components and integrated fluxes change sign under face reversal.
Declare `oriented=True` so a filter reversing a carried face negates them.
Leave pressure, area and labels unoriented. The flag cannot verify an imported
solver's sign convention or face numbering.

`SideTraces(data)` keeps both cells' values at face corners. The
[trace recipe](create-fields.md#read-both-sides-of-a-face) compares material
and continuous-temperature jumps. Face ids and sides come from
[topology](topology-and-geometry.md#points-edges-faces-and-cells).

## A space belongs to a topology

Allocate using `field.space.size`. Equal lengths on different meshes do not
make fields interchangeable. Spaces are interned on their topology: repeated
requests for the same space reuse its layout. Fields can share that space
while keeping their own values and units.

Fields can share a name in different spaces. If the name is ambiguous, select
`(name, space)` explicitly. The [selection recipe](create-fields.md#fields-that-share-a-name)
shows insertion, iteration, filtering and VTK round trips.

`Constant(data)` describes a piecewise-constant function; `ConstantArray`
supplies one repeated array value. Stored and implicit arrays can supply a
space without changing its meaning. See [Arrays, Storage and Layouts](field-storage.md)
when attaching solver storage or writing advanced kernels.

Continue with [Coordinates, Units and Time](12-units.md), or
[Create and Convert Fields](create-fields.md) to put these choices to work.

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="fields-spaces-and-implicit-arrays" data-guide-forward="#fields-and-spaces"></span>
    [Fields, spaces and implicit arrays](03-fields-and-spaces.md#fields-and-spaces).

    <span id="fields-that-share-a-name" data-guide-forward="../create-fields/#fields-that-share-a-name"></span>
    [Fields that share a name](create-fields.md#fields-that-share-a-name).

    <span id="construct-an-l2-field" data-guide-forward="../create-fields/#construct-an-l2-field"></span>
    [Construct an L2 field](create-fields.md#construct-an-l2-field).

    <span id="initialize-each-cells-coefficients" data-guide-forward="../create-fields/#initialize-each-cells-coefficients"></span>
    [Initialize each cell's coefficients](create-fields.md#initialize-each-cells-coefficients).

    <span id="copy-coefficients-from-an-existing-solver-array" data-guide-forward="../create-fields/#copy-coefficients-from-an-existing-solver-array"></span>
    [Copy coefficients from an existing solver array](create-fields.md#copy-coefficients-from-an-existing-solver-array).

    <span id="a-constant-inside-each-cell-l2-order-0" data-guide-forward="../create-fields/#a-constant-inside-each-cell-l2-order-0"></span>
    [A constant inside each cell: L2 order 0](create-fields.md#a-constant-inside-each-cell-l2-order-0).

    <span id="evaluate-and-check-the-discontinuity" data-guide-forward="../create-fields/#evaluate-and-check-the-discontinuity"></span>
    [Evaluate and check the discontinuity](create-fields.md#evaluate-and-check-the-discontinuity).

    <span id="construct-a-quadratic-h1-field" data-guide-forward="../create-fields/#construct-a-quadratic-h1-field"></span>
    [Construct a quadratic H1 field](create-fields.md#construct-a-quadratic-h1-field).

    <span id="sample-at-the-fields-nodes" data-guide-forward="../create-fields/#sample-at-the-fields-nodes"></span>
    [Sample at the field's nodes](create-fields.md#sample-at-the-fields-nodes).

    <span id="evaluate-through-the-quadratic-basis" data-guide-forward="../create-fields/#evaluate-through-the-quadratic-basis"></span>
    [Evaluate through the quadratic basis](create-fields.md#evaluate-through-the-quadratic-basis).

    <span id="a-material-interface-exposes-the-difference" data-guide-forward="../create-fields/#a-material-interface-exposes-the-difference"></span>
    [A material interface exposes the difference](create-fields.md#a-material-interface-exposes-the-difference).

    <span id="projection-is-an-algorithmic-choice" data-guide-forward="../create-fields/#projection-is-an-algorithmic-choice"></span>
    [Projection is an algorithmic choice](create-fields.md#projection-is-an-algorithmic-choice).

    <span id="read-both-sides-of-a-face" data-guide-forward="../create-fields/#read-both-sides-of-a-face"></span>
    [Read both sides of a face](create-fields.md#read-both-sides-of-a-face).

    <span id="run-and-reproduce-the-figure" data-guide-forward="../create-fields/#run-and-reproduce-the-figure"></span>
    [Run and reproduce the figure](create-fields.md#run-and-reproduce-the-figure).

    <span id="how-the-values-are-stored" data-guide-forward="../field-storage/#how-the-values-are-stored"></span>
    [How the values are stored](field-storage.md#how-the-values-are-stored).

    <span id="the-mesh-points-cells-edges-and-faces" data-guide-forward="../field-storage/#the-mesh-points-cells-edges-and-faces"></span>
    [The mesh: points, cells, edges and faces](field-storage.md#the-mesh-points-cells-edges-and-faces).

    <span id="where-each-value-is" data-guide-forward="../field-storage/#where-each-value-is"></span>
    [Where each value is](field-storage.md#where-each-value-is).

    <span id="a-cells-nodes-in-order" data-guide-forward="../field-storage/#a-cells-nodes-in-order"></span>
    [A cell's nodes in order](field-storage.md#a-cells-nodes-in-order).

    <span id="h1-above-order-1-numbered-by-entity" data-guide-forward="../field-storage/#h1-above-order-1-numbered-by-entity"></span>
    [H1 above order 1: numbered by entity](field-storage.md#h1-above-order-1-numbered-by-entity).

    <span id="vectors" data-guide-forward="../field-storage/#vectors"></span>
    [Vectors](field-storage.md#vectors).

    <span id="every-value-of-a-cell-at-any-order" data-guide-forward="../field-storage/#every-value-of-a-cell-at-any-order"></span>
    [Every value of a cell, at any order](field-storage.md#every-value-of-a-cell-at-any-order).

    <span id="finding-values-on-the-host" data-guide-forward="../field-storage/#finding-values-on-the-host"></span>
    [Finding values on the host](field-storage.md#finding-values-on-the-host).

    <span id="where-a-cells-values-sit" data-guide-forward="../field-storage/#where-a-cells-values-sit"></span>
    [Where a cell's values sit](field-storage.md#where-a-cells-values-sit).

    <span id="values-can-be-computed-instead-of-stored" data-guide-forward="../field-storage/#values-can-be-computed-instead-of-stored"></span>
    [Values can be computed instead of stored](field-storage.md#values-can-be-computed-instead-of-stored).

    <span id="geometrys-space-and-its-implicit-array-are-different-layers" data-guide-forward="../field-storage/#geometrys-space-and-its-implicit-array-are-different-layers"></span>
    [Geometry's space and its implicit array are different layers](field-storage.md#geometrys-space-and-its-implicit-array-are-different-layers).

    <span id="other-implicit-coefficient-arrays" data-guide-forward="../field-storage/#other-implicit-coefficient-arrays"></span>
    [Other implicit coefficient arrays](field-storage.md#other-implicit-coefficient-arrays).

    <span id="higher-orders" data-guide-forward="../00-function-spaces/#higher-orders"></span>
    [Higher orders](00-function-spaces.md#higher-orders).

    <span id="know-the-present-higher-order-limits" data-guide-forward="../16-cells-and-bases/#what-each-shape-supports"></span>
    [Know the present higher-order limits](16-cells-and-bases.md#what-each-shape-supports).
