# Zarrmony

Zarrmony converts vendor microscopy formats into OME-Zarr stores. This glossary
fixes the vocabulary used to describe what it writes, so that conversations
about output geometry stay unambiguous.

## Language

### Storage geometry

**Geometry**:
The set of choices that fix an output store's shape rather than its content —
how many pyramid levels there are, what each level's extent is, and how each
level is divided into chunks. Distinct from metadata, which describes the
content, and from the pixels themselves.
_Avoid_: layout (reserved for per-scene / plate / bf2raw output structure)

**Chunk**:
The smallest independently readable unit of an array — one compressed object in
the store, addressed by its grid coordinate. The unit a viewer fetches, decodes,
and culls against the camera.
_Avoid_: block, tile, brick

**Shard**:
A container holding many chunks in a single storage object, with an index that
allows any one chunk to be range-read out of it. A shard changes how many
objects exist, never how finely the array can be read.
_Avoid_: super-chunk, chunk group

**Tile**:
An acquisition-time field of view produced by a stage position on the
microscope, before stitching. A property of the input, never of the output
array.
_Avoid_: using "tile" for a chunk

### Pyramid

**Pyramid level**:
One resolution in a multiscale array, indexed from `0` (full resolution) upward.
Each level is a whole, self-contained image of the same scene.

**Coarse level**:
The pyramid level small enough that a viewer can hold the entire volume at once
and use it as spatial context. A property a level either has or lacks; a pyramid
may contain no such level.
_Avoid_: overview, thumbnail, lowest level

**Detail level**:
The pyramid level a viewer inspects at the current camera, held only over the
region in view rather than whole. Defaults to level `0`.
_Avoid_: full-res, native level

**Anisotropy**:
The ratio between the largest and smallest physical voxel spacing of a level.
A level is isotropic when that ratio is 1 — a voxel spans the same distance on
every axis.
_Avoid_: aspect ratio (reserve that for chunk shape in voxels)

**Scarce axis**:
The axis a microscope samples least finely — Z, for most volumetric light
microscopy. Downsampling it costs information that was never captured densely
to begin with, unlike the oversampled lateral axes.
