"""Issue #139 — a resource-backed source has to stay open across the write.

The Bio-Formats path hands back a dask array that carries its reader's file
handle: the graph is built inside the reader's open context, the handle is
closed on the way out, and the array is expected to reopen it around
``.compute()``. ``write_pyramid`` never calls ``.compute()`` — ``da.store`` and
``da.to_zarr`` run ``dask.compute`` on the graph directly — so that reopen
never fires and the tasks read through a closed handle. They survive only on a
per-task reopen inside the reader, which snapshots "was this closed?" before
taking the reader's lock and therefore lets one worker close the handle under
another.

No format fixture is needed to pin that, because none of it is about a format:
the contract is the resource-backing protocol (``closed`` plus the context
manager methods) and the failure is an ordering between two reader tasks. Both
are reproduced here exactly, with the ordering pinned by events rather than
left to the scheduler — a race that reproduces "usually" is a test that passes
by luck in the direction that matters.
"""

import threading
from contextlib import nullcontext

import dask
import dask.array as da
import numpy as np
import pytest
import zarr

from zarrmony.writers.scene import ZarrmonyWriter, _hold_open

# Only ever paid when the choreography below does not complete — a scheduler
# that ran the two reader tasks one after the other, which would make the test
# vacuous rather than failing. Asserted on, so it cannot pass unnoticed.
_HANDOFF_TIMEOUT = 10.0


class _FakeResource:
    """A file handle in the shape ``resource_backed_dask_array`` expects.

    Three members are the whole protocol — ``closed``, ``__enter__``,
    ``__exit__`` — and the open/close bookkeeping is the reader's rather than a
    refcount: ``__enter__`` snapshots whether the resource was already closed,
    and the matching ``__exit__`` closes it again only if it was. That is the
    TOCTOU #139 turns on, so the fake reproduces it instead of tidying it up; a
    refcounting fake would make the bug unreachable and the test meaningless.

    ``events`` is the ordered log of what the handle actually did, which is
    what the test asserts against — "the handle was open for the whole store"
    is a property of that log, not of the final state.
    """

    def __init__(self) -> None:
        self.closed = True
        self.events: list[str] = []
        self._lock = threading.RLock()
        self._snapshots = threading.local()

    def _record(self, event: str) -> None:
        with self._lock:
            self.events.append(event)

    def __enter__(self) -> "_FakeResource":
        with self._lock:
            stack = getattr(self._snapshots, "stack", None)
            if stack is None:
                stack = self._snapshots.stack = []
            stack.append(self.closed)
            if self.closed:
                self.closed = False
                self._record("open")
        return self

    def __exit__(self, *exc: object) -> bool:
        with self._lock:
            if self._snapshots.stack.pop():
                self.closed = True
                self._record("close")
        return False

    def read(self) -> None:
        """What a reader task does once it holds the reader's lock.

        The error text is the one bffile raises out of
        ``_ensure_java_reader()`` when a close has nulled the Java reader.
        """
        with self._lock:
            if self.closed:
                raise RuntimeError("File not open - call open() first")
            self._record("read")


class _ResourceBackedArray(da.Array):
    """Stand-in for ``resource_backed_dask_array.ResourceBackedDaskArray``.

    A ``da.Array`` subclass carrying ``_context``, which is all
    :func:`~zarrmony.writers.scene._hold_open` looks at — deliberately not the
    real class, since the point of duck-typing the protocol is that no reader's
    identity is involved. ``rechunk`` re-wraps because the real one does (via
    its method proxy), and ``write_pyramid`` rechunks whenever the source's
    blocks do not already match the write grid.
    """

    def __new__(
        cls, dask_graph, name, chunks, dtype=None, meta=None, shape=None, _context=None
    ):
        arr = super().__new__(
            cls, dask_graph, name, chunks, dtype=dtype, meta=meta, shape=shape
        )
        arr._context = _context
        return arr

    @classmethod
    def wrap(cls, arr: da.Array, ctx: _FakeResource) -> "_ResourceBackedArray":
        return cls(
            arr.dask,
            arr.name,
            arr.chunks,
            dtype=arr.dtype,
            meta=arr._meta,
            shape=arr.shape,
            _context=ctx,
        )

    def rechunk(self, *args, **kwargs) -> "_ResourceBackedArray":
        return type(self).wrap(super().rechunk(*args, **kwargs), self._context)


def _two_task_source(
    resource: _FakeResource, handoffs: list[bool]
) -> tuple[da.Array, np.ndarray]:
    """A 2-block source whose tasks interleave the way the failing run did.

    Each block does what a Bio-Formats tile read does: reopen the resource
    around itself, then touch the handle. The events force the one interleaving
    that loses the handle — the second task snapshots the state the *first*
    task opened, so it will not reopen, and it reads after the first task has
    left its context. With the resource closed when the store starts, the first
    task's exit closes the handle and the second task reads through it.

    Holding the resource open across the store makes that unreachable: the
    first task snapshots "open", so its exit closes nothing and the ordering
    below is harmless.
    """
    opened = threading.Event()
    snapshotted = threading.Event()
    released = threading.Event()

    def read_block(block: np.ndarray, block_info=None) -> np.ndarray:
        first = block_info[0]["chunk-location"][-1] == 0
        if first:
            with resource:
                opened.set()
                handoffs.append(snapshotted.wait(_HANDOFF_TIMEOUT))
                resource.read()
            released.set()
        else:
            handoffs.append(opened.wait(_HANDOFF_TIMEOUT))
            with resource:
                snapshotted.set()
                handoffs.append(released.wait(_HANDOFF_TIMEOUT))
                resource.read()
        return block

    base = np.arange(2 * 64 * 64, dtype=np.uint8).reshape(1, 1, 64, 128) % 251
    plain = da.from_array(base, chunks=(1, 1, 64, 64)).map_blocks(
        read_block, dtype=base.dtype, meta=np.empty((0, 0, 0, 0), dtype=base.dtype)
    )
    return _ResourceBackedArray.wrap(plain, resource), base


def _single_level_writer(store_path) -> ZarrmonyWriter:
    """A writer whose only level is the source's own shape and blocking.

    One level keeps the source graph the only thing computed — every level
    above 0 is read back off the store and has no resource behind it — and
    matching the chunk shape to the source keeps the two reader tasks two
    tasks.
    """
    return ZarrmonyWriter(
        store=str(store_path),
        level_shapes=[(1, 1, 64, 128)],
        dtype=np.uint8,
        zarr_format=3,
        chunk_shape=[[1, 1, 64, 64]],
    )


def test_resource_backed_source_is_held_open_across_the_write(tmp_path) -> None:
    resource = _FakeResource()
    handoffs: list[bool] = []
    src, expected = _two_task_source(resource, handoffs)
    writer = _single_level_writer(tmp_path / "held.zarr")

    with dask.config.set(scheduler="threads", num_workers=4):
        writer.write_pyramid(src)

    assert all(handoffs), "the two reader tasks did not overlap; test is vacuous"

    g = zarr.open_group(str(tmp_path / "held.zarr"), mode="r")
    np.testing.assert_array_equal(g["0"][:], expected)

    reads = [i for i, e in enumerate(resource.events) if e == "read"]
    assert len(reads) == 2
    assert (
        "close" not in resource.events[reads[0] : reads[-1]]
    ), f"the handle was closed mid-store: {resource.events}"
    # Opened once for the whole write, and left the way we found it — the
    # writer borrows the handle, it does not take it over.
    assert resource.events.count("open") == 1
    assert resource.closed is True


def test_an_unwrapped_source_is_written_unchanged(tmp_path) -> None:
    """No ``_context``, no context: an ordinary dask array must not acquire a
    guard, and must write exactly as it did before #139.
    """
    base = np.arange(1 * 1 * 64 * 128, dtype=np.uint8).reshape(1, 1, 64, 128) % 251
    plain = da.from_array(base, chunks=(1, 1, 64, 64))
    assert isinstance(_hold_open(plain), nullcontext)

    writer = _single_level_writer(tmp_path / "plain.zarr")
    with dask.config.set(scheduler="threads", num_workers=4):
        writer.write_pyramid(plain)

    g = zarr.open_group(str(tmp_path / "plain.zarr"), mode="r")
    np.testing.assert_array_equal(g["0"][:], base)


def test_an_already_open_resource_is_left_alone(tmp_path) -> None:
    """A caller holding the handle open keeps it open.

    Entering an already-open context would close it on the way out, which is
    the same handle loss from the other direction — so an open resource gets a
    ``nullcontext`` and the store leaves its state alone.
    """
    resource = _FakeResource()
    src, _ = _two_task_source(resource, [])
    with resource:
        assert resource.closed is False
        assert isinstance(_hold_open(src), nullcontext)
    assert resource.closed is True
    assert _hold_open(src) is resource


@pytest.mark.parametrize("attr", [object(), None])
def test_a_context_without_closed_is_not_treated_as_a_resource(attr) -> None:
    """``_context`` alone is not the protocol — ``closed`` is what says the
    object can report its own open state, and entering something that cannot is
    a guess about a stranger's lifecycle.
    """
    base = da.zeros((4, 4), chunks=(2, 2), dtype=np.uint8)
    wrapped = _ResourceBackedArray.wrap(base, attr)
    assert isinstance(_hold_open(wrapped), nullcontext)
