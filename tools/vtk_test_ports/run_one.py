# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Run one of VTK's Python tests as VTK's test driver does (``rtImageTest``), with
VTK's filters (``vtk``) or with Tack's in their place (``tack``, ``shims.py``):

    python run_one.py <test.py> vtk|tack <output directory> <data root> [<baseline png>]

Rendering is offscreen. The test's image goes to the output directory, compared
with the baseline by vtkTesting (its ``ImageError`` measurement on stdout); with
``tack``, the stand-ins' records go to ``records.json`` there."""

import atexit
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def offscreen():
    """Every render window offscreen: ``vtkRenderWindow`` (in its module, and in
    ``vtkmodules.all`` and ``vtk`` when imported) made one that renders offscreen.
    VTK overrides a class only with a pure Python subclass, so the name is wrapped."""
    import vtkmodules.vtkRenderingCore as core
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 -- the render window factory

    real = core.vtkRenderWindow

    def vtkRenderWindow(*args, **kwargs):
        window = real(*args, **kwargs)
        window.SetOffScreenRendering(1)
        return window

    core.vtkRenderWindow = vtkRenderWindow
    import vtkmodules.all as everything

    everything.vtkRenderWindow = vtkRenderWindow


def main():
    test, mode, out, data = sys.argv[1:5]
    baseline = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(out, exist_ok=True)
    offscreen()
    if mode == "tack":
        sys.path.insert(0, HERE)
        import shims

        import tack

        tack.set_default_device("cpu")
        replaced = shims.install()
        shims.CURRENT["test"] = test

        def write():
            with open(os.path.join(out, "records.json"), "w") as f:
                json.dump({"replaced": replaced, "records": shims.LOG}, f, indent=1,
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

        atexit.register(write)
    from vtkmodules.test import rtImageTest
    from vtkmodules.vtkTestingRendering import vtkTesting

    args = [test, "-D", data, "-T", out]
    if baseline:
        args += ["-V", baseline, "-B", os.path.dirname(baseline)]
    sys.argv = ["rtImageTest.py", *args]
    try:
        code = rtImageTest.main(test) if baseline else run_without_image(test)
    except SystemExit as exit:                      # unittest-style tests (Testing.main)
        code = vtkTesting.PASSED if exit.code in (0, None) else vtkTesting.FAILED
    print(f"RESULT {'PASSED' if code == vtkTesting.PASSED else 'FAILED'}", flush=True)
    sys.exit(0 if code == vtkTesting.PASSED else 1)


def run_without_image(test):
    """A test with no baseline image, as CTest runs it: passed if it runs to the end
    (rtImageTest calls a test without an image failed). The interactor's Start does
    nothing, as rtImageTest makes it."""
    import traceback

    import vtkmodules.vtkRenderingUI  # noqa: F401 -- the interactor factory
    from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
    from vtkmodules.vtkTestingRendering import vtkTesting

    kind = type(vtkRenderWindowInteractor())

    class Interactor(kind):
        def Start(self):
            pass

    kind.override(Interactor)
    directory = os.path.abspath(os.path.dirname(test))
    sys.path.insert(0, directory)
    try:
        with open(test) as f:
            exec(compile(f.read(), test, "exec"), {"__name__": "__main__"})
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return vtkTesting.FAILED
    return vtkTesting.PASSED


if __name__ == "__main__":
    main()
