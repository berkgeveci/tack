### Changed

- Every source file opens with an SPDX copyright and license header, as in
  ParaView and VTK (`tools/spdx_headers.py`, checked in CI's lint job).
  `tack.dataset`'s clip case tables now credit their authors: they are VisIt's
  (Lawrence Livermore National Security, BSD-3-Clause), which VTK keeps in
  `vtkTableBasedClipCases.h` and Viskores copied, not Viskores' own.
