### Fixed

- Examples 50-53 on Level Zero make their device in VTK's Level Zero context
  (`init_level_zero()`) where VTK has one, else Tack's own: under a VTK that
  holds Level Zero memory, their outputs went back to VTK from Tack's own
  context and were refused. The examples test skips 42 and 43, which need
  that context, under a VTK without one, as it skips a missing optional module.
