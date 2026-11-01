### Fixed

- A `vtkCharArray` comes across as `u8` where the platform's `char` is unsigned (Linux on Arm) and as `i8` elsewhere; it was always read as `i8`, which turned values above 127 negative on Arm.
