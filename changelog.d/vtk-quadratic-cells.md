### Fixed

- `vtk_to_dataset` refuses VTK cells it reads neither as shapes nor as
  higher-order spaces (biquadratic and triquadratic cells, quadratic
  pyramids), naming them by VTK's class names; they came across as cells no
  algorithm took, every one failing with "are not linear shapes".
