with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "cuampcor-hip";
  src = lib.cleanSource ./.;
  nativeBuildInputs = [
    cmake
  ];
  buildInputs = [
    gdal
    hip-cpu
    hip-headers
    tbb
    fftwFloat
    fftw
  ];
  enableParallelBuilding = true;
}
