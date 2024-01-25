{ pkgs ? import <nixpkgs> {}
, lib ? pkgs.lib
}:

pkgs.stdenv.mkDerivation rec {
  name = "okfm";

  buildInputs = with pkgs; [
    rust-analyzer
    cargo
    wayland
    libxkbcommon
    libGL
  ];

  LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
}
