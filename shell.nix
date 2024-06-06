{ pkgs, ... }:

(pkgs.mkShell rec {

  buildInputs = (with pkgs; [ python39 poetry gcc gnumake zsh zlib libGL glib]);

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc];

  shellHook = ''
  export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
  poetry env use ${pkgs.python39}/bin/python
  poetry install
  '';
})