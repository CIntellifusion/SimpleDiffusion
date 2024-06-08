{ pkgs, ... }:

(pkgs.mkShell rec {

  buildInputs =
    (with pkgs; [ python3 poetry gcc gnumake libGL zlib glib cudatoolkit ]);

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath
    ([ pkgs.cudaDrivers pkgs.stdenv.cc.cc.lib ] ++ buildInputs);

  shellHook = ''
    poetry env use ${pkgs.python3}/bin/python
    poetry install
  '';
})
