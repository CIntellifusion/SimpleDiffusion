{
  nixConfig = {
    extra-substituters = [ "https://cache.garnix.io" ];
    extra-trusted-public-keys =
      [ "cache.garnix.io:CTFPyKSLcx5RMJKfLo5EEPUObbA78b0YQ2DTCJXqr9g=" ];
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgsGcc.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgsDrv.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, nixpkgsGcc, nixpkgsDrv, flake-utils }:
    let
      system = "x86_64-linux";
      pkgsDrv = import nixpkgsDrv {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      pkgsGcc = import nixpkgsGcc { inherit system; };
      driverOverlay = final: prev: {
        cudaDrivers = pkgsDrv.linuxPackages.nvidia_x11.overrideAttrs (old:
          let version = "550.78";
          in {
            src = pkgsDrv.fetchurl {
              url =
                "https://download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
              sha256 = "sha256-NAcENFJ+ydV1SD5/EcoHjkZ+c/be/FQ2bs+9z+Sjv3M=";
            };
          });
      };
      gccOverlay = final: prev: { portableGcc = pkgsGcc.gcc; };
      overlays = [ driverOverlay gccOverlay ];
      pkgs = import nixpkgs {
        inherit system overlays;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in { devShells.${system}.default = import ./nix/shell.nix { inherit pkgs; }; };
}
