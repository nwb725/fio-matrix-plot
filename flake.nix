{
  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.11"; };
  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pythonPackages = pkgs.python312Packages;
    in rec {
      packages.x86_64-linux.default = pythonPackages.buildPythonApplication {
        pname = "plot";
        version = "0.1.0";
        pyproject = false;
        dontUnpack = true;
        propagatedBuildInputs = with pythonPackages; [
          numpy
          scipy
          matplotlib
          pandas
          binary
        ];

        installPhase = ''
          install -Dm755 "${./plot.py}" $out/bin/fio-matrix-plot
        '';

      };

      devShells.x86_64-linux.default = pkgs.mkShell {
        inputsFrom = [ packages.x86_64-linux.default ];
        packages = [ pythonPackages.coverage pkgs.feh ];
      };
    };
}
