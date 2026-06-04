{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, utils, ... }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        lib = pkgs.lib;

        jupyterlab-vim =
          pp:
          with pp;
          buildPythonPackage rec {
            pname = "jupyterlab-vim";
            version = "4.1.4";
            format = "wheel";

            src = fetchPypi {
              inherit version format;
              pname = "jupyterlab_vim";
              python = "py3";
              dist = "py3";
              hash = "sha256-E8Kf8f04X93zPrhaixZEG+acYgMocLuXowVLLPAKsTU=";
            };

            propagatedBuildInputs = [
              jupyterlab
            ];
          };

        jcamp = pkgs.python3Packages.callPackage (
          {
            buildPythonPackage,
            fetchFromGitHub,
            flit-core,
            numpy,
          }:
          buildPythonPackage {
            pname = "jcamp";
            version = "80a0adb";
            format = "pyproject";

            src = fetchFromGitHub {
              owner = "nzhagen";
              repo = "jcamp";
              rev = "80a0adb84b6f22ff14ec1ce0bfd45c15c03f44d9";
              hash = "sha256-6NOce/7m1Zyn8xOC92MG5pLr4JBnizNzlRZG9sVLV5Q=";
            };

            nativeBuildInputs = [ flit-core ];
            dependencies = [ numpy ];
          }
        ) { };

        python = pkgs.python3.withPackages (
          pp: with pp; [
            jupyter
            jupyterlab
            (jupyterlab-vim pp)
            ipython
            tqdm
            ipympl
            ipywidgets

            numpy
            pandas
            pandas-stubs
            dask
            matplotlib
            seaborn
            pyarrow

            jcamp
            openbabel
            rdkit

            scikit-learn
            scikit-image
            statsmodels

            torchWithCuda
            torchvision
            torchmetrics
            lightning
            jsonargparse
            tensorboard

            datasets
            evaluate
            transformers
            diffusers
            accelerate
            hf-xet
          ] ++ jsonargparse.optional-dependencies.signatures
        );

        packages = [
          python
        ];

      in
      {
        packages.jcamp = jcamp;
        devShells.default = pkgs.mkShell {
          inherit packages;

          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
          '';
        };
      }
    );
}
