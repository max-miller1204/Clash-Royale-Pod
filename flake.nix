{
  description = "Clash Royale Post-Game Analyzer — CV + statistical modeling pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;

        systemDeps = with pkgs; [
          ffmpeg-full
          tesseract
          zstd
          stdenv.cc.cc.lib
          zlib
          libGL
          glib
        ];

        devTools = with pkgs; [
          uv
          ruff
          git
          gnumake
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python ] ++ devTools ++ systemDeps;

          env = {
            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
            PIP_DISABLE_PIP_VERSION_CHECK = "1";
            TESSDATA_PREFIX = "${pkgs.tesseract}/share/tessdata";
          };

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemDeps}:$LD_LIBRARY_PATH"
            if [ ! -d .venv ]; then
              echo "Creating venv with uv..."
              uv sync
            fi
            echo "crpod dev shell ready — run 'uv run crpod --help'"
          '';
        };

        packages.default = pkgs.stdenvNoCC.mkDerivation {
          pname = "clash-royale-pod";
          version = "0.1.0";
          src = ./.;
          dontBuild = true;
          installPhase = ''
            mkdir -p "$out/share/crpod"
            cp -R src pyproject.toml README.md "$out/share/crpod/"
          '';
        };
      });
}
