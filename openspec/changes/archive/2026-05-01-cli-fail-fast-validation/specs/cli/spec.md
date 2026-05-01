## ADDED Requirements

### Requirement: Argument validation precedes heavy work

The crpod CLI SHALL validate every argument value (file paths, output
directories, numeric ranges) before any module that performs filesystem I/O for
ML weights, video decoding, or HuggingFace dataset access is imported or
invoked. This applies to all subcommands that accept such arguments.

#### Scenario: Mistyped weights path on analyze

- **GIVEN** the user runs the `analyze` subcommand with a `--weights` value that does not exist on disk
- **WHEN** the CLI parses the arguments
- **THEN** the CLI MUST exit with a non-zero status before any pipeline, dataset, or YOLO module is loaded
- **AND** the error MUST be emitted to stderr and identify the missing weights path

#### Scenario: Mistyped optional model path on analyze

- **GIVEN** the user runs the `analyze` subcommand with an existing `--weights` and a `--model` value that does not exist on disk
- **WHEN** the CLI parses the arguments
- **THEN** the CLI MUST exit with a non-zero status before any pipeline or dataset module is loaded
- **AND** the error MUST identify the missing model path

#### Scenario: Mistyped weights path on train

- **GIVEN** the user runs the `train` subcommand with a `--weights` value that does not exist on disk
- **WHEN** the CLI parses the arguments
- **THEN** the CLI MUST exit with a non-zero status before any HuggingFace dataset access occurs
- **AND** the error MUST identify the missing weights path

### Requirement: Output directories are validated before heavy work

The crpod CLI SHALL ensure that any user-supplied output destination is
reachable (the parent directory exists or can be created) before producing
artifacts that take more than a few seconds to compute.

#### Scenario: Train output points into a missing parent directory

- **GIVEN** the user runs the `train` subcommand with `--out` whose parent directory does not exist
- **WHEN** the CLI parses the arguments
- **THEN** the CLI MUST exit with a non-zero status before any dataset loading occurs
- **AND** the error MUST identify the unreachable output path

### Requirement: Numeric arguments enforce positive-value preconditions

The crpod CLI SHALL reject non-positive values for arguments that bound
iteration or sampling (frames per second, replay counts) before any heavy work
begins.

#### Scenario: Non-positive max-replays on train

- **GIVEN** the user runs the `train` subcommand with `--max-replays 0` (or a negative integer)
- **WHEN** the CLI parses the arguments
- **THEN** the CLI MUST exit with a non-zero status before any dataset loading occurs
- **AND** the error MUST state that `--max-replays` must be greater than zero

### Requirement: CLI errors share a consistent surface

The crpod CLI SHALL emit user-facing argument errors to stderr using a stable
prefix and exit code so that wrapper scripts can rely on a uniform error
surface.

#### Scenario: Argument error format

- **GIVEN** any CLI invocation that fails argument validation
- **WHEN** the CLI emits its error
- **THEN** the message MUST be written to stderr
- **AND** the message MUST begin with the literal prefix `error: `
- **AND** the process MUST exit with status code `1`
