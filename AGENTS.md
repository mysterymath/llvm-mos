# Repository Guidelines

## Project Structure & Module Organization
- `llvm/` holds the forked LLVM core; MOS backend work lives in `llvm/lib/Target/MOS` with tests under `llvm/test`.
- `clang/`, `lld/`, `mlir/`, and other siblings mirror upstream projects; each keeps its own `test/` tree.
- Use an out-of-tree `build/` directory; helper scripts and developer tools sit under `utils/` and `local-bin/`.

## Build, Test, and Development Commands
- Configure once with `cmake -C clang/cmake/caches/MOS.cmake -S llvm -B build -G Ninja`.
- Incremental rebuilds: `cmake --build build --target <tool>` (leave `<tool>` empty for a full build).
- Regression suites: `ninja -C build check-all`; focused runs such as `ninja -C build check-llvm-codegen-mos` or `ninja -C build check-clang` keep iterations fast.
- Reproduce a single test with `build/bin/llvm-lit <path/to/test>`.

## Coding Style & Naming Conventions
- Observe LLVM coding standards: two-space indentation, UpperCamelCase types, lowerCamelCase functions, snake_case locals.
- Run `clang-format -style=file <files>` before review; supplement with `clang-tidy` using the repository `.clang-tidy` rules.
- Keep CMake target names lowercase with hyphens and mirror upstream directory layout for new MOS code.

## Testing Guidelines
- Place new IR, MIR, or frontend tests beside similar cases (`llvm/test/CodeGen/MOS`, `clang/test/CodeGen/mos`).
- Name files with concise, hyphenated descriptions and add `// RUN:` lines that execute the intended checks.
- Expect contributors to run `ninja -C build check-all` before submitting and to mention any skipped suites in the review.
- When debugging, use `llvm-lit -vv` to capture command lines and attach reduced repros to issues.

## Commit & Change Submission
- Commit summaries stay short and imperative (`Handle MOS copy folding`); elaboration belongs in the message body.
- Manage history with Jujutsu: `jj status`, amend via `jj describe`, rebase on `main` using `jj rebase -d main`, then publish through `jj git push --branch <topic>`.
- Pull requests should link to the motivating issue, outline functional impact, and mention benchmarks or screenshots when behavior is user visible.

## Jujutsu Workflow Tips
- Keep topic branches focused (`jj branch create mos-fix-foo`) and drop experiments with `jj abandon`.
- Review your patch stack via `jj log -r @` and squash fixups before requesting review.
