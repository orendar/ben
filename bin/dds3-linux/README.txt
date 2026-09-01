Linux DDS Python extension (dds3), built from github.com/dds-bridge/dds v3.1.0.

    bin/dds3-linux/dds3/__init__.py   python/dds3/__init__.py from that tag
    bin/dds3-linux/dds3/_dds3.so      the built extension

The upstream recipe is `bazel build -c opt //python:_dds3`, but bazel fetches a
hermetic LLVM, JDK and emsdk -- ~15 GB for a target that is 41 translation
units. Compiling directly is equivalent and faster; these are the same flags
bazel's //:build_linux config passes (CPPVARIABLES.bzl plus .bazelrc's
-std=c++20), minus warning flags:

    clang++ -O3 -std=c++20 -fPIC -pthread -DNDEBUG         -I<dds>/library/src -I<inc> -I<dds>/python/src         -I$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')         -I$(python -c 'import pybind11;print(pybind11.get_include())')         -c <each .cpp under library/src, plus python/src/{bindings,converters}.cpp>
    clang++ -shared -pthread -O3 *.o -o _dds3.so

where <inc> holds a symlink `dds` -> <dds>/library/src, for `#include <dds/dds.hpp>`.

The extension is not abi3: build it against the same Python minor version the
server runs (3.12 here). Build with clang, not gcc -- gcc 13 is ~6% slower on
this code, and the 3.0-era builds measured slower still.
