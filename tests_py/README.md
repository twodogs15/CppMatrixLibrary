# On Debugging EMatrix Testing

There does not seem to be an extensive amount information about debugging
mixed language, Python / C++, programming.  The most useful tip is in the 
following paper:

[1]  Enkovaara, J., et al. "GPAW - massively parallel electronic structure calculations with Python-based software." International Conference on Computational Science. 2011. http://dx.doi.org/10.1016/j.procs.2011.04.003.

The procedure is as follows:

```bash
# Make sure the python test file and the compiled test module, code.cpython-39-x86_64-linux-gnu.so, are in the same directory.
$ gdb -args python3.9 -m pdb test_EMatrix.py
(gdb) b test_EMatrix.cpp:19 # Answer y to "Make breakpoint pending on future shared library load? (y or [n])"
(gdb) r # Run the program which invokes the Python pdb module.
(Pdb) c # Continue running pdb.  The program will stop in gdb within the C++!!
```