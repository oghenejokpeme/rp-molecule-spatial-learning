## Rough Paths Signature Generator for Molecules

The program is located in `src` and assumes that the `output` folder exists. 
This is where all the path signatures for each molecule will be stored. The molecule database input path can be specified using the `DBPATH` variable in `src/main.py`. The program assumes that this input folder contains individual files for each molecule one is interested in generating path signatures for. Therefore, it assumes that there is a pool of molecules.

When run, the signature paths, based on the specified configuration will be placed in `output/molecule_id`, where `molecule_id` is the molecule filename supplied in `DBPATH`.

Path signature generation can be done sequentially or in parallel. This is done by passing `True` or `False` to `main()`.  Parallel execution is used if `True`. Path signature generation is done in parallel by default.

Once all this is setup, running the program requires executing `main.py` from the terminal or within a notebook.
