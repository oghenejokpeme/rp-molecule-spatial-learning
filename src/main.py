import os
import copy
import numpy as np
import multiprocessing
import esig.tosig as ts
from functools import partial
from collections import namedtuple

# Global path to the molecules to analyse.
DBPATH = '../input/CASF_structures_subset/'

Atom = namedtuple('Atom', ['index', 'type', 'coordinates'])
Molecule = namedtuple('Molecule', 
                      ['id', 'atoms', 'mean_coordinate', 'adjacency_matrix'])

def create_adjacency_matrix(num_of_atoms, interactions):
    adjacency_matrix = np.zeros([num_of_atoms, num_of_atoms])
    for atom_a, atom_b in interactions:
        adjacency_matrix[atom_a, atom_b] = 1
        adjacency_matrix[atom_b, atom_a] = 1
    
    return adjacency_matrix

def process_molecule(db_filename):
    db_filepath = DBPATH + db_filename
    molecule_id = db_filename[:-4]
   
    atoms = []
    coordinates = []
    interactions = []
    atom_types = set()

    with open(db_filepath, 'r') as f:
        for line in f:
            nline = line.strip().split('\t')
            entity = nline[0]
            if entity == 'PRT' or entity == 'LIG':
                atom_index = int(nline[1])
                atom_type  = nline[2]
                atom_coordinates = np.asarray(nline[3:6], float)
                atom = Atom(atom_index, atom_type, atom_coordinates)
                
                atoms.append(atom)
                coordinates.append(atom_coordinates)

                if atom_type not in atom_types:
                    atom_types.add(atom_type)

            elif entity == 'INT':
                atom_pair = (int(nline[1]) - 1, int(nline[2]) - 1)
                interactions.append(atom_pair)
    
    mean_coordinate = np.array(coordinates).mean(axis = 0)
    adjacency_matrix = create_adjacency_matrix(len(atoms), interactions)
    molecule = Molecule(molecule_id, atoms, mean_coordinate, adjacency_matrix)

    return molecule, atom_types

def read_input_database():
    """Reads the input molecular database. Where the "database" is a folder
    containing individual files for each molecule one is interested in
    generating signatures for. The path to this folder is defined in DBPATH
    as a global variable.

    Returns: 
        Tuple, where the first argument is a list of all the molecules and the
        second argument is the index_map, which is a dictionary that holds all 
        unique atoms in the molecule database one is interested in analysing 
        and their respective indices after sorting.
    """

    db_filenames = os.listdir(DBPATH)
    db_filenames = [fname for fname in db_filenames if fname[-4:] == '.txt']

    molecules = []
    all_atom_types = set()
    for db_filename in db_filenames:
        molecule, atom_types = process_molecule(db_filename)
        molecules.append(molecule)
        all_atom_types = all_atom_types.union(atom_types)
    index_map = {atom: index for index, atom in 
                 enumerate(sorted(all_atom_types))}

    return (molecules, index_map)

def get_atom_paths(atom_index, adjacency_matrix, radius):
    """Get all unique paths for a given atom within a molecule.

    Args:
        atom_index: This is the atom index within the molecule - 1. 
            1 is subtracted because array indexes start at zero.
        adjacency_matrix: N by N adjacency matrix which indicates
            whether two atoms within a molecule are connected. N is
            the total number of atoms in a given molecule.
        radius: The distance from an atom within a molecule under consideration
            to explore.
    
    Returns:
        A list of valid paths as numpy arrays.    
    """

    paths = []
    path = np.zeros(radius + 1, int)
    path[0] = atom_index
    if radius == 0:
        paths.append(path)
    else:
        if radius == 1:
            temp = np.nonzero(adjacency_matrix[atom_index, :])
            for index in list(temp[0]):
                path[1] = index
                paths.append(copy.deepcopy(path))
        else:
            subpaths = get_atom_paths(atom_index, adjacency_matrix, radius - 1)
            for subpath in subpaths:
                temp = list(np.nonzero(adjacency_matrix[subpath[-1], :])[0])
                temp.remove(subpath[-2])
                for e in temp:
                    path[:-1] = subpath
                    path[-1]  = e
                    paths.append(copy.deepcopy(path))
    
    return paths

def categorical_path(molecule, atom_path, index_map, category_dimension, radius):
    """Creates categorical path matrix for a given atom path.
    
    Args:
        molecule: Namedtuple holding the details for a molecule.
        atom_path: A single atom path for a given atom within a molecule.
        index_map: Dictionary holds all unique atoms in the molecule database
            one is interested in analysing and their respective indices after
            sorting.
        category_dimension: The number of all unique atoms present in the
            molecule database currently being analysed.
        radius: The distance from an atom within a molecule under consideration
            to explore.        

    Returns:
        Path matrix for atom_path given the index_map and category dimension.
    """

    path_matrix = np.zeros([category_dimension, radius + 2])
    for i in range(radius + 1):
        atom_type = molecule.atoms[atom_path[i]].type

        if i == 0:
            path_matrix[index_map[atom_type], i + 1] = 1
        else:
            temp = np.zeros([category_dimension])
            temp[index_map[atom_type]] = 1
            path_matrix[:, i + 1] = path_matrix[:, i] + temp
    
    return path_matrix

def write_categorical_ps(molecule, molecule_atom_paths, index_map, radius, 
                         signature_degree, flag = True):
    """Writes the categorical path signatures for a given molecule to file.
    
    Args:
        molecule: Namedtuple holding the details for a molecule.
        molecule_atom_paths: List of lists holding the paths for each atom
            within a molecule.
        index_map: Dictionary hold all unique atoms in the molecule database
            one is interested in analysing and their respective indices after
            sorting.
        radius: The distance from an atom within a molecule under consideration
            to explore.        
        signature_degree: Integer signature degree to use in generating the 
            categorical path signatures.
        flag: Boolean indicating sigdim and stream2sig should be used or
            logsigdim and steam2logsig.
    
    Returns:
        None, but writes signatures to ../output/moleculeID/
    """
    # Instead of signal_dimension as in write_categorical_ps()
    category_dimension = len(index_map)
    signature_dimension = (ts.sigdim(category_dimension, signature_degree) if flag
                           else ts.logsigdim(category_dimension, signature_degree))
    signature_function  = ts.stream2sig if flag else ts.stream2logsig

    file_path = ('../output/' + molecule.id + '/categorical_sig.txt' if flag else
                 '../output/' + molecule.id + '/categorical_logsig.txt')

    with open(file_path, 'w') as f:
        for atom_index, atom_paths in enumerate(molecule_atom_paths, 1):
            for path_index, atom_path in enumerate(atom_paths, 1):
                path_matrix = categorical_path(molecule, atom_path, 
                                               index_map, category_dimension, 
                                               radius)
                sig = signature_function(np.transpose(path_matrix), signature_degree)
                rsig = [str(round(val, 2)) for val in sig]
                atom_path_index = str(atom_index) + '_' + str(path_index)
                line = atom_path_index + ' ' + ' '.join(rsig) + '\n'
                f.write(line)

def write_coordinate_ps(molecule, molecule_atom_paths, signature_degree, 
                        signal_dimension = 3, flag = True):
    """Writes the coordinate path signatures for a given molecule to file.
    
    Args:
        molecule: Namedtuple holding the details for a molecule.
        molecule_atom_paths: List of lists holding the paths for each atom
            within a molecule.
        signature_degree: Integer signature degree to use in generating the 
            coordinate path signatures.
        signal_dimension: Integer signal dimension to be used in esig's 
            sigdim or logsigdim.
        flag: Boolean indicating sigdim and stream2sig should be used or
            logsigdim and steam2logsig.
    
    Returns:
        None, but writes signatures to ../output/moleculeID/
    """

    signature_dimension = (ts.sigdim(signal_dimension, signature_degree) if flag
                           else ts.logsigdim(signal_dimension, signature_degree))
    signature_function  = ts.stream2sig if flag else ts.stream2logsig

    file_path = ('../output/' + molecule.id + '/coordinate_sig.txt' if flag else
                 '../output/' + molecule.id + '/coordinate_logsig.txt')

    with open(file_path, 'w') as f:
        for atom_index, atom_paths in enumerate(molecule_atom_paths, 1):
            for path_index, atom_path in enumerate(atom_paths, 1):
                coord_matrix = np.zeros([len(atom_path), 3])
                for j in range(len(atom_path)):
                    coord_matrix[j, :] = molecule.atoms[atom_path[j]].coordinates
                sig = signature_function(np.transpose(coord_matrix),
                                         signature_degree)
                rsig = [str(val) for val in sig]
                atom_path_index = str(atom_index) + '_' + str(path_index)
                line = atom_path_index + ' ' + ' '.join(rsig) + '\n'
                f.write(line)

def write_path_signatures(molecule, index_map, sigconfig, signature_degree = 2, 
                          radius = 4):
    """Generates the paths for all atoms in molecule. This is then passed to
    the signature generating function based on sigconfig.

    Args:
        molecule: Namedtuple holding the details for a molecule.
        index_map: Dictionary holds all unique atoms in the molecule database
            one is interested in analysing and their respective indices after
            sorting.
        sigconfig: Tuple pair (A, B). A is one of either categorical or
            coordinate, and B can be either True or False. If True the 
            standard sigdim function in esig is used, if False, logsigdim
            is used.
        signature_degree: Integer signature degree to use in generating the 
            categorical and coordinate path signatures.
        radius: The distance from an atom within a molecule under consideration
            to explore.
    
    Returns:
        None
    """
    # Comment out the line below if one is not  interested in seeing what
    # molecule is currently being processed.
    print(molecule.id)

    # Create output folder for molecule if it does not exist.
    opath = '../output/' + molecule.id + '/'
    if not os.path.isdir(opath):
        os.mkdir(opath) 

    molecule_atom_paths = [get_atom_paths(atom.index - 1, 
                                          molecule.adjacency_matrix, 
                                          radius)
                           for atom in molecule.atoms]

    sigtype, flag = sigconfig
    if sigtype == 'coordinate':
        write_coordinate_ps(molecule, molecule_atom_paths, signature_degree, 
                            flag = flag)
    
    elif sigtype == 'categorical':
        write_categorical_ps(molecule, molecule_atom_paths, index_map, radius, 
                             signature_degree, flag = flag)

 
def main(parallel = True, sigconfig = ('coordinate', True)):
    """Reads the molecules from the input database paths and generates
    the signature for each molecule based on the specified configuration.
    
    Args:
        parallel: Boolean indicates whether signature generation should
            be done in parallel.

        sigconfig: Tuple pair (A, B). A is one of either categorical or
            coordinate, and B can be either True or False. If True the 
            standard sigdim function in esig is used, if False, logsigdim
            is used.
    
    Returns:
        None
    """

    molecules, index_map = read_input_database()
    
    if parallel:
        print('Parallel execution')
        partial_wps = partial(write_path_signatures, index_map = index_map, 
                              sigconfig = sigconfig)
        with multiprocessing.Pool() as pool:
            pool.map(partial_wps, molecules)
    else:
        print('Sequential execution')
        for molecule in molecules:
            write_path_signatures(molecule, index_map, sigconfig)

if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    duration = round((time.time() - start_time)/60, 2)
    print(f'Duration {duration} minutes')