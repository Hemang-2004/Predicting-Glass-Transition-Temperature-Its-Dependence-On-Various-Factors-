import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D

# --- 1. Define Calculation Functions ---

def calculate_polymer_properties(smiles_string):
    """
    Calculates Radius of Gyration, Molar Refractivity, and Branching Points
    for a given polymer SMILES string.
    """
    if not isinstance(smiles_string, str):
        return np.nan, np.nan, np.nan

    try:
        clean_smiles = smiles_string.replace('*', '')
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return np.nan, np.nan, np.nan

        molar_refractivity = Descriptors.MolMR(mol)
        branching_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
        
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=42) == -1:
            radius_of_gyration = np.nan
        else:
            radius_of_gyration = Descriptors3D.RadiusOfGyration(mol_h)
            
        return radius_of_gyration, molar_refractivity, branching_points

    except Exception:
        return np.nan, np.nan, np.nan

def calculate_solvent_properties(smiles_string):
    """
    Calculates MolWt, MolMR, TPSA, H-Donors, and H-Acceptors for a solvent SMILES.
    """
    if not isinstance(smiles_string, str):
        return [np.nan] * 5

    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return [np.nan] * 5

        mol_wt = Descriptors.MolWt(mol)
        mol_mr = Descriptors.MolMR(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        return [mol_wt, mol_mr, tpsa, h_donors, h_acceptors]
    except Exception:
        return [np.nan] * 5

# --- 2. Load Data and Define Solvent Properties ---
try:
    df = pd.read_csv("original.csv")

    dielectric_constants_map = {
        '1,1,1,2-tetrafluoroethane': 9.6, '1,1-difluoroethane': 10.9, '1,4-dioxane': 2.2,
        '1-butanol': 17.5, '1-chloro-1,1-difluoroethane': 7.4, '1-hexanol': 13.3,
        '1-hexene': 2.1, '1-pentanol': 15.1, '1-propanol': 20.3, '2-butanol': 17.9,
        '2-methylpropane': 1.8, '2-propanol': 19.9, 'r22': 6.1, 'acetone': 21.0,
        'argon': 1.5, 'benzene': 2.3, 'butyl acetate': 5.0, 'carbon dioxide': 1.6,
        'carbon monoxide': 1.6, 'carbon tetrachloride': 2.2, 'chlorine': 2.0,
        'chloroform': 4.8, 'cyclohexane': 2.0, 'cyclopentane': 2.0, 'di-n-propyl ether': 3.4,
        'dichloromethane': 9.1, 'diethyl ether': 4.3, 'diethyl ketone': 17.0,
        'dimethyl carbonate': 3.1, 'dimethyl methyl phosphonate': 20.6, 'dinitrogen monoxide': 1.6,
        'ethane': 1.9, 'ethanol': 24.5, 'ethene': 1.9, 'ethyl acetate': 6.0,
        'ethylbenzene': 2.4, 'helium': 1.05, 'hexafluoroethane': 2.0, 'hydrogen': 1.2,
        'hydrogen sulfide': 9.3, 'isobutanol': 17.9, 'isooctane, 2,2,4-trimethylpentane': 1.9,
        'isopentane': 1.8, 'krypton': 1.8, 'm-xylene': 2.4, 'methane': 1.7,
        'methanol': 33.0, 'methylacetate': 6.7, 'methylethylketone': 18.5,
        'n-butane': 1.8, 'n-heptane': 1.9, 'n-hexane': 1.9, 'n-pentane': 1.8,
        'neon': 1.1, 'neopentane': 1.8, 'nitrobenzene': 34.8, 'nitrogen': 1.4,
        'nitromethane': 35.9, 'nonane': 2.0, 'octafluoropropane': 2.0, 'octane': 2.0,
        'oxygen': 1.5, 'p-xylene': 2.3, 'propane': 1.6, 'propene': 2.1,
        'propylacetate': 5.6, 'sulfur hexafluoride': 1.8, 'sulphur dioxide': 15.6,
        'tert-butyl alcohol': 12.5, 'tetrafluoromethane': 1.6, 'tetrahydrofuran': 7.5,
        'toluene': 2.4, 'water': 80.1, 'xenon': 2.0
    }

    # --- 3. Apply Calculations and Create New Columns ---
    print("Calculating polymer and solvent properties... This may take a moment.")

    df['dielectric_constant'] = df['solvent_name'].str.lower().map(dielectric_constants_map)

    poly_props = df['polymer_smiles'].apply(lambda s: pd.Series(calculate_polymer_properties(s)))
    df[['radius_of_gyration', 'molar_refractivity', 'branching']] = poly_props

    solv_props = df['solvent_smiles'].apply(lambda s: pd.Series(calculate_solvent_properties(s)))
    df[['solvent_mol_wt', 'solvent_mol_mr', 'solvent_tpsa', 'solvent_h_donors', 'solvent_h_acceptors']] = solv_props

    # --- 4. Finalize and Save the DataFrame ---

    final_column_order = [
        'polymer_name', 'polymer_smiles', 'solvent_formula', 'solvent_name',
        'solvent_smiles', 'mn', 'mw', 'tg', 'dens', 'temperature',
        'dielectric_constant', 'molar_refractivity', 'radius_of_gyration',
        'branching', 'solvent_mol_wt', 'solvent_mol_mr', 'solvent_tpsa', 
        'solvent_h_donors', 'solvent_h_acceptors'
    ]

    final_df = df[final_column_order]
    final_df.to_csv('ok.csv', index=False)
    
    print("\n✅ Processing complete!")
    print("The new file 'final_augmented_dataset.csv' has been created with all properties.")
    print("\nPreview of the final data:")
    print(final_df.head())

except FileNotFoundError:
    print("Error: 'original.csv' not found. Please ensure it is in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")