import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, GraphDescriptors
import numpy as np

# --- 1. Define Functions for Calculation ---

def calculate_descriptors(smiles_string):
    """
    Calculates branching index (BalabanJ) and molar refractivity (MolMR)
    for a given SMILES string. The '*' characters for polymer repeating
    units are removed before calculation.
    """
    if not isinstance(smiles_string, str):
        return None, None

    # RDKit cannot process '*', so we remove them for the calculation.
    # This treats the repeating unit as a standalone molecule.
    clean_smiles = smiles_string.replace('*', '')
    
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            # Handle cases where SMILES parsing fails
            return None, None
            
        # Calculate Balaban J index as a branching descriptor
        branching_index = GraphDescriptors.BalabanJ(mol)
        
        # Calculate Molar Refractivity as a proxy for excluded volume
        molar_refractivity = Crippen.MolMR(mol)
        
        return branching_index, molar_refractivity
        
    except Exception:
        # Return None if any error occurs during calculation
        return None, None

# --- 2. Load Data and Define Solvent Properties ---

# Load your dataset
df = pd.read_csv("merged_polymer_dataset_no_duplicates.csv")

# Dictionary of solvent names and their approximate dielectric constants at room temp
dielectric_constants_map = {
    '1,1,1,2-tetrafluoroethane': 9.6,
    '1,1-difluoroethane': 10.9,
    '1,4-dioxane': 2.2,
    '1-butanol': 17.5,
    '1-chloro-1,1-difluoroethane': 7.4,
    '1-hexanol': 13.3,
    '1-hexene': 2.1,
    '1-pentanol': 15.1,
    '1-propanol': 20.3,
    '2-butanol': 17.9,
    '2-methylpropane': 1.8, # Isobutane
    '2-propanol': 19.9,
    'r22': 6.1, # Chlorodifluoromethane
    'acetone': 21.0,
    'argon': 1.5,
    'benzene': 2.3,
    'butyl acetate': 5.0,
    'carbon dioxide': 1.6,
    'carbon monoxide': 1.6,
    'carbon tetrachloride': 2.2,
    'chlorine': 2.0,
    'chloroform': 4.8,
    'cyclohexane': 2.0,
    'cyclopentane': 2.0,
    'di-n-propyl ether': 3.4,
    'dichloromethane': 9.1,
    'diethyl ether': 4.3,
    'diethyl ketone': 17.0,
    'dimethyl carbonate': 3.1,
    'dimethyl methyl phosphonate': 20.6,
    'dinitrogen monoxide': 1.6,
    'ethane': 1.9,
    'ethanol': 24.5,
    'ethene': 1.9,
    'ethyl acetate': 6.0,
    'ethylbenzene': 2.4,
    'helium': 1.05,
    'hexafluoroethane': 2.0,
    'hydrogen': 1.2,
    'hydrogen sulfide': 9.3,
    'isobutanol': 17.9,
    'isooctane, 2,2,4-trimethylpentane': 1.9,
    'isopentane': 1.8,
    'krypton': 1.8,
    'm-xylene': 2.4,
    'methane': 1.7,
    'methanol': 33.0,
    'methylacetate': 6.7,
    'methylethylketone': 18.5,
    'n-butane': 1.8,
    'n-heptane': 1.9,
    'n-hexane': 1.9,
    'n-pentane': 1.8,
    'neon': 1.1,
    'neopentane': 1.8,
    'nitrobenzene': 34.8,
    'nitrogen': 1.4,
    'nitromethane': 35.9,
    'nonane': 2.0,
    'octafluoropropane': 2.0,
    'octane': 2.0,
    'oxygen': 1.5,
    'p-xylene': 2.3,
    'propane': 1.6,
    'propene': 2.1,
    'propylacetate': 5.6,
    'sulfur hexafluoride': 1.8,
    'sulphur dioxide': 15.6,
    'tert-butyl alcohol': 12.5,
    'tetrafluoromethane': 1.6,
    'tetrahydrofuran': 7.5,
    'toluene': 2.4,
    'water': 80.1,
    'xenon': 2.0
}

# --- 3. Apply Calculations and Create New Columns ---

# Add Dielectric Constant column by mapping the solvent name
# .str.lower() makes the matching case-insensitive
df['dielectric_constant'] = df['solvent_name'].str.lower().map(dielectric_constants_map)

# Apply the descriptor calculation function to the 'polymer_smiles' column
# This creates two new series (lists) with the results
results = df['polymer_smiles'].apply(calculate_descriptors)

# Unpack the results into two new columns in the DataFrame
df['branching_index'] = results.apply(lambda x: x[0] if x is not None else np.nan)
df['molar_refractivity'] = results.apply(lambda x: x[1] if x is not None else np.nan)

# --- 4. Save the Updated DataFrame ---

# Save the final dataframe to a new CSV file
df.to_csv('polymers_with_descriptors.csv', index=False)

print("Processing complete!")
print("The new file 'polymers_with_descriptors.csv' has been created.")
print("\nPreview of the first 5 rows of the new data:")
print(df.head())