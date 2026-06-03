from rdkit import Chem

"""SMARTS strings for functional groups we care about."""
SMARTS_STRINGS : dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "amino": "[NX3H2]",
    "sulfonic_acid": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
    "guanidino": "[$([NX3][CX3](=[NX2])[NX3]),$([NX3][CX3]([NX3])=[NX2])]",
}

"""SMARTS queries for functional groups we care about."""
SMARTS_QUERIES = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SMARTS_STRINGS.items()
}

def get_labels(smiles_str: str) -> dict[str, bool] | None:
    """Get labels (amino, carboxylic_acid, guanidino, sulfonic_acid) of a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        return {
            name: mol.HasSubstructMatch(pattern)
            for name, pattern in SMARTS_QUERIES.items()
        }
    except Exception as _:
        return None
