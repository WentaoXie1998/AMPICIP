# 输入一个药物 SMILES -> 输出该药物的原子图 PyG Data
# 节点特征：55维（explicit_H=True, use_chirality=True）
# 边特征：16维（bond type + conjugation + ring + stereo）


import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def get_atom_valence(atom):
    if hasattr(atom, "GetTotalValence"):
        try:
            return int(atom.GetTotalValence())
        except Exception:
            pass
    ev = int(atom.GetExplicitValence()) if hasattr(atom, "GetExplicitValence") else 0
    iv = int(atom.GetImplicitValence()) if hasattr(atom, "GetImplicitValence") else 0
    return ev + iv



def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [float(x == s) for s in allowable_set]


def atom_features(atom, explicit_H=True, use_chirality=True):
    atom_symbol_set = [
        'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
        'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In',
        'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
    ]  # 44

    hyb_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]  # 5

    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), atom_symbol_set)
        + [
            atom.GetDegree() / 10.0,
            float(get_atom_valence(atom)),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons())
        ]
        + one_of_k_encoding_unk(atom.GetHybridization(), hyb_set)
        + [float(atom.GetIsAromatic())]
    )

    if explicit_H:
        results = results + [float(atom.GetTotalNumHs())]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']
            ) + [float(atom.HasProp('_ChiralityPossible'))]
        except Exception:
            results = results + [0.0, 0.0] + [float(atom.HasProp('_ChiralityPossible'))]

    return torch.tensor(np.array(results, dtype=np.float32))


def bond_features(bond):
    bt = bond.GetBondType()

    bond_type = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bt == Chem.rdchem.BondType.IONIC,
    ]

    conj = one_of_k_encoding_unk(bond.GetIsConjugated(), [0, 1, "nonbond"])
    ring = one_of_k_encoding_unk(bond.IsInRing(), [0, 1, "nonbond"])

    stereo = one_of_k_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "nonbond"]
    )

    feats = bond_type + conj + ring + stereo
    return torch.tensor(np.array(feats, dtype=np.float32))


# ----------------------------
# 单分子 -> 原子图 Data
# ----------------------------
def build_atom_graph_from_smiles(smiles: str,
                                 explicit_H: bool = True,
                                 use_chirality: bool = True,
                                 add_dummy_self_loop_if_no_bond: bool = True) -> Data:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass

    x_list = [
        atom_features(a, explicit_H=explicit_H, use_chirality=use_chirality)
        for a in mol.GetAtoms()
    ]
    x = torch.stack(x_list, dim=0) if len(x_list) else torch.empty((0, 55), dtype=torch.float32)

    edge_src, edge_dst, edge_attr_list = [], [], []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bf = bond_features(bond)  # [16]

        edge_src += [a1, a2]
        edge_dst += [a2, a1]
        edge_attr_list += [bf, bf]

    if len(edge_src) == 0:
        if add_dummy_self_loop_if_no_bond and mol.GetNumAtoms() > 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 16), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 16), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list, dim=0).to(torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol_smiles=smiles,
        num_atoms=mol.GetNumAtoms()
    )
    return data


if __name__ == "__main__":
    test_smiles = "CC[C@@]1(C[C@@H]2C3=CC(=C(C=C3CCN2C[C@H]1CC(C)C)OC)OC)O"
    data = build_atom_graph_from_smiles(test_smiles)

    print("SMILES:", data.mol_smiles)
    print("num_atoms:", data.num_atoms)
    print("x shape:", data.x.shape)           # [N, 55]
    print("edge_index shape:", data.edge_index.shape)
    print("edge_attr shape:", data.edge_attr.shape)  # [E, 16]
