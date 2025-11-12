import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from typing import Union, List, Tuple


class FingerprintExtractor:
    """
    分子指纹提取器，支持多种指纹类型和分子描述符
    """

    def __init__(self, fingerprint_type: str = 'morgan', n_bits: int = 1024):
        """
        初始化指纹提取器

        Args:
            fingerprint_type: 指纹类型 ('morgan', 'rdkit')
            n_bits: 指纹位数
        """
        self.fingerprint_type = fingerprint_type
        self.n_bits = n_bits

    def smiles_to_mol(self, smiles: str) -> Union[Chem.Mol, None]:
        """
        将SMILES字符串转换为RDKit分子对象

        Args:
            smiles: SMILES字符串

        Returns:
            RDKit分子对象或None（如果转换失败）
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def extract_fingerprint(self, smiles: str) -> Union[np.ndarray, None]:
        """
        从SMILES字符串提取指纹

        Args:
            smiles: SMILES字符串

        Returns:
            指纹向量或None（如果转换失败）
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        try:
            if self.fingerprint_type == 'morgan':
                # Morgan指纹 (ECFP类似)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=self.n_bits
                )
            elif self.fingerprint_type == 'rdkit':
                # RDKit指纹
                fingerprint = Chem.RDKFingerprint(
                    mol, fpSize=self.n_bits
                )
            else:
                raise ValueError(f"不支持的指纹类型: {self.fingerprint_type}")

            # 转换为numpy数组
            return np.array(fingerprint)
        except:
            return None

    def extract_descriptors(self, smiles: str) -> Union[np.ndarray, None]:
        """
        从SMILES字符串提取分子描述符

        Args:
            smiles: SMILES字符串

        Returns:
            描述符向量或None（如果转换失败）
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        try:
            # 基本分子描述符
            descriptors = [
                Descriptors.MolWt(mol),           # 分子量
                Descriptors.MolLogP(mol),         # LogP (疏水性)
                Descriptors.TPSA(mol),            # 极性表面积
                Descriptors.NumHDonors(mol),      # 氢键供体数
                Descriptors.NumHAcceptors(mol),   # 氢键受体数
                Descriptors.NumRotatableBonds(mol),  # 可旋转键数
                Descriptors.NumAromaticRings(mol),   # 芳香环数
                Descriptors.NumAliphaticRings(mol),  # 脂肪环数
                Descriptors.NHOHCount(mol),       # NH和OH基团数
                Descriptors.NOCount(mol),         # 氮和氧原子数
                Descriptors.RingCount(mol),       # 环总数
                QED.qed(mol),                     # 药物相似性指数
                Crippen.MolMR(mol),               # 摩尔折射率
            ]
            
            # Lipinski规则相关描述符
            descriptors.extend([
                Lipinski.NumHDonors(mol),         # 氢键供体数 (Lipinski)
                Lipinski.NumHAcceptors(mol),      # 氢键受体数 (Lipinski)
                Lipinski.NumRotatableBonds(mol),  # 可旋转键数 (Lipinski)
                Lipinski.NumAliphaticCarbocycles(mol),  # 脂肪族碳环数
                Lipinski.NumAliphaticHeterocycles(mol), # 脂肪族杂环数
                Lipinski.NumAliphaticRings(mol),        # 脂肪环数 (Lipinski)
                Lipinski.NumAromaticCarbocycles(mol),   # 芳香族碳环数
                Lipinski.NumAromaticHeterocycles(mol),  # 芳香族杂环数
                Lipinski.NumSaturatedCarbocycles(mol),  # 饱和碳环数
                Lipinski.NumSaturatedHeterocycles(mol), # 饱和杂环数
                Lipinski.NumSaturatedRings(mol),        # 饱和环数
            ])
            
            return np.array(descriptors)
        except:
            return None

    def extract_features(self, smiles: str) -> Union[np.ndarray, None]:
        """
        从SMILES字符串提取所有特征（指纹+描述符）

        Args:
            smiles: SMILES字符串

        Returns:
            特征向量或None（如果转换失败）
        """
        # 提取指纹
        fingerprint = self.extract_fingerprint(smiles)
        if fingerprint is None:
            return None

        # 提取描述符
        descriptors = self.extract_descriptors(smiles)
        if descriptors is None:
            return None

        # 拼接所有特征
        features = np.concatenate([fingerprint, descriptors])
        return features

    def extract_features_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取特征

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            (特征矩阵, 有效的SMILES列表)
        """
        features_list = []
        valid_smiles = []

        for smiles in smiles_list:
            features = self.extract_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_smiles.append(smiles)

        if len(features_list) > 0:
            return np.vstack(features_list), valid_smiles
        else:
            return np.array([]), []