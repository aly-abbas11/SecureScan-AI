"""
Preprocessing pipeline for SecureScan AI.
Handles cleaning, balancing, and tokenization.
"""

import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import Tuple


class CodePreprocessor:
    """Preprocessing pipeline for source code vulnerability detection.
    
    Args:
        tokenizer_name: HuggingFace tokenizer name.
        max_length: Maximum token sequence length.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        tokenizer_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        seed: int = 42
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.seed = seed

    def clean_code(self, code: str) -> str:
        """Remove comments and extra whitespace from code.
        
        Args:
            code: Raw source code string.
        
        Returns:
            Cleaned source code string.
        """
        code = re.sub(r'//.*', '', str(code))
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'\n\s*\n', '\n', code)
        return code.strip()

    def balance(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        ratio: int = 3
    ) -> pd.DataFrame:
        """Balance dataset using undersampling.
        
        Fit on training data only to prevent leakage.
        
        Args:
            df: Input dataframe.
            label_col: Name of label column.
            ratio: Ratio of safe to vulnerable samples.
        
        Returns:
            Balanced dataframe.
        """
        vuln = df[df[label_col] == 1]
        safe = df[df[label_col] == 0].sample(
            n=min(len(vuln) * ratio, len(df[df[label_col] == 0])),
            random_state=self.seed)
        return pd.concat([vuln, safe]).sample(
            frac=1, random_state=self.seed).reset_index(drop=True)

    def split(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        test_size: float = 0.15,
        val_size: float = 0.176
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, val, test sets.
        
        Args:
            df: Input dataframe.
            label_col: Name of label column.
            test_size: Fraction for test set.
            val_size: Fraction of remaining for val set.
        
        Returns:
            Tuple of (train, val, test) dataframes.
        """
        train_val, test = train_test_split(
            df, test_size=test_size,
            random_state=self.seed, stratify=df[label_col])
        train, val = train_test_split(
            train_val, test_size=val_size,
            random_state=self.seed, stratify=train_val[label_col])
        return train, val, test

    def tokenize(self, texts) -> dict:
        """Tokenize source code texts.
        
        Args:
            texts: List or Series of code strings.
        
        Returns:
            Dictionary with input_ids and attention_mask tensors.
        """
        return self.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
