from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
import uproot


class RootReader:
    """
    Read ROOT TTrees into pandas DataFrames using uproot.

    Parameters
    ----------
    tree_name:
        Name of the TTree inside the ROOT files.
    step_size:
        Chunk size for streaming. Can be int (#entries) or a string like "200 MB".
    """

    def __init__(self, tree_name: str, step_size: Union[int, str] = "200 MB"):
        self.tree_name = tree_name
        self.step_size = step_size

    def iterate(
        self,
        files: Sequence[str],
        branches: Sequence[str],
        cut: Optional[str] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> Iterable[pd.DataFrame]:
        """
        Stream chunks as DataFrames.
        """
        # Uproot expects file paths with optional ":treename" suffix
        file_specs = [f"{f}:{self.tree_name}" for f in files]

        for chunk in uproot.iterate(
            file_specs,
            expressions=list(branches),
            aliases=aliases,
            step_size=self.step_size,
            library="pd",
            allow_missing=False,
        ):
            if cut:
                chunk = chunk.query(cut)
            yield chunk

    def read(
        self,
        files: Sequence[str],
        branches: Sequence[str],
        cut: Optional[str] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Read all chunks and concatenate.
        """
        dfs: List[pd.DataFrame] = []
        for df in self.iterate(files=files, branches=branches, cut=cut, aliases=aliases):
            dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=list(branches))
        return pd.concat(dfs, ignore_index=True)
