from typing import Optional, List

import pandas as pd


def df_to_str(df: pd.DataFrame, columns_to_include: Optional[List[str]] = None) -> str:
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           'display.expand_frame_repr', False):
        if columns_to_include:
            return str(df[columns_to_include])
        return str(df)
