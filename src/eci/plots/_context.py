from typing import ContextManager

import seaborn as sns

STYLE = "whitegrid"


def _get_context() -> ContextManager:
    """Return a seaborn context manager enforcing the ECI plot style."""
    return sns.axes_style(
        STYLE,
        rc={
            "axes.facecolor": "#f9f9f9",
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "font.family": "sans-serif",
        },
    )
