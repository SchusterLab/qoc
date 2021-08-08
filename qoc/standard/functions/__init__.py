"""
functions - a directory for exposing common operations
"""

from qoc.standard.functions.convenience import (commutator,
                                                conjugate_transpose,
                                                krons,
                                                matmuls,
                                                rms_norm,
                                                column_vector_list_to_matrix,
                                                matrix_to_column_vector_list,)
from qoc.standard.functions.expm import expm
from qoc.standard.functions.expm_manual import conjugate_transpose_m
__all__ = [
    "commutator", "conjugate_transpose", "krons", "matmuls",
    "rms_norm",
    "column_vector_list_to_matrix", "matrix_to_column_vector_list",
    "expm","expm_manual","conjugate_transpose_m"
]
