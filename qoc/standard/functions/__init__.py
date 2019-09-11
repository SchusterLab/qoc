"""
functions - a directory for exposing common operations
"""

from qoc.standard.functions.convenience import (commutator,
                                                conjugate,
                                                conjugate_transpose,
                                                krons,
                                                l2_norm,
                                                matmuls,
                                                mult_cols, mult_rows,
                                                rms_norm,
                                                transpose,
                                                column_vector_list_to_matrix,
                                                matrix_to_column_vector_list,
                                                complex_to_real_imag_flat,
                                                real_imag_to_complex_flat,)
from qoc.standard.functions.expm import expm

__all__ = [
    "commutator", "conjugate", "conjugate_transpose", "krons", "l2_norm", "matmuls",
    "rms_norm",
    "column_vector_list_to_matrix", "matrix_to_column_vector_list",
    "complex_to_real_imag_flat", "real_imag_to_complex_flat",
    "expm",
]
