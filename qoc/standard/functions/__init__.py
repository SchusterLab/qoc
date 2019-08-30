"""
functions - a directory for exposing common operations
"""

from qoc.standard.functions.convenience import (commutator,
                                                conjugate,
                                                conjugate_transpose,
                                                krons,
                                                matmuls,
                                                mult_cols, mult_rows,
                                                transpose,
                                                column_vector_list_to_matrix,
                                                matrix_to_column_vector_list,
                                                complex_to_real_imag_flat,
                                                real_imag_to_complex_flat,)

from qoc.standard.functions.expm import expm, expm_vjp

__all__ = [
    "commutator", "conjugate", "conjugate_transpose", "krons", "matmuls",
    "column_vector_list_to_matrix", "matrix_to_column_vector_list",
    "complex_to_real_imag_flat", "real_imag_to_complex_flat",
    "abs_gpu", "add_gpu", "conj_gpu", "divide_gpu", "matmul_gpu",
    "multiply_gpu", "subtract_gpu", "trace_gpu", "transpose_gpu",
    "expm",
]
