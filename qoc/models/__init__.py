"""
models - a directory for qoc's data models 
"""

from qoc.models.cost import Cost
<<<<<<< HEAD
from qoc.models.grapestate import (GrapeStateDiscrete, GrapeResult, EvolveResult)
from qoc.models.magnusmethod import MagnusMethod
from qoc.models.operationtype import OperationType
from qoc.models.optimizer import Optimizer

__all__ = ["Cost", "GrapeStateDiscrete", "EvolveResult", "GrapeResult",
           "MagnusMethod", "OperationType", "Optimizer"]
=======
from qoc.models.grapestate import (GrapeStateDiscrete,)
from qoc.modles.magnusorder import MagnusOrder
from qoc.models.operationtype import OperationType
from qoc.models.optimizer import Optimizer

__all__ = ["Cost", "GrapeStateDiscrete", "MagnusOrder", "OperationType", "Optimizer"]
>>>>>>> 318831567b7dd9196600b9274c11d12a071d3af0
