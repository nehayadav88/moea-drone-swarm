"""Multi-objective evolutionary algorithms for drone swarm path planning.

Algorithms implemented (inspired by PlatEMO):
  - NSGA-II   (Deb et al., 2002)
  - NSGA-III  (Deb & Jain, 2014)
  - MOEA/D    (Zhang & Li, 2007)
  - SPEA2     (Zitzler et al., 2001)
  - MOPSO     (Coello Coello et al., 2004)
  - RVEA      (Cheng et al., 2016)
  - IBEA      (Zitzler & Künzli, 2004)
  - GDE3      (Kukkonen & Lampinen, 2005)
  - SMS-EMOA  (Beume et al., 2007)
  - AGE-MOEA  (Panichella, 2019)
  - HypE      (Bader & Zitzler, 2011)
"""

from .nsga2 import NSGA2
from .nsga3 import NSGA3
from .moead import MOEAD
from .spea2 import SPEA2
from .mopso import MOPSO
from .rvea import RVEA
from .ibea import IBEA
from .gde3 import GDE3
from .sms_emoa import SMSEMOA
from .age_moea import AGEMOEA
from .hype import HypE

ALGORITHM_REGISTRY = {
    "nsga2": NSGA2,
    "nsga3": NSGA3,
    "moead": MOEAD,
    "spea2": SPEA2,
    "mopso": MOPSO,
    "rvea": RVEA,
    "ibea": IBEA,
    "gde3": GDE3,
    "sms_emoa": SMSEMOA,
    "age_moea": AGEMOEA,
    "hype": HypE,
}

__all__ = [
    "NSGA2", "NSGA3", "MOEAD", "SPEA2", "MOPSO",
    "RVEA", "IBEA", "GDE3", "SMSEMOA", "AGEMOEA", "HypE",
    "ALGORITHM_REGISTRY",
]
