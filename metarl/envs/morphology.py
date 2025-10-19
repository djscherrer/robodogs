"""
Procedural morphology generation per MetaLoco Sec. IV-D:
- For link types (base, calf, thigh), scale dimensions lx, ly, lz and mass m by U[0.5, 1.5].
- Recompute cube inertia matrix I (Eq. 4).
"""
import numpy as np

def sample_scales(rng, low=0.5, high=1.5):
    return rng.uniform(low, high, size=4)  # sx, sy, sz, sm

def cube_inertia(m, lx, ly, lz):
    Ixx = (1/12)*m*(ly**2 + lz**2)
    Iyy = (1/12)*m*(lx**2 + lz**2)
    Izz = (1/12)*m*(lx**2 + ly**2)
    # Simple diagonal inertia (omit products for cube approx); extend as needed
    return np.array([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]])

class MorphologySpec:
    def __init__(self, link_dims, link_masses):
        """link_dims: dict[name]->(lx,ly,lz), link_masses: dict[name]->m"""
        self.link_dims = dict(link_dims)
        self.link_masses = dict(link_masses)

    def randomized(self, rng=np.random.RandomState(0), scale_range=(0.5,1.5)):
        dims2, masses2, inertias2 = {}, {}, {}
        for name,(lx,ly,lz) in self.link_dims.items():
            sx,sy,sz,sm = sample_scales(rng, *scale_range)
            lx2, ly2, lz2 = lx*sx, ly*sy, lz*sz
            m2 = self.link_masses[name]*sm
            I = cube_inertia(m2, lx2, ly2, lz2)
            dims2[name] = (lx2, ly2, lz2)
            masses2[name] = m2
            inertias2[name] = I
        return dims2, masses2, inertias2
