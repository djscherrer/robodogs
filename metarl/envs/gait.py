"""
Gait phase utilities and simple foothold planning.

References: MetaLoco Sec. IV-A (gait planner), Shao et al. (phase parameterization).
"""
import numpy as np

def contact_phase(t, freq_hz=2.0):
    """Return phase in [-pi, pi), swing: [-pi,0), stance: [0,pi)."""
    phase = (2*np.pi*freq_hz*t) % (2*np.pi) - np.pi
    return phase

def phase_embedding(phase):
    return np.cos(phase), np.sin(phase)

def trot_contact_schedule(t, freq_hz=2.0):
    """Diagonal legs in phase opposition for trot. Returns list[4] of {0,1} contacts."""
    ph = contact_phase(t, freq_hz)
    stance = (ph >= 0.0).astype(int) if hasattr(ph, "astype") else (1 if ph >= 0 else 0)
    # Very simple: alternate pairs (LF+RH) vs (RF+LH)
    return [stance, 1-stance, 1-stance, stance]
