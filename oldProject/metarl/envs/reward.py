"""
Motion imitation reward per MetaLoco Eq. (2)-(3).
r = (r_h * r_v * r_ee * r_yawrate) * (r_daction * r_slip * r_pitchroll)
Each r_x = exp(-||x - x_ref||^2 / sigma_x^2).
"""
import numpy as np

def rbf(x, xref, sigma):
    diff = np.array(x) - np.array(xref)
    if np.isscalar(sigma):
        denom = sigma**2
    else:
        sigma = np.array(sigma)
        denom = np.sum(sigma**2)
    return float(np.exp(- (np.sum(diff*diff)) / (denom + 1e-8)))

def imitation_and_regularization(obs, act, prev_act, refs, sigma, weights):
    # refs: dict with "h","v","feet","yawrate"
    rh = rbf(obs["base_height"], refs["h"], sigma["base_height"])
    rv = rbf(obs["base_linvel"], refs["v"], sigma["base_velocity"])
    ree = rbf(obs["feet_pos"], refs["feet"], sigma["feet_pos"])
    ry = rbf(obs["yaw_rate"], refs["yawrate"], sigma["yaw_rate"] if "yaw_rate" in sigma else sigma["yaw_rate"] if "yaw_rate" in sigma else 0.5)

    rI = (rh**weights.get("rh",1.0)) * (rv**weights.get("rv",1.0)) * (ree**weights.get("ree",1.0)) * (ry**weights.get("r_yawrate",1.0))

    # Regularizers (placeholders; you'll compute slip velocity, pitch/roll from state)
    r_daction = rbf(act, prev_act, sigma["action_rate"])
    r_slip = 1.0  # TODO: exp(-||contact_feet_vel||^2 / sigma_slip^2)
    r_pr = 1.0    # TODO: exp(-||[pitch,roll]||^2 / sigma_pitchroll^2)

    rR = (r_daction**weights.get("r_daction",0.1)) * (r_slip**weights.get("r_slip",0.1)) * (r_pr**weights.get("r_pitchroll",0.2))
    return rI * rR
