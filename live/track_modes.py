"""Which body parts may drive the avatar: Auto, Full body, Upper body, or Head & face.

Applied to a live packet before it reaches Blender or a recording. Parts that
are switched off are marked not visible, so the receiver holds them still.
"""
MODES = {
    'auto': 'Auto (legs when clearly seen)',
    'full': 'Full body',
    'upper': 'Upper body (legs still)',
    'head': 'Head & face only',
}
LEGS = range(23, 33)
HEAD = range(0, 11)
# Leg landmark -> the one above it: a knee counts only with its hip, an ankle with its knee, heel/toe with the ankle.
LEG_PARENT = {25: 23, 26: 24, 27: 25, 28: 26, 29: 27, 30: 28, 31: 27, 32: 28}
# Auto mode: MediaPipe still reports legs hidden under a desk or behind the frame edge, often 0.75-0.9 visible.
AUTO_KNEE, AUTO_ANKLE = 0.9, 0.85


def leg_chain(valid):
    """Partly visible legs: keep only the connected part from the hip down, so a guessed ankle or knee
    below a hidden one (under a desk, past the frame edge) never drives the leg."""
    valid = list(valid)
    for i in sorted(LEG_PARENT):  # parents come first, so one pass follows the chain
        valid[i] = valid[i] and valid[LEG_PARENT[i]]
    return valid


def lying(packet):
    """Torso more than 45 deg from upright (pose in Blender axes, z up), shoulders and hips seen."""
    v, P = packet['pose_valid'], packet.get('pose')
    if not P or not all(v[i] for i in (11, 12, 23, 24)):
        return False
    d = [(P[11][k] + P[12][k] - P[23][k] - P[24][k]) / 2 for k in range(3)]
    n = sum(x * x for x in d) ** 0.5
    return n > 1e-6 and d[2] / n < 0.707


def auto_legs(packet):
    """Auto: hips always count when seen (lying down, bending at the waist); a leg drives only where it is
    clearly in view, from the hip down. Standing in full view = full body; at a desk = legs still."""
    vis = packet.get('pose_visibility') or [1.0] * len(packet['pose_valid'])
    valid = list(packet['pose_valid'])
    if lying(packet):  # lying down the legs are in view (no desk to hide them) but MediaPipe rates them low
        return leg_chain(valid)
    for i in (25, 26):
        valid[i] = valid[i] and vis[i] >= AUTO_KNEE
    for i in (27, 28, 29, 30, 31, 32):
        valid[i] = valid[i] and vis[i] >= AUTO_ANKLE
    return leg_chain(valid)


def apply_mode(packet, mode):
    if mode == 'full' or 'pose_valid' not in packet:
        return packet
    if mode == 'auto':
        packet = dict(packet, pose_valid=auto_legs(packet))
        legs = [i for i in (25, 26) if packet['pose_valid'][i]]
        packet['track_mode'] = 'auto:full' if len(legs) == 2 else 'auto:upper' if not legs else 'auto:partial'
        return packet
    keep = (lambda i: i not in LEGS) if mode == 'upper' else (lambda i: i in HEAD)
    packet = dict(packet)
    packet['pose_valid'] = [bool(v and keep(i)) for i, v in enumerate(packet['pose_valid'])]
    if mode == 'head':
        packet.pop('hands', None)
    packet['track_mode'] = mode
    return packet
