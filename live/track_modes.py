"""Which body parts may drive the avatar: Full body, Upper body, or Head & face.

Applied to a live packet before it reaches Blender or a recording. Parts that
are switched off are marked not visible, so the receiver holds them still.
"""
MODES = {
    'full': 'Full body',
    'upper': 'Upper body (legs still)',
    'head': 'Head & face only',
}
LEGS = range(23, 33)
HEAD = range(0, 11)


def apply_mode(packet, mode):
    if mode == 'full' or 'pose_valid' not in packet:
        return packet
    keep = (lambda i: i not in LEGS) if mode == 'upper' else (lambda i: i in HEAD)
    packet = dict(packet)
    packet['pose_valid'] = [bool(v and keep(i)) for i, v in enumerate(packet['pose_valid'])]
    if mode == 'head':
        packet.pop('hands', None)
    packet['track_mode'] = mode
    return packet
