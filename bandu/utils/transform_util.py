import torch

def torch_quat2mat(quat):
    # from DCP code
    # Quaternion to rotation matrix
    if len(quat.shape) == 1:
        quat = quat.unsqueeze(0)
    if len(quat.shape) == 2:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
        return rotMat
    elif len(quat.shape) == 3:
        # nO: num_objects
        nO = quat.size(1)

        # select one index out of [nB, nO, 4] -> [nB, nO] for each scalar component
        x, y, z, w = quat[:, :, 0], quat[:, :, 1], quat[:, :, 2], quat[:, :, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)

        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=2).reshape(B, nO, 3, 3)
        return rotMat
    else:
        raise NotImplementedError