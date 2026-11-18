import torch
import numpy as np


def quaternion_xyzw_to_rotation_matrix(quaternion):
    """
    Convert one normalized-or-unnormalized [x, y, z, w] quaternion to a
    3x3 rotation matrix without importing SciPy.
    """
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(
            'quaternion must contain exactly four values [x, y, z, w]')

    norm = np.linalg.norm(q)
    if norm < np.finfo(np.float64).eps:
        raise ValueError('zero-norm quaternion cannot define a rotation')

    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def normalize_dino(img):
    _img = img[..., :3]
    _ndims = len(img.shape) - 1
    _dims = [1] * _ndims + [3]
    vmin = _img.reshape(-1, 3).min(dim=0)[0].view(*_dims)
    vmax = _img.reshape(-1, 3).max(dim=0)[0].view(*_dims)
    return (_img - vmin) / (vmax - vmin)


def tf_msg_to_htm(tf_msg):
    p = np.array([
        tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z,
    ])
    q = np.array([
        tf_msg.transform.rotation.x,
        tf_msg.transform.rotation.y,
        tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w,
    ])
    R = quaternion_xyzw_to_rotation_matrix(q)
    htm = np.eye(4)
    htm[:3, :3] = R
    htm[:3, -1] = p
    return torch.from_numpy(htm).float()


def _pcl_msg_to_numpy(pcl_msg):
    """Parse PointCloud2 message into a structured numpy array without ros_numpy."""
    import struct

    dtype_map = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    names = []
    formats = []
    offsets = []
    for field in pcl_msg.fields:
        names.append(field.name)
        formats.append(dtype_map[field.datatype])
        offsets.append(field.offset)

    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                      'itemsize': pcl_msg.point_step})
    try:
        cloud_arr = np.frombuffer(pcl_msg.data, dtype=dtype)
    except (TypeError, BufferError):
        # Some custom PointCloud2 wrappers do not expose the buffer protocol.
        cloud_arr = np.frombuffer(bytes(pcl_msg.data), dtype=dtype)
    return cloud_arr


def pcl_msg_to_xyz(pcl_msg):
    pcl_np = _pcl_msg_to_numpy(pcl_msg)
    # Fill the final float32 array directly. np.stack followed by astype made
    # two full Nx3 allocations (plus three flatten copies) per point cloud.
    xyz = np.empty((pcl_np.size, 3), dtype=np.float32)
    xyz[:, 0] = pcl_np['x'].reshape(-1)
    xyz[:, 1] = pcl_np['y'].reshape(-1)
    xyz[:, 2] = pcl_np['z'].reshape(-1)
    return torch.from_numpy(xyz)


def pcl_msg_to_xyzrgb(pcl_msg):
    pcl_np = _pcl_msg_to_numpy(pcl_msg)
    xyz = np.stack([pcl_np['x'], pcl_np['y'], pcl_np['z']], axis=-1)
    colors_raw = pcl_np['rgb'].view(np.uint32)
    red = ((colors_raw & 0x00FF0000) >> 16)
    green = ((colors_raw & 0x0000FF00) >> 8)
    blue = ((colors_raw & 0x000000FF) >> 0)
    colors = np.stack([red, green, blue], axis=-1) / 255.
    return torch.from_numpy(np.concatenate([xyz, colors], axis=-1).astype(np.float32)).float()


def pose_to_htm(pose):
    p = pose[:3]
    q = pose[3:7]
    R = quaternion_xyzw_to_rotation_matrix(q)
    htm = np.eye(4)
    htm[:3, :3] = R
    htm[:3, -1] = p
    return torch.from_numpy(htm).float()


def transform_points(points, htm):
    pt_pos = points[:, :3]
    pt_pos = torch.cat([pt_pos, torch.ones_like(pt_pos[:, [0]])], dim=-1)
    pt_tf_pos = htm.view(1, 4, 4) @ pt_pos.view(-1, 4, 1)
    points[:, :3] = pt_tf_pos[:, :3, 0]
    return points


def transform_xyz(points, htm):
    """
    Transform XYZ points and return a new tensor without first cloning the
    source. The homogeneous matrix multiplication is identical to
    transform_points; this form is for callers that need both coordinate
    frames and only consume transformed XYZ.
    """
    pt_pos = points[:, :3]
    pt_pos = torch.cat(
        [pt_pos, torch.ones_like(pt_pos[:, [0]])], dim=-1)
    pt_tf_pos = htm.view(1, 4, 4) @ pt_pos.view(-1, 4, 1)
    return pt_tf_pos[:, :3, 0]
