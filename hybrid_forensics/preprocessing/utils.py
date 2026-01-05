"""
Preprocessing utilities for mouth cropping and landmark handling
Adapted from LipForensics preprocessing
"""

import numpy as np
from skimage import transform as tf


def warp_img(src, dst, img, std_size):
    """
    Warp image to match mean face landmarks using affine transformation
    
    Parameters
    ----------
    src : numpy.array
        Key landmarks of initial face
    dst : numpy.array
        Key landmarks of mean face
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    
    Returns
    -------
    tuple
        (warped_image, transformation)
    """
    tform = tf.estimate_transform('similarity', src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    """
    Apply affine transformation to image
    
    Parameters
    ----------
    transform : skimage.transform._geometric.GeometricTransform
        Transformation object
    img : numpy.array
        Frame to be aligned
    std_size : tuple
        Target size for frames
    
    Returns
    -------
    numpy.array
        Transformed image
    """
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    """
    Crop square mouth region given landmarks
    
    Parameters
    ----------
    img : numpy.array
        Frame to be cropped
    landmarks : numpy.array
        Landmarks corresponding to mouth region (20 points)
    height : int
        Half-height of output image
    width : int
        Half-width of output image
    threshold : int
        Threshold for boundary checking
    
    Returns
    -------
    numpy.array
        Cropped mouth region
    """
    center_x, center_y = np.mean(landmarks, axis=0)
    
    if center_y - height < 0:
        center_y = height
    if int(center_y) - height < 0 - threshold:
        raise Exception("too much bias in height")
    
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")
    
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")
    
    img_cropped = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return img_cropped


def linear_interpolate(landmarks, start_idx, stop_idx):
    """
    Linear interpolation of landmarks between two frames
    
    Parameters
    ----------
    landmarks : list
        List of landmarks
    start_idx : int
        Start frame index
    stop_idx : int
        Stop frame index
    
    Returns
    -------
    list
        Interpolated landmarks
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """
    Interpolate landmarks for frames where detection failed
    
    Parameters
    ----------
    landmarks : list
        List of landmarks (None for failed detections)
    
    Returns
    -------
    list or None
        Interpolated landmarks, or None if no valid landmarks found
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    
    if not valid_frames_idx:
        return None
    
    # Interpolate between valid frames
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(
                landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
            )
    
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    
    # Handle frames at beginning and end
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )
    
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    
    return landmarks
