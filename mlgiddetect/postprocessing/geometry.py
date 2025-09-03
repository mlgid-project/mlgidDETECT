import numpy as np
from mlgiddetect.configuration import Config
import torch

def polar_to_q(config: Config, radii: np.array, angles: np.array, q_max: float):
    """
    Convert polar coordinates to Q-space coordinates.

    Parameters:
    config (Config)
    radii (array-like): Array of radii values in pixel.
    angles (array-like): Array of angle values in pixels.
    q_max (float): Maximum Q value.

    Returns:
    tuple: Two arrays representing the x and y coordinates in Q-space.
    """
    #returns boxes in xy and z in Q-space
    radius_q = (radii/config.PREPROCESSING_POLAR_SHAPE[1]) * q_max
    angle_rad = np.deg2rad((angles / config.PREPROCESSING_POLAR_SHAPE[0]) * 90)
    if config.PREPROCESSING_QUAZIPOLAR:
        angle_rad = np.deg2rad((angles / config.PREPROCESSING_POLAR_SHAPE[0] * q_max/radius_q * 90))*.60#0.619047619

    return radius_q * np.cos(angle_rad), radius_q * np.sin(angle_rad)    

def boxes_polar_to_reciprocal(config: Config, polar_boxes: np.array, q_max: float = 2.5):
    """
    Convert polar coordinates to reciprocal space coordinates.

    Parameters:
    config
    polar_boxes (numpy.ndarray): Array of polar coordinates in the format [x_min, y_min, x_max, y_max].
    q_max (float, optional): Maximum q value. Defaults to 2.5.

    Returns:
    numpy.ndarray: Array of reciprocal space coordinates in the format [xy_min, z_min, xy_max, z_min].
    """

    if config.GEO_QMAX:
        q_max = config.GEO_QMAX

    reciprocal_boxes =  list(polar_to_q(config, polar_boxes[:,0], polar_boxes[:,1], q_max=q_max))
    for i in polar_to_q(config, polar_boxes[:,2], polar_boxes[:,3], q_max=q_max): reciprocal_boxes.append(i)
    return np.array(reciprocal_boxes).T

def boxes_reciprocal_q_to_xy(config, img_container, reciprocal_q_boxes: np.array):
    """
    Convert reciprocal space coordinates to real space coordinates.

    Parameters:
    config (Config): Configuration object containing the GEO_PIXELPERANGSTROEM attribute.
    reciprocal_q_boxes (np.array): Array of reciprocal space coordinates.

    Returns:
    list: A list containing the following elements:
        - radius (np.array): The radius in real space.
        - angle (np.array): The angle in degrees, adjusted to be in the first quadrant.
        - angle_std (np.array): The standard deviation of the angles.
        - width (np.array): The width of the box in real space.
    """

    radius1 = np.sqrt(((reciprocal_q_boxes[:,0]) ** 2) + (((reciprocal_q_boxes[:,1])) ** 2))
    radius2 = np.sqrt(((reciprocal_q_boxes[:,2]) ** 2) + (((reciprocal_q_boxes[:,3])) ** 2))
    theta1 = np.degrees(np.arctan2(reciprocal_q_boxes[:,0], (reciprocal_q_boxes[:,1]))) + 270
    theta2 = np.degrees(np.arctan2(reciprocal_q_boxes[:,2], (reciprocal_q_boxes[:,3]))) + 270

    img_container.radius = (radius1 + radius2) / 2
    img_container.radius_width = radius2 - radius1
    img_container.angle = 360 - (theta1 + theta2) / 2
    img_container.angle_width = theta2 - theta1

    return img_container

def polar_to_cartesian(img_container):
    img_container.qzqxyboxes = np.array([ img_container.radius*np.sin(np.deg2rad(img_container.angle)), img_container.radius*np.cos(np.deg2rad(img_container.angle))])
    return img_container