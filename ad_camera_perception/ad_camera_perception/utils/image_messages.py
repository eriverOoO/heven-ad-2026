"""ROS image message helpers supporting raw and compressed transports."""

from typing import Type, Union

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


ImageMessage = Union[Image, CompressedImage]


def image_message_type(image_transport: str) -> Type[ImageMessage]:
    """Return the ROS message class for ``raw`` or ``compressed`` input."""
    if image_transport == "raw":
        return Image
    if image_transport == "compressed":
        return CompressedImage
    raise ValueError("image_transport must be 'raw' or 'compressed'")


def image_message_to_bgr(
    message: ImageMessage, image_transport: str
) -> np.ndarray:
    """Decode one ROS image message as a BGR OpenCV array."""
    if image_transport == "raw":
        # Keep cv_bridge off the compressed Camera-4 path. Some ROS Humble
        # cv_bridge binaries are ABI-incompatible with a system-wide NumPy 2.
        from cv_bridge import CvBridge

        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    if image_transport == "compressed":
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("compressed image could not be decoded")
        return image
    raise ValueError("image_transport must be 'raw' or 'compressed'")
