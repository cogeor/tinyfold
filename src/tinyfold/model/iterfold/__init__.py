"""IterFold model components.

Anchor-conditioned structure prediction:
- IterFold: Main model with trunk encoder and anchor conditioning
- AnchorDecoder: Anchor-conditioned decoder
- FrameDecoder: Frame-based decoder (centroid + rotation)
"""

from tinyfold.model.iterfold.model import IterFold, AnchorDecoder
from tinyfold.model.iterfold.frame_decoder import FrameDecoder

__all__ = [
    "IterFold",
    "AnchorDecoder",
    "FrameDecoder",
]
