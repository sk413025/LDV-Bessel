from typing import Tuple, NamedTuple

class PhysicsVerification(NamedTuple):
    """物理參數驗證結果"""
    is_valid: bool
    actual_value: float
    valid_range: Tuple[float, float]
    description: str  # Changed from 'message' to 'description' to match usage