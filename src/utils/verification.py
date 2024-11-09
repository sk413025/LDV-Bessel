from dataclasses import dataclass
from typing import Tuple, Any

@dataclass
class PhysicsVerification:
    """物理參數驗證結果類"""
    is_valid: bool              # 驗證結果
    actual_value: Any          # 實際數值
    valid_range: Tuple[Any, Any]  # 有效範圍
    description: str           # 驗證描述
    
    def __str__(self) -> str:
        """格式化輸出驗證結果"""
        status = "通過" if self.is_valid else "未通過"
        return (
            f"驗證項目: {self.description}\n"
            f"驗證結果: {status}\n"
            f"實際數值: {self.actual_value}\n"
            f"有效範圍: {self.valid_range[0]} 到 {self.valid_range[1]}"
        )

    def get_validation_message(self) -> str:
        """獲取驗證訊息"""
        if self.is_valid:
            return f"{self.description}: 參數在有效範圍內"
        return (
            f"{self.description}: 參數超出有效範圍\n"
            f"當前值: {self.actual_value}, "
            f"應在 {self.valid_range[0]} 到 {self.valid_range[1]} 之間"
        )

    def validate(self) -> bool:
        """執行驗證並返回結果"""
        return self.is_valid