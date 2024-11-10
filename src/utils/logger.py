
import logging
import os
from datetime import datetime

class LDVLogger:
    """LDV系統 logger"""
    
    def __init__(self, name: str = "LDV"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 設定 log 目錄
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 檔案處理器 - 詳細記錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            f"{log_dir}/ldv_{timestamp}.log",
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        
        # 控制台處理器 - 重要訊息
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 設定格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)

# 創建全局logger實例
logger = LDVLogger().logger