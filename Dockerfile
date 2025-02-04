# 使用 PyTorch CPU 版本作為基礎映像
FROM pytorch/pytorch:latest

# 設定工作目錄
WORKDIR /app

# 複製 Python 代碼 & 數據集
COPY train.py .
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 預設執行 Python 腳本
CMD ["python", "train.py"]
