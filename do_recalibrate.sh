#!/bin/bash
# Script thực hiện Recalibrate theo 4 bước yêu cầu

echo "========================================="
echo "BƯỚC 1: Kill các process capture cũ"
echo "========================================="
pkill -9 -f "pipeline/capture.py"
pkill -9 -f "pipeline/demo_stream.py"
sleep 1
echo "Kiểm tra process còn sót lại (nếu trống là OK):"
ps aux | grep -E "capture\.py|demo_stream\.py" | grep -v grep

echo -e "\n========================================="
echo "BƯỚC 2: Backup weights TRƯỚC KHI RECALIBRATE"
echo "========================================="
cd "edge-ai-traffic-anomaly" || exit 1
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

if [ -f "model/weights/mu_norm.npy" ] && [ -f "model/weights/delta.npy" ]; then
    cp model/weights/mu_norm.npy "model/weights/mu_norm.npy.bak_$BACKUP_DATE"
    cp model/weights/delta.npy "model/weights/delta.npy.bak_$BACKUP_DATE"
    echo "Đã backup thành:"
    echo " - mu_norm.npy.bak_$BACKUP_DATE"
    echo " - delta.npy.bak_$BACKUP_DATE"
else
    echo "⚠️ Lỗi: Không tìm thấy model/weights/mu_norm.npy hoặc delta.npy để backup!"
    exit 1
fi

echo -e "\nDelta CŨ trong config.json:"
grep "delta" model/weights/config.json

echo -e "\n========================================="
echo "BƯỚC 3: Chạy Recalibrate Threshold (300 giây)"
echo "========================================="
echo "Vui lòng đợi 5 phút để quá trình capture và tính toán hoàn tất..."
../venv/bin/python scripts/recalibrate_threshold.py \
    --iface wlp61s0 \
    --duration 300 \
    --percentile 99

echo -e "\n========================================="
echo "BƯỚC 4: Xác thực kết quả Recalibrate"
echo "========================================="
echo "Delta MỚI trong config.json:"
grep "delta" model/weights/config.json

echo -e "\nSo sánh mu_norm.npy:"
../venv/bin/python -c "
import numpy as np
import sys
try:
    old = np.load('model/weights/mu_norm.npy.bak_$BACKUP_DATE')
    new = np.load('model/weights/mu_norm.npy')
    delta = np.load('model/weights/delta.npy')
    
    print(f'Old mu_norm shape: {old.shape}')
    print(f'New mu_norm shape: {new.shape}')
    print(f'Old first 5 vals : {old[0][:5]}')
    print(f'New first 5 vals : {new[0][:5]}')
    print(f'New delta.npy val: {delta}')
    
    if delta > 50000:
        print('\n❌ CẢNH BÁO: Delta mới quá lớn (hàng chục ngàn/triệu). Recalibrate CÓ VẤN ĐỀ!')
    elif delta < 10:
        print('\n❌ CẢNH BÁO: Delta mới quá nhỏ. Recalibrate CÓ VẤN ĐỀ!')
    else:
        print('\n✅ Delta nằm trong khoảng dự kiến.')

    if not np.allclose(old, new):
        print('✅ mu_norm.npy đã được cập nhật thành công (giá trị đã thay đổi).')
    else:
        print('⚠️ CẢNH BÁO: mu_norm.npy giống hệt bản cũ. Có thể chưa được ghi đè!')
except Exception as e:
    print(f'Lỗi khi kiểm tra numpy files: {e}')
"
echo "========================================="
echo "HOÀN TẤT! Nếu mọi thứ xanh (✅), hãy chạy lại live capture:"
echo "sudo bash start.sh"
