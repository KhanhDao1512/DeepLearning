# Bài thực hành 3: RNN cho phân loại văn bản và gán nhãn chuỗi

| Bài | Mô hình | Dữ liệu | Độ đo | Script |
|-----|---------|---------|-------|--------|
| 1 | LSTM 5 lớp, hidden 256 | UIT-VSFC | F1 macro | `assignment_1.py` |
| 2 | GRU 5 lớp, hidden 256 | UIT-VSFC | F1 macro | `assignment_2.py` |
| 3 | Encoder 5 lớp BiLSTM, hidden 256 | PhoNER_COVID19 | F1 thực thể (seqeval) | `assignment_3.py` |

## Chuẩn bị dữ liệu
- UIT-VSFC: đặt `UIT-VSFC-{train,dev,test}.json` vào `datasets/UIT-VSFC/`.
- PhoNER_COVID19: đặt `{train,dev,test}_word.conll` vào `datasets/PhoNER_COVID19/`
  (nếu dùng bản syllable, sửa đường dẫn đầu `assignment_3.py`).

## Chạy (từ thư mục gốc)
```
pip install -r requirements.txt
python assignment_1.py
python assignment_2.py
python assignment_3.py
```

## Kết quả (F1 trên test)
| Bài | Mô hình | Best epoch | Dev F1 | Test F1 |
|-----|---------|-----------|--------|---------|
| 1 | LSTM 5 lớp | 5 | 0.7543 | 0.7637 |
| 2 | GRU 5 lớp | 8 | 0.7520 | 0.7539 |
| 3 | BiLSTM 5 lớp (NER) | 22 | 0.9011 | 0.8910 |

Log huấn luyện: `logs/`. Kết quả chi tiết từng epoch: `results/`. Model tốt nhất: `checkpoints/`.
