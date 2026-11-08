# Kịch bản demo (khoảng 3 phút)

Demo chạy trên ảnh VisA thật, bằng code của repo, trên máy Apple M4 (MPS). Toàn bộ mất khoảng 50 giây
máy chạy, phần còn lại là thời gian nói.

## Chuẩn bị trước buổi trình bày

```bash
cd fscad
source .venv/bin/activate          # hoặc: uv sync
ls data/visa/pcb1 splits/visa/pcb1 # cần dữ liệu VisA và split (xem HUONG_DAN.md)
python scripts/demo_visa.py --no-pause   # chạy thử một lần: tải trọng số DINOv2 vào cache
```

Chạy thử một lần trước để trọng số DINOv2 đã nằm trong cache, lúc trình bày không phải chờ tải. Mở sẵn
một cửa sổ terminal chữ to và một cửa sổ xem ảnh `outputs/demo/demo_panel.png`.

## Lúc trình bày

```bash
python scripts/demo_visa.py
```

Mỗi bước dừng lại chờ bấm Enter. Lời dẫn gợi ý cho từng bước như sau.

**Mở đầu, trong lúc nạp mô hình (khoảng 5 giây).** "Đây là DINOv2 ViT-S/14, giữ đóng băng và dùng chung
cho mọi sản phẩm. Hệ thống hiện chưa biết sản phẩm nào cả."

**Bước 1, đăng ký 3 sản phẩm (khoảng 20 giây).** "Em đăng ký pcb1, candle và capsules, mỗi sản phẩm chỉ
từ 4 ảnh tốt. Tên 4 ảnh hiện ngay trên màn hình. Mỗi sản phẩm mất khoảng 7 giây với kNN và chiếm 18 MB.
Hệ thống tự đặt ngưỡng lỗi δ cho từng sản phẩm, và ngưỡng từ chối τ cho bộ định tuyến."

**Bước 2, kiểm tra 8 ảnh (khoảng 3 giây).** Đọc bảng từ trên xuống:

| Ảnh | Kết quả | Điều cần nói |
|---|---|---|
| pcb1 ảnh tốt | ok | Router chọn đúng pcb1, điểm dưới ngưỡng |
| pcb1 lỗi nhỏ ở chân cắm | ok (bỏ sót) | Lỗi chỉ chiếm vài patch, đúng ca khó đã phân tích trong báo cáo |
| pcb1 lỗi lớn ở hai màng | TỪ CHỐI | Lỗi lớn làm token CLS lệch hẳn, nên router từ chối. Vì vậy ảnh bị từ chối phải được coi là cảnh báo (chính sách alarm) |
| candle ảnh tốt | ok | |
| candle lỗi ở mép nến | LỖI | Điểm vượt ngưỡng, bản đồ lỗi trùng vùng lỗi |
| capsules lỗi trên một viên | LỖI | Tìm đúng một viên lỗi giữa khoảng hai mươi viên |
| pcb4 chưa đăng ký | TỪ CHỐI | Sản phẩm lạ không bị chấm nhầm bằng adapter của sản phẩm khác |
| pcb1 ánh sáng đã đổi | TỪ CHỐI | Ánh sáng đổi mạnh làm ảnh rơi ra ngoài vùng router tin cậy |

**Bước 3, bản đồ lỗi.** Mở `outputs/demo/demo_panel.png`. Hàng trên là ảnh vào, viền đỏ là vùng lỗi
thật. Hàng dưới là bản đồ lỗi và quyết định. "Ở candle và capsules, vùng nóng nằm đúng chỗ lỗi. Ở pcb1 lỗi
nhỏ, bản đồ có điểm nóng ở chân cắm nhưng điểm cả ảnh vẫn dưới ngưỡng."

**Bước 4, giám sát drift (khoảng 15 giây).** "Em cho 60 ảnh tốt của pcb1 chạy qua như một luồng sản xuất,
lần đầu với ánh sáng gốc, lần sau với ánh sáng đã đổi. Luồng sạch không cảnh báo, còn luồng đổi ánh sáng
làm kiểm định KS trên token CLS bật cảnh báo drift. Bộ giám sát không cần nhãn."

**Kết.** "Toàn bộ hệ thống chạy trên một laptop, một xương sống dùng chung, mỗi sản phẩm mới chỉ cần vài
ảnh và vài giây đến vài phút để đăng ký."

## Kết quả của lần chạy thử (08/11/2026, kNN)

```text
pcb1, ảnh tốt                   ok         pcb1          0.976    0.110   0.177
pcb1, lỗi nhỏ ở chân cắm        ok         pcb1          0.973    0.128   0.177
pcb1, lỗi lớn ở hai màng        TỪ CHỐI    -             0.668        -       -
candle, ảnh tốt                 ok         candle        0.985    0.180   0.198
candle, lỗi ở mép nến           LỖI        candle        0.971    0.283   0.198
capsules, lỗi trên một viên     LỖI        capsules      0.980    0.306   0.185
pcb4, sản phẩm chưa đăng ký     TỪ CHỐI    -             0.519        -       -
pcb1, ánh sáng đã đổi           TỪ CHỐI    -             0.961        -       -
luồng sạch               p(CLS) = 0.066   p(điểm) = 0.92   ->  bình thường
ánh sáng đổi, mức 0,6    p(CLS) = 0.0054  p(điểm) = 0.12   ->  CẢNH BÁO DRIFT
```

## Các biến thể

- `--method lora`: đăng ký bằng LoRA như cấu hình (E) của báo cáo, khoảng 3 phút mỗi sản phẩm. Chỉ dùng
  khi có thời gian, hoặc chạy trước rồi trình bày kết quả.
- `--products checkpoints/visa-lora-k4-s0`: nạp các sản phẩm LoRA đã huấn luyện, tải từ Google Drive theo
  `HUONG_DAN.md`. Bước 1 khi đó chỉ nạp, không đăng ký lại.
- `--no-pause`: chạy liền một mạch, dùng để tập trước hoặc quay video.

Nếu máy không có dữ liệu VisA, có thể dùng demo API trên dữ liệu toy: `python scripts/demo.py`.
