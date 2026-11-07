# Hướng dẫn sử dụng và bàn giao `fscad`

Tài liệu này dành cho người nhận repo: cài đặt, chuẩn bị dữ liệu, train và đánh giá, tái lập bảng và
hình của báo cáo, lấy checkpoint và chạy inference. Các lệnh dưới đây là đúng các lệnh đã dùng để
sinh số liệu trong báo cáo. Tổng quan kiến trúc và chi tiết kỹ thuật bằng tiếng Anh nằm ở `README.md`.

## Mục lục

1. [Repo gồm những gì](#1-repo-gồm-những-gì)
2. [Cài đặt](#2-cài-đặt)
3. [Dữ liệu và split](#3-dữ-liệu-và-split)
4. [Train và đánh giá](#4-train-và-đánh-giá)
5. [Tái lập bảng và hình của báo cáo](#5-tái-lập-bảng-và-hình-của-báo-cáo)
6. [Checkpoint](#6-checkpoint)
7. [Inference](#7-inference)
8. [Kiểm thử](#8-kiểm-thử)
9. [Lưu ý quan trọng](#9-lưu-ý-quan-trọng)

## 1. Repo gồm những gì

| Thư mục / file | Nội dung |
|---|---|
| `configs/` | Cấu hình Hydra: `dataset/`, `model/`, `lora/`, `train/`, `method/` (`knn` hoặc `lora`), `serve.yaml` |
| `src/fscad/` | Thư viện chính: dữ liệu, backbone DINOv2, LoRA, bộ nhớ kNN, router, loss, đánh giá, dịch vụ inference |
| `src/fscad/evaluate.py` | Train (nếu `method=lora`) + đánh giá đơn tác vụ, lưu checkpoint, ghi MLflow |
| `src/fscad/train.py` | Chỉ train và lưu checkpoint, không đánh giá |
| `src/fscad/continual.py` | Thí nghiệm học liên tục với router (H2) |
| `src/fscad/drift_eval.py` | Thí nghiệm drift ánh sáng tổng hợp (H3) |
| `serving/app.py` | API FastAPI: `/predict`, `/onboard`, `/products`, `/drift`, `/metrics` |
| `scripts/` | Tạo split, xuất checkpoint, inference từ dòng lệnh, sinh bảng và hình báo cáo, baseline |
| `splits/visa/` | Danh sách ảnh shot và ảnh validation cố định cho mỗi (lớp, số shot, seed) |
| `tests/` | Test đơn vị và đầu-cuối trên dữ liệu toy, không cần tải dữ liệu |

Hai phương pháp được so sánh:

* `method=knn`: DINOv2 đóng băng + bộ nhớ patch + kNN, tức bản cài lại AnomalyDINO. Không train.
* `method=lora`: như trên nhưng có LoRA adapter ($r=8$, trên $W_Q, W_V$ của 4 khối cuối) train từ các
  ảnh shot và lỗi tổng hợp. Cấu hình mặc định: lr `1e-4`, 200 bước, batch 8.

## 2. Cài đặt

Cần Python 3.10 trở lên. Đã chạy trên macOS (Apple M4, backend MPS); trên Linux có GPU thì cài bản
PyTorch CUDA tương ứng.

```bash
git clone --recurse-submodules https://github.com/ThanhKieuVT/Module_05_Anomaly_Detection.git fscad
cd fscad
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # chọn bản CUDA nếu có GPU NVIDIA
pip install -e ".[mlops,serve,dev]"
pytest -q                              # khoảng 30 giây, không cần tải dữ liệu
```

Trọng số DINOv2 (`vit_small_patch14_dinov2.lvd142m`) được `timm` tự tải từ Hugging Face ở lần chạy
đầu tiên.

Thiết bị chọn tự động (`device=auto`: CUDA, rồi MPS, rồi CPU). Có thể ép bằng `device=cpu`.

## 3. Dữ liệu và split

### VisA (đã dùng trong báo cáo)

Tải bản chính thức của VisA (Amazon, repo `amazon-science/spot-diff`) và giải nén sao cho có:

```
data/visa/<lớp>/Data/Images/{Normal,Anomaly}/*.JPG
data/visa/<lớp>/Data/Masks/Anomaly/*.png
data/visa/split_csv/1cls.csv
```

12 lớp: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4,
pipe_fryum.

### MVTec AD, MVTec AD 2

Cần đăng ký trên trang MVTec rồi mới tải được. Đặt vào `data/mvtec_ad/<lớp>/...` và
`data/mvtec_ad2/<lớp>/...`. Chưa dùng trong báo cáo hiện tại.

### Split few-shot

Split của VisA đã có sẵn trong `splits/visa/<lớp>/k<K>_s<seed>.json` (K = 1, 4, 16; seed = 0, 1, 2).
Mỗi file ghi rõ đường dẫn các ảnh shot (`train`) và 30 ảnh validation bình thường (`val`, chỉ dùng để
đặt ngưỡng). Mọi phương pháp dùng đúng các file này, nên kết quả so sánh được với nhau.

Sinh lại (kết quả giống hệt vì thứ tự ảnh được xáo bằng seed cố định):

```bash
python scripts/make_splits.py --dataset visa --shots 1 4 16 --seeds 0 1 2
```

## 4. Train và đánh giá

Một lệnh vừa train (với LoRA), vừa đánh giá trên tập test, vừa lưu checkpoint và ghi kết quả vào
MLflow:

```bash
# baseline kNN (không train), 12 lớp VisA, 4-shot, 3 seed
python -m fscad.evaluate phase=baseline method=knn dataset=visa shots=4 'seeds=[0,1,2]' tracking.final=true

# LoRA, cấu hình mặc định
python -m fscad.evaluate phase=single_task method=lora dataset=visa shots=4 'seeds=[0,1,2]' tracking.final=true

# chỉ vài lớp, chạy thử nhanh (không ghi là kết quả chính thức)
python -m fscad.evaluate method=lora dataset=visa shots=4 'seeds=[0]' 'classes=[pcb1,candle]' eval.max_test_images=20
```

Đổi số shot bằng `shots=1` hoặc `shots=16`. Trong zsh phải đặt các tham số có dấu ngoặc vuông trong
nháy đơn như trên.

Thời gian tham khảo trên Apple M4 (12 lớp, 1 seed, 4-shot): kNN khoảng 10 phút, LoRA khoảng 1 giờ
(khoảng 3,5 phút train mỗi lớp). Với 16-shot, bộ nhớ và thời gian chấm điểm tăng khoảng 4 lần.

### Kết quả nằm ở đâu

| Nơi | Nội dung |
|---|---|
| MLflow, `mlflow.db` | Mỗi (lớp, seed) là một run: toàn bộ cấu hình, I-AUROC, AU-PRO, P-AUROC, F1-max, recall/FPR tại ngưỡng, độ trễ, bộ nhớ, thời gian train; tag `git_commit`, `final`, `checkpoint`, `checkpoint_sha256` |
| `outputs/<ngày>/<giờ>_evaluate/results.csv` | Bảng kết quả của lần chạy |
| `artifacts/<run_name>/<fingerprint>/<dataset>/<lớp>/k<K>_s<seed>.pt` | Checkpoint (xem mục 6) |

Xem MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`, mở `http://localhost:5000`.

### Các tham số hay dùng

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `train.lr`, `train.steps` | learning rate, số bước train LoRA | `1e-4`, `200` |
| `train.synthetic` | lỗi tổng hợp: `perlin`, `cutpaste`, `mixed` | `mixed` |
| `train.use_synthetic` | `false` = chỉ loss compactness | `true` |
| `lora.r`, `lora.blocks` | hạng LoRA, số khối cuối có LoRA | `8`, `4` |
| `model` | `dinov2_vits14` hoặc `dinov2_vitb14` | `dinov2_vits14` |
| `model.img_size` | độ phân giải vào | `448` |
| `memory.augment` | các view thêm vào bộ nhớ | `[hflip, vflip]` |
| `eval.masking` | che nền: `auto` (theo AnomalyDINO), `true`, `false` | `auto` |
| `eval.threshold_quantile` | ngưỡng $\delta$ = phân vị này của điểm validation | `0.99` |

### Quy ước kết quả chính thức

* `tracking.final=true` chỉ chạy khi git sạch (mọi thay đổi code đã commit), để mỗi kết quả gắn được
  với đúng một commit. Nếu báo `refusing to run with uncommitted changes`, hãy commit trước.
* Báo cáo chỉ dùng các run có tag `final=true`.

## 5. Tái lập bảng và hình của báo cáo

```bash
# bảng và hình kết quả (đọc thẳng từ MLflow, chỉ run final=true)
python scripts/make_report_results.py --out ../report

# hình minh hoạ trên ảnh VisA thật (luồng suy luận, router, patch token, lỗi tổng hợp, so sánh kNN/LoRA)
python scripts/make_report_assets.py --out ../report/figures
```

`make_report_results.py` chọn cấu hình theo tham số đã ghi trong MLflow và chỉ so các cặp (lớp, seed)
mà cả kNN và LoRA đều đã chạy xong. Mọi con số nằm trong `results.json` đi kèm.

## 6. Checkpoint

### Một checkpoint chứa gì

Mỗi file `.pt` là một sản phẩm đã đăng ký (`fscad.product.ProductModel`):

* trọng số LoRA (nếu `method=lora`, khoảng 0,2 MB),
* bộ nhớ patch của các ảnh shot (khoảng 18 MB ở 4-shot),
* token CLS của shot và của ảnh validation (cho router và giám sát drift),
* ngưỡng $\delta$ và điểm validation,
* `meta`: số shot, seed, che nền, cấu hình đầy đủ (`config`), mã cấu hình (`fingerprint`), commit git,
  id của run MLflow, và các thước đo đánh giá (`metrics`).

Đường dẫn chứa `fingerprint` (8 ký tự đầu của mã băm cấu hình), nên hai cấu hình khác nhau (ví dụ hai
learning rate) không bao giờ ghi đè checkpoint của nhau.

### Tải bộ checkpoint đã huấn luyện (không cần train lại)

Checkpoint của các run chính thức trên VisA đã được xuất sẵn và lưu trên Google Drive:
<https://drive.google.com/drive/folders/1ajZUBnxbVLf8y5SgEQj1ndRs8cVp-QYG>
(người nhận cần được chia sẻ quyền xem thư mục này).

* `visa-<knn|lora>-k<1|4|16>-s<0|1|2>/`: các bundle, cấu trúc như mục dưới; `manifest.json` liệt kê
  từng file kèm sha256, run MLflow và thước đo.
* `_ablation_raw/`: checkpoint gốc của các thử nghiệm chọn cấu hình trên pcb (bảng (A)–(E) của báo cáo),
  giữ nguyên đường dẫn trong `artifacts/`. Không dùng cho inference.

Tải về thư mục `checkpoints/` của repo (thư mục này đã nằm trong `.gitignore`): tải thư mục trên giao
diện web, hoặc bằng `rclone` (sau khi `rclone config` tạo remote Google Drive tên `gdrive`):

```bash
rclone copy gdrive: checkpoints --drive-root-folder-id 1ajZUBnxbVLf8y5SgEQj1ndRs8cVp-QYG \
    --exclude "_ablation_raw/**"
```

`make_report_assets.py` đọc hình so sánh kNN/LoRA trên macaroni2 từ `checkpoints/` (đổi bằng
`--checkpoints`).

### Xuất bộ checkpoint chính thức

```bash
python scripts/export_checkpoints.py dataset=visa method=lora shots=4 'seeds=[0,1,2]' +export.out=checkpoints
python scripts/export_checkpoints.py dataset=visa method=knn  shots=4 'seeds=[0,1,2]' +export.out=checkpoints
```

Script chỉ lấy các run `final=true` có cấu hình đúng bằng cấu hình đang chọn, xác định đúng file của
từng run (theo tag `checkpoint` và sha256, hoặc với run cũ theo thời điểm file được ghi trong lúc run
chạy), kiểm tra file đúng lớp, số shot, seed, rồi ghi:

```
checkpoints/
  manifest.json                         mọi file: sản phẩm, seed, sha256, run MLflow, commit, I-AUROC, AU-PRO, ngưỡng
  visa-lora-k4-s0/visa__candle/model.pt
  visa-lora-k4-s0/visa__candle/meta.json
  ...
```

Mỗi thư mục `visa-lora-k4-s<seed>/` là một "bundle" gồm 12 sản phẩm, dùng thẳng được cho inference.

Kiểm tra toàn vẹn sau khi tải về: so `sha256sum <file>` với trường `sha256` trong `manifest.json`.
Đã kiểm chứng: chấm lại tập test bằng checkpoint đã xuất cho ra đúng I-AUROC và AU-PRO đã ghi trong
MLflow.

## 7. Inference

### Từ dòng lệnh, không cần chạy server

```bash
# router tự chọn sản phẩm cho từng ảnh (có thể truyền file hoặc thư mục)
python scripts/predict.py --products checkpoints/visa-lora-k4-s0 --out outputs/predict \
    data/visa/pcb1/Data/Images/Anomaly/ data/visa/candle/Data/Images/Normal/0790.JPG

# đã biết sản phẩm: bỏ qua router
python scripts/predict.py --products checkpoints/visa-lora-k4-s0 --product visa/pcb1 anh_moi.png
```

Kết quả trong `--out`:

* `predictions.csv`: mỗi ảnh một dòng gồm sản phẩm được chọn, sản phẩm gần nhất và độ tương đồng
  cos, có bị từ chối không, điểm bất thường $S$, ngưỡng $\delta$, và quyết định `ok`, `anomaly` hoặc
  `rejected`;
* `NNNN_<tên ảnh>.png`: ảnh và bản đồ lỗi chồng lên (tắt bằng `--no-maps`);
* `summary.json`: số ảnh theo từng quyết định.

`rejected` nghĩa là ảnh không giống sản phẩm nào đã đăng ký (cos với mọi prototype dưới ngưỡng
$\tau$). Trong vận hành, ảnh bị từ chối phải được chuyển cho người kiểm tra, không được cho qua.

Script đọc kích thước backbone, LoRA và cách chấm điểm từ chính checkpoint, nên luôn khớp với lúc
train; điểm cho ra giống hệt lúc đánh giá (có test kiểm tra điều này).

### Qua API

```bash
FSCAD_PRODUCTS_DIR=checkpoints/visa-lora-k4-s0 uvicorn serving.app:app --port 8000
# tài liệu API: http://localhost:8000/docs

curl -F file=@data/visa/pcb1/Data/Images/Anomaly/000.JPG "http://localhost:8000/predict?return_map=true"

# đăng ký sản phẩm mới từ vài ảnh bình thường (masking=true cho vật thể đặt trên nền)
curl -F name=line3/connector -F masking=true \
     -F files=@good1.png -F files=@good2.png -F files=@good3.png -F files=@good4.png \
     http://localhost:8000/onboard
```

API dùng backbone và LoRA theo `configs/serve.yaml` (ViT-S/14, $r=8$, 4 khối), khớp với checkpoint
mặc định. Với checkpoint có cấu hình khác, truyền thêm, ví dụ
`FSCAD_OVERRIDES="model=dinov2_vitb14"`; `scripts/predict.py` thì tự đọc cấu hình từ checkpoint.

Hoặc chạy cả hệ thống (API, MLflow, Prometheus, Grafana) bằng `docker compose up --build`; xem
`README.md`, mục "Serving and monitoring".

## 8. Kiểm thử

```bash
pytest -q          # 27 test: LoRA, kNN, router, loss, split, lỗi tổng hợp, che nền, checkpoint, dịch vụ, API
ruff check . && black --check .
```

CI trên GitHub chạy lint, test và một lần chạy thử đầu-cuối trên dữ liệu toy cho mỗi commit.

## 9. Lưu ý quan trọng

* **Tính tái lập.** kNN cho kết quả giống hệt khi chạy lại. LoRA trên MPS không hoàn toàn tất định
  (dao động khoảng 1 điểm I-AUROC khi train lại cùng seed); checkpoint đã xuất thì cho kết quả cố định.
* **Chọn cấu hình.** Cấu hình LoRA mặc định (lr `1e-4`) được chọn trên pcb1–pcb4 của VisA; kết luận
  trong báo cáo dựa trên 8 lớp còn lại. Khi chạy dữ liệu mới, giữ nguyên cấu hình, không chỉnh theo
  tập test.
* **Ngưỡng $\delta$.** Đặt từ 30 ảnh validation nên FPR thực tế khoảng 4% thay vì 1% như mục tiêu.
  Muốn dùng trong vận hành nên đăng ký sản phẩm với nhiều ảnh bình thường hơn cho validation.
* **macaroni2.** Lớp duy nhất LoRA kém kNN rõ rệt (vùng bóng giữa các vật thể nằm sát nhau); xem
  báo cáo, mục phân tích lỗi.
* **Che nền.** Mặt nạ luôn tính trên token DINOv2 đóng băng, không trên token đã qua LoRA.
