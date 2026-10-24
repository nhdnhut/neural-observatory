"""Generate a measured Vietnamese report and guided notebooks from saved results."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifacts/experiment.json').read_text())
groups=json.loads((ROOT/'artifacts/aggregate.json').read_text())
robust=json.loads((ROOT/'artifacts/robustness.json').read_text())
main=next(r for r in manifest['results'] if r['model']=='observatory_64_32' and r['seed']==42 and r['variant']=='fp32')
fp=next(g for g in groups if g['model']=='observatory_64_32' and g['variant']=='fp32')
q=next(g for g in groups if g['model']=='observatory_64_32' and g['variant']=='int8')
labels={'observatory_64_32':'784–64–32–10 / ReLU','baseline_20_sigmoid':'784–20–10 / Sigmoid','compact_20_relu':'784–20–10 / ReLU','budget_6_relu':'784–6–10 / ReLU'}
rows=[]
for g in groups:
 rows.append(f"| {labels[g['model']]} | {g['variant'].upper()} | {100*g['accuracy_mean']:.3f} ± {100*g['accuracy_sd']:.3f}% | {g['model_bytes']:,} B | {g['latency_p50_ms']:.3f} ms |")
table='\n'.join(['| Kiến trúc | Trọng số | Test accuracy: mean ± SD | Tệp mô hình | Packed Python p50 |','|---|---|---:|---:|---:|',*rows])
report=f'''# Neural Observatory — Báo cáo kết quả thực nghiệm

**Tên đề tài:** Xây dựng từ đầu, quan sát và nén mạng neural nhận dạng chữ số dưới ngân sách bộ nhớ.

**Nguyễn Huỳnh Đăng Nhựt — 26521522.** Phát triển từ Neural-Network-from-Scratch, commit `c3ce89b4cc22e034611287e30af1b5dda6d17bbe`.

## Kết quả chính

Mạng 784–64–32–10 FP32 đạt **{100*fp['accuracy_mean']:.3f}% ± {100*fp['accuracy_sd']:.3f} điểm phần trăm** trên test MNIST qua ba seed. Trọng số INT8 đạt **{100*q['accuracy_mean']:.3f}% ± {100*q['accuracy_sd']:.3f} điểm phần trăm**. Tệp giảm từ **{fp['model_bytes']:,} byte** xuống **{q['model_bytes']:,} byte**, tương đương **{fp['model_bytes']/q['model_bytes']:.3f} lần** hoặc giảm **{100*(1-q['model_bytes']/fp['model_bytes']):.2f}%**.

Chênh lệch accuracy trung bình chỉ {100*(q['accuracy_mean']-fp['accuracy_mean']):+.4f} điểm phần trăm; không có cơ sở để tuyên bố nén làm tăng accuracy. Với seed 42, accuracy giảm nhẹ 97,72% → 97,70%. Phải xem cả ba seed và các ảnh bị đổi dự đoán.

## 1. Bài toán và đóng góp

Bài toán là xây dựng một mạng có thể huấn luyện, quan sát, can thiệp và nén, rồi đo kết quả theo ngân sách tệp mô hình. Đóng góp nằm ở implementation đúng, khả năng kiểm chứng và toàn bộ quy trình đo; không tuyên bố thuật toán mới hoặc SOTA.

Công cụ cho thấy từng lớp thật: ảnh 784 chiều, lớp 64 neuron, lớp 32 neuron, logits và softmax 10 lớp. Khi thay ảnh, checkpoint, precision hoặc tắt neuron, trình duyệt tính lại toàn bộ forward. Những checkpoint được ghi ở epoch thật; không mô phỏng việc training bằng animation.

## 2. Thiết kế thí nghiệm

- Dữ liệu chính thức: 60.000 training và 10.000 test. Tách 500 ảnh mỗi lớp làm validation bằng seed 2026, còn 55.000 train. Tập test không dùng chọn checkpoint.
- Kiểm tra SHA-256 bốn tệp gốc; chỉ chuẩn hóa pixel bằng 255.
- 4 cấu hình × 3 seed (42,43,44), mỗi cấu hình train 18 epoch, batch 128. Tổng cộng 12 lượt train; mỗi checkpoint được xuất FP32/INT8, thành 24 mô hình.
- SGD momentum 0,9 tự viết. LR ban đầu: 0,04 với ReLU, 0,12 với đối chứng sigmoid; cosine decay đến 10%.
- Chọn checkpoint theo validation cross-entropy nhỏ nhất. Các lựa chọn cấu hình được cố định trước khi đánh giá test.
- Đường train mỗi epoch dùng tập con cố định 5.000 ảnh để giảm chi phí log; metric train cuối dùng đủ 55.000 ảnh.
- Mạng seed 42 hiển thị trong explorer dùng epoch **{main['best_epoch']}**, có **{main['parameters']:,}** tham số.

## 3. Kết quả đầy đủ

{table}

SD là độ lệch chuẩn mẫu của ba seed. Đây không phải khoảng tin cậy cho mọi loại chữ viết. Latency trong bảng là trung vị của ba p50 tương ứng ba seed. Dữ liệu từng lần đo nằm trong `experiment.json`; số liệu mỗi run nằm trong `benchmark.csv`.

Đối chứng 20 sigmoid kế thừa kích thước và hidden activation của repo, nhưng dùng bộ train/softmax/loss đúng mới. Vì các cấu hình khác nhau cả kiến trúc, activation hoặc LR, không quy toàn bộ chênh lệch accuracy cho riêng độ sâu.

## 4. Kết quả theo ngân sách

Ở ngân sách 64 KiB, mạng 64–32 INT8 vừa ngân sách ({q['model_bytes']/1024:.2f} KiB), trong khi bản FP32 của nó vượt ngân sách ({fp['model_bytes']/1024:.2f} KiB). Có thể so với mạng 20 neuron FP32 cũng vừa ngân sách. Giao diện chọn ứng viên theo **accuracy validation trung bình**, không dùng test để chọn.

Ngân sách đang nói về **tệp mô hình**, không phải tổng RAM. `packed_weight_buffer_bytes` chỉ đo bộ đệm trọng số, không gồm Python objects, activation, allocator hoặc RAM tiến trình.

## 5. INT8: đúng phạm vi tuyên bố

Quantizer đối xứng theo từng output channel: scale = max(abs(weights))/127, round rồi clip về [-127,127]. Trọng số được đóng gói thành signed byte; scale và bias float32. Hàng toàn 0 dùng scale 1. Bộ runtime Python tính tích với activation số thực và nhân scale mỗi hàng, không dựng lại toàn bộ ma trận FP32.

Đây là **weight-only quantization**, không phải integer-only inference. Trong môi trường đo, p50 chính là {fp['latency_p50_ms']:.3f} ms (FP32) và {q['latency_p50_ms']:.3f} ms (INT8). INT8 **chậm hơn** ở runtime Python này. Lợi ích được chứng minh là dung lượng, không phải tăng tốc hoặc giảm điện năng.

## 6. Quan sát và can thiệp

- Weight map lớp 1: xếp 784 trọng số theo vị trí pixel, màu đối xứng âm/dương. Không gọi nó là khái niệm neuron đã hiểu.
- Activation maps: hiện giá trị qua từng lớp cho ảnh/checkpoint đang chọn. Mỗi lớp có thang riêng.
- Ablation: đặt activation của neuron được chọn bằng 0 rồi chạy phần mạng phía sau. Có thể tắt nhiều neuron; thao tác khôi phục trả về forward ban đầu.
- Xếp hạng tác động: ablate từng neuron riêng trên 1.024 ảnh validation. Đây là can thiệp trong mô hình, phụ thuộc dữ liệu và tương tác neuron.
- Saliency: trị tuyệt đối đạo hàm của logit lớp dự đoán theo pixel. Là độ nhạy cục bộ, không phải chứng minh ngữ nghĩa.
- Ảnh minh hoạ cố ý có cả đúng, sai và đổi dự đoán sau nén; không dùng tỉ lệ của nhóm minh hoạ để suy ra accuracy.

## 7. Một giới hạn thực nghiệm rõ ràng

Với seed 42, ảnh test gốc đạt **{100*robust['clean']['accuracy']:.2f}%**. Chỉ dịch 2 pixel sang phải và điền 0 đã làm accuracy còn **{100*robust['shifted']['accuracy']:.2f}%**. Đây là dấu hiệu nhạy với vị trí của MLP đã train không augmentation. Không dùng phép thử này để điều chỉnh mô hình trong nghiên cứu chính.

Nét vẽ bằng chuột/tay trong demo không cùng phân phối MNIST; xác suất softmax không bảo đảm độ tin cậy trên đầu vào đó.

## 8. Python thuần và quy định học phần

`python -S scripts/train_pure.py --samples 1024 --epochs 2` đã thực sự chạy, dùng thư viện chuẩn. Mạng 784–20–10 đạt 79,6875% trên 256 ảnh validation của bản nhỏ sau 2 epoch; không đánh giá test. Không so trực tiếp con số này với full benchmark vì dữ liệu, optimizer và số epoch khác.

Reference backprop, softmax, quantization, IDX và inference không có thư viện bên thứ ba. Full benchmark sử dụng NumPy cho phép toán mảng và phải được khai báo. Không có PyTorch, TensorFlow, sklearn hay autograd trong implementation mới. Nếu cấm NumPy toàn bộ, dùng đường Python thuần và tăng quy mô theo thời gian cho phép.

## 9. Kiểm tra và tái lập

Các kiểm tra gồm softmax ở logits lớn, mọi gradient của các mạng nhỏ bằng sai phân hữu hạn, đối chiếu forward/gradient reference–NumPy, bound sai số quantization, zero-row và lưu/nạp mô hình. Giao diện được kiểm tra đối chiếu xác suất JavaScript với dữ liệu Python và các thao tác chính. Xem `artifacts/browser_validation.json` để biết các phép kiểm tra đã chạy trên bản bàn giao.

Lệnh:

```bash
python main.py test
python -S scripts/train_pure.py --samples 1024 --epochs 2
python main.py train --epochs 18 --seeds 42 43 44
python main.py build
python -S scripts/infer.py --index 0
```

Môi trường: {manifest['cpu']}; Python {manifest['python']}; NumPy {manifest['numpy']}; BLAS 1 thread; {manifest['platform']}. Độ trễ không phải kết quả đo trên laptop của người dùng. Mốc UTC trong `protocol.json` là thời gian runtime thực thi, không phải hạn nộp đồ án.

## 10. Phát triển tiếp có kiểm chứng

1. Cài integer kernel thật bằng C/C++ rồi benchmark trên máy mục tiêu.
2. Thử augmentation dịch ảnh, với protocol mới khai báo trước và tập test giữ độc lập.
3. Pruning neuron có fine-tune, đối chứng cùng ngân sách; không chỉ gán trọng số bằng 0.
4. Sau khi kiểm soát được nền tảng, mở nhánh mô hình ngôn ngữ nhỏ. Chưa gọi phiên bản hiện tại là decoding LLM.

## Nguồn

- Repo gốc: https://github.com/DangNhutNguyen/Neural-Network-from-Scratch
- MNIST mirror/IDX: https://github.com/cvdfoundation/mnist
- Nền tảng quantization: https://arxiv.org/abs/1712.05877
'''
(ROOT/'docs/EXPERIMENT_REPORT.md').write_text(report)
readme=f'''# Neural Observatory

**Xây dựng từ đầu, quan sát và nén mạng neural nhận dạng chữ số dưới ngân sách bộ nhớ.**

Phát triển trực tiếp từ dự án của **Nguyễn Huỳnh Đăng Nhựt — 26521522**. Mã gốc giữ trong `legacy/`, xuất phát từ commit `c3ce89b4cc22e034611287e30af1b5dda6d17bbe`. Bản mở rộng có hỗ trợ AI; người học cần hiểu, kiểm tra và giải thích được phần cài đặt khi bảo vệ.

Hình báo cáo được tạo tại `artifacts/figures/experiment_overview.png` và có sẵn trong gói bàn giao đầy đủ.

## Mở ngay

Mở **`artifacts/Neural_Observatory.html`** bằng Chrome/Edge/Firefox. File tự chứa trọng số, ảnh và số đo; không cần Internet, server hay cài thư viện. Có 4 tab:

1. **Khám phá lớp mạng:** chọn ảnh test, replay checkpoint, xem activation, chọn/tắt neuron, xem weights/saliency, thử nét vẽ mới và đổi FP32/INT8.
2. **Quá trình học:** loss, accuracy, activation zeros và gradient norm thật.
3. **Ngân sách:** chọn giới hạn KiB và so sánh 8 cấu hình/định dạng qua 3 seed.
4. **Kiểm chứng:** confusion matrix, ablation, dịch ảnh, nguồn gốc và giới hạn.

## Kết quả đã chạy

{table}

MNIST: **55.000 train / 5.000 validation / 10.000 test**. 12 lượt train, 18 epoch/lượt, seed 42/43/44. Điểm hiển thị là test độc lập. INT8 là **weight-only**, không phải integer-only; packed Python inference chậm hơn FP32 trong phép đo này. Không tuyên bố giảm RAM toàn tiến trình hoặc tăng tốc.

## Chạy lại — Python 3.11+

```bash
python -m pip install -r requirements.txt
python main.py test
python main.py train --epochs 18 --seeds 42 43 44
python main.py build
```

Lệnh train tải MNIST với SHA-256 rồi huấn luyện. `--resume` dùng lại các run đã xong cùng số epoch; bỏ `--resume` để train lại. `build` cần kết quả full suite và kết quả Python thuần:

```bash
python -S scripts/train_pure.py --samples 1024 --epochs 2
python -S scripts/infer.py --index 0
```

`-S` bỏ site-packages: đường Python thuần không cần NumPy. Để train nhiều ảnh hơn, đổi `--samples` (tối đa 55.000), `--epochs`, `--width`; thời gian sẽ tăng. Kết quả 1.024 ảnh là bản chứng minh thuật toán, tách khỏi full benchmark.

## Phần nào dùng thư viện?

| Thành phần | Dependency |
|---|---|
| Reference MLP, SGD, IDX, quantization, packing, inference | Python chuẩn |
| Full training/evaluation | NumPy, không autograd/framework |
| Hình báo cáo | Matplotlib |
| Giao diện và browser inference | JavaScript/Canvas thuần |

NumPy kế thừa hướng repo gốc. Nếu học phần cấm mọi thư viện ngoài, dùng pure-Python path; không trình bày full NumPy benchmark như kết quả không thư viện. GPU T4 không tự tăng tốc backend CPU này.

## Tài liệu

- [Định nghĩa bài toán](docs/PROJECT_DEFINITION.md)
- [Báo cáo số liệu thật](docs/EXPERIMENT_REPORT.md)
- [Kiến trúc và định dạng mô hình](docs/ARCHITECTURE.md)
- [Kịch bản demo 6 phút](docs/DEMO_SCRIPT.md)
- `notebooks/01_Baseline_and_Gradient_Check.ipynb`
- `notebooks/02_Training_and_Quantization.ipynb`
- `notebooks/03_Layers_and_Inspection.ipynb`

## Dữ liệu và kết quả

`artifacts/experiment.json` chứa lịch sử và các số đo latency riêng lẻ; `benchmark.csv` chứa 24 run/variant; `aggregate.json` chứa mean/SD; `models/` chứa 24 mô hình FP32/INT8; `checkpoints/` chứa lịch sử và trọng số theo epoch. `protocol.json` ghi môi trường, dataset hashes và split hashes.

`dashboard/data.js` được tạo từ checkpoint thật. `dashboard/index.html` là source; `artifacts/Neural_Observatory.html` là bản offline đóng gói. Mở bằng file hoặc `python -m http.server 8000` ở thư mục gốc.

Đối chứng 20 sigmoid đã sửa loss/train protocol; không phải kết quả nguyên trạng script cũ. Architecture comparisons thay đổi cả depth/width/activation/LR, nên không tách riêng tác động của depth. Ba seed không phải confidence interval của toàn bộ quần thể chữ viết.

## Tiến độ phiên bản

Đã có training, kiểm tra số, quantizer, browser inference, can thiệp neuron, visualizations và dữ liệu đo. Chưa có LLM/decoding, integer-only kernel, đo điện năng, pruning-retrain hoặc deployment trên điện thoại.

MIT License giữ từ dự án gốc. MNIST được tải riêng từ mirror được ghi trong source.
'''
(ROOT/'README.md').write_text(readme)

def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
def code(s):return {'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]}
def save_nb(name,cells):
 obj={'nbformat':4,'nbformat_minor':4,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'cells':cells}
 (ROOT/'notebooks'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
setup='''from pathlib import Path
import os, sys, subprocess
# Local: launch notebook from the extracted project or its notebooks/ folder.
# Colab: extract Neural_Observatory.zip into /content first (see instructions).
candidates = [Path.cwd(), Path.cwd().parent, Path('/content/Neural_Observatory')]
project = next((p for p in candidates if (p/'observatory/engine.py').exists()), None)
assert project is not None, 'Extract the full project ZIP first and set project to its root.'
os.chdir(project)
sys.path.insert(0, str(project))
print('Project:', project)
'''
colab='''## Mở trên Colab hoặc Jupyter

Đây là notebook hướng dẫn tái lập, các cell chưa được đánh dấu là đã thực thi. Kết quả huấn luyện thật có sẵn trong `artifacts/`.

**Colab:** tải ZIP đầy đủ lên Files, giải nén vào `/content/` để có `/content/Neural_Observatory`. Có thể dùng `import zipfile; zipfile.ZipFile('/content/Neural_Observatory.zip').extractall('/content')` với ZIP được bàn giao. Chọn CPU; bật T4 không tăng tốc NumPy tự động. Nếu cần dependency, chạy `%pip install -r /content/Neural_Observatory/requirements.txt`.

**Jupyter:** mở notebook từ `notebooks/` sau khi cài `requirements.txt`. Chạy các cell từ trên xuống.
'''
save_nb('01_Baseline_and_Gradient_Check.ipynb',[
 md('# 01 · Baseline, toán học và kiểm tra gradient\n\nTừ repo của Nhựt đến một baseline có thể kiểm chứng.'),md(colab),code(setup),
 md('## Vì sao cần sửa?\n\nRepo gốc cập nhật lớp ra trước khi tính gradient lớp ẩn và chưa giữ tập test độc lập. Bản mới tính mọi gradient rồi mới cập nhật. Forward giữ quy ước W[out,in]. Loss cuối là cross-entropy với softmax ổn định.'),
 code("from observatory.reference import ReferenceMLP\nm = ReferenceMLP((3, 4, 2), 'sigmoid', 42)\nx, y = [0.3, -0.7, 0.4], 1\nloss, dw, db = m.loss_and_grad(x, y)\nprint('Loss:', loss, 'probabilities:', m.forward(x)[0][-1])"),
 md('## Đạo hàm số\n\nThay một trọng số bởi w+ε và w−ε; độ dốc xấp xỉ là (L(w+ε)−L(w−ε))/(2ε). So sánh với backprop. Mạng nhỏ dùng double precision để giảm sai số số học.'),
 code("eps = 1e-5\nold = m.weights[0][0][0]\nm.weights[0][0][0] = old + eps\nplus = m.loss_and_grad(x,y)[0]\nm.weights[0][0][0] = old - eps\nminus = m.loss_and_grad(x,y)[0]\nm.weights[0][0][0] = old\nprint('Numeric:', (plus-minus)/(2*eps), 'Backprop:', dw[0][0][0])\nsubprocess.run([sys.executable,'main.py','test'],check=True)"),
 md('## Chạy thật bằng Python chuẩn\n\nLệnh -S bỏ site-packages. 1.024 ảnh train, 256 validation, 2 epoch: bản chứng minh end-to-end riêng, không phải benchmark đầy đủ.'),
 code("subprocess.run([sys.executable,'-S','scripts/train_pure.py','--samples','1024','--epochs','2'],check=True)"),
 md('## Tự kiểm tra hiểu bài\n\n1. Vì sao không cập nhật W trước khi tính hết gradient?\n2. Vì sao softmax trừ max(logits)?\n3. Vì sao accuracy trên dữ liệu đang train không phải test accuracy?')])
save_nb('02_Training_and_Quantization.ipynb',[
 md('# 02 · Train, đối chứng và lượng tử hóa\n\n4 kiến trúc × 3 seed; dữ liệu test giữ độc lập.'),md(colab),code(setup),
 md('## Protocol\n\n55.000 train, 5.000 validation, 10.000 test. SGD momentum 0,9; batch 128; 18 epoch. Chọn checkpoint bằng validation loss. Đọc protocol trước khi xem test. Full training dùng NumPy CPU, không autograd.'),
 code("import json\nprotocol=json.loads(Path('artifacts/protocol.json').read_text())\nprotocol['protocol']"),
 code("RETRAIN = False  # Change to True to rerun every model.\nif RETRAIN:\n    subprocess.run([sys.executable,'main.py','train','--epochs','18','--seeds','42','43','44'],check=True)\nelse:\n    print('Using the completed measured experiment provided in artifacts/.')"),
 md('## Quantization tự viết\n\nMột scale mỗi output channel. Trọng số chuyển sang signed byte; bias/scale float32. Đây là weight-only quantization. Đừng suy ra tốc độ hay RAM toàn chương trình từ số bit.'),
 code("from observatory.packing import quantize_row, load_packed, packed_forward\nweights=[-0.91,-0.2,0.,0.11,0.72]\nq,s=quantize_row(weights)\nprint('scale=',s,'q=',q,'reconstructed=',[v*s for v in q])"),
 md('## Kết quả 24 bản\n\nMean ± SD là biến thiên ba seed. Hãy tìm ảnh đổi dự đoán và trường hợp INT8 chậm hơn.'),
 code("import csv\nrows=list(csv.DictReader(open('artifacts/benchmark.csv')))\nfor row in rows:\n    print(row['model'],row['seed'],row['variant'],row['test_accuracy'],row['model_bytes'],row['latency_p50_ms'])"),
 md('## Câu hỏi nghiên cứu\n\nVới ngân sách 64 KiB, mạng 64–32 INT8 hay mạng 20 FP32 đạt điểm validation tốt hơn? Tại sao giảm dung lượng không đủ để khẳng định tăng tốc?')])
save_nb('03_Layers_and_Inspection.ipynb',[
 md('# 03 · Xem lớp mạng và can thiệp\n\nĐọc trạng thái thật từ checkpoint; không dùng animation mô phỏng việc học.'),md(colab),code(setup),
 code("from observatory.dataset import load_numpy\nfrom observatory.engine import MLP\nx,y,xv,yv,xt,yt,_,_=load_numpy()\nm=MLP.load('artifacts/checkpoints/observatory_64_32_s42/best.npz')\na,z=m.forward(xt[:1])\nfor i,v in enumerate(a):\n    print('Layer',i,'shape',v.shape,'range',float(v.min()),float(v.max()))\nprint('Truth:',yt[0],'Prediction:',a[-1].argmax())"),
 md('## Activation và weight maps\n\nWeight map của lớp đầu có cấu trúc không gian của ảnh; lưới của lớp sâu hơn chỉ là cách sắp neuron. Activation = 0 ở một ảnh chưa phải neuron chết.'),
 code("import matplotlib.pyplot as plt\nfig,axs=plt.subplots(1,4,figsize=(12,3))\naxs[0].imshow(xt[0].reshape(28,28),cmap='gray');axs[0].set_title('Input')\naxs[1].imshow(a[1].reshape(8,8),cmap='Purples');axs[1].set_title('Hidden 1: 64')\naxs[2].imshow(a[2].reshape(4,8),cmap='Purples');axs[2].set_title('Hidden 2: 32')\naxs[3].bar(range(10),a[-1][0]);axs[3].set_title('Softmax')\nplt.tight_layout();plt.show()"),
 md('## Can thiệp neuron\n\nĐặt activation một neuron bằng 0, giữ các trọng số còn lại. Xem thay đổi trên cùng ảnh, rồi trên validation. Không gán ý nghĩa nhân quả thế giới thực cho thao tác này.'),
 code("before=m.forward(xt[:1])[0][-1][0]\nafter=m.forward(xt[:1],intervention=(1,11))[0][-1][0]\nprint('Before:',before,'After:',after)\nprint('Pred:',before.argmax(),'->',after.argmax())"),
 md('## Mở toàn bộ observatory\n\nFile HTML offline chứa trọng số thật. Nếu đã chạy lại thí nghiệm, build lại trước. Trên Colab có thể tải HTML về và mở bằng trình duyệt.'),
 code("REBUILD=False\nif REBUILD:\n    subprocess.run([sys.executable,'main.py','build'],check=True)\nprint(Path('artifacts/Neural_Observatory.html').resolve())"),
 md('## Điều cần báo cáo\n\nHãy mở ảnh sai, INT8 đổi dự đoán, ảnh bị dịch và latency chậm hơn. Không chỉ chọn những ảnh mạng làm đúng.')])
print('Wrote README, measured report and three guided notebooks.')
