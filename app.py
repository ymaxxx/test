import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import gradio as gr
import numpy as np
import io
import sys

#  --------- モデル定義 ----------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [batch, 32, 26, 26]
        x = F.relu(self.conv2(x))      # [batch, 64, 24, 24]
        x = F.max_pool2d(x, 2)         # [batch, 64, 12, 12]
        x = torch.flatten(x, 1)        # [batch, 9216]
        x = F.relu(self.fc1(x))        # [batch, 128]
        x = self.fc2(x)                # [batch, 10]
        return x

#  --------- モデル訓練 ----------
def train_model(model_path="model.pth", epochs=1, capture_log=False):
    """モデルを訓練し、ログをキャプチャすることも可能"""
    if capture_log:
        log_capture = io.StringIO()
        stdout_backup = sys.stdout
        sys.stdout = log_capture
    
    print(f"モデル訓練開始、エポック数：{epochs}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), model_path)
    print(f"モデル訓練完了し、{model_path} に保存しました。")
    
    if capture_log:
        sys.stdout = stdout_backup
        log_output = log_capture.getvalue()
        return model, log_output
    
    return model

# ---------- 模型加载 ----------
def load_model(model_path="model.pth"):
    model = CNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("既存のモデルを読み込みました。")
    else:
        model = train_model(model_path)
    model.eval()
    return model

model = load_model()

# ---------- 图像预处理 ----------
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_with_options(pil_img, invert=True, binarize=True, crop=True, scale20=True, center_mass=True):
    """preprocess_imageと同様だが、各ステップをオンオフ可能で、アブレーション実験に便利。
    (tensor, display_img)を返す
    ステップの意味：
      - invert: 背景を自動反転
      - binarize: 二値化（閾値分割）
      - crop: 前景のバウンディングボックスにトリミング
      - scale20: トリミング結果をアスペクト比を保って20x(<=20)にリサイズし、28x28に中央配置；そうでなければ直接28x28にリサイズ
      - center_mass: 重心を中央に平行移動
    """
    # グレースケール化 8-bit グレースケール画像
    img = pil_img.convert('L')
    # numpy 配列 uint8 型（符号なし 8 ビット整数）に変換
    arr = np.array(img).astype(np.uint8)

    # invert
    if invert and arr.mean() > 127:
        arr = 255 - arr

    # binarize
    if binarize:
        thresh = arr.mean()
        mask_pos = (arr > thresh)
        mask_neg = (arr < thresh)
        # choose the mask with smaller foreground area (assume digit occupies small area)
        if mask_pos.sum() <= mask_neg.sum():
            mask = mask_pos
        else:
            mask = mask_neg
        bw = (mask.astype(np.uint8) * 255)
    else:
        # if not binarizing, treat non-zero as foreground
        bw = (arr > 0).astype(np.uint8) * 255

    # crop
    if crop:
        coords = np.column_stack(np.where(bw > 0))
        if coords.size == 0:
            display = Image.fromarray(arr).resize((28, 28)).convert('L')
            tensor = transform_tensor(display).unsqueeze(0)
            return tensor, display
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # use the foreground mask values for the crop (foreground=255, background=0)
        crop_arr = bw[y0:y1+1, x0:x1+1]
    else:
        crop_arr = arr

    # scaling
    h, w = crop_arr.shape
    if scale20:
        if h > w:
            new_h = 20
            new_w = max(1, int(round(w * (20.0 / h))))
        else:
            new_w = 20
            new_h = max(1, int(round(h * (20.0 / w))))
        try:
            resized = Image.fromarray(crop_arr).resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = Image.fromarray(crop_arr).resize((new_w, new_h))
        new_img = Image.new('L', (28, 28), color=0)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        new_img.paste(resized, (left, top))
    else:
        # 28x28にリサイズ
        try:
            new_img = Image.fromarray(crop_arr).resize((28, 28), Image.LANCZOS).convert('L')
        except Exception:
            new_img = Image.fromarray(crop_arr).resize((28, 28)).convert('L')

    # center of mass
    if center_mass:
        arr_new = np.array(new_img).astype(np.float32)
        total = arr_new.sum()
        if total > 0:
            ys = np.arange(arr_new.shape[0])
            xs = np.arange(arr_new.shape[1])
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            cx = (arr_new * xx).sum() / total
            cy = (arr_new * yy).sum() / total
            target_x = (arr_new.shape[1] - 1) / 2.0
            target_y = (arr_new.shape[0] - 1) / 2.0
            shift_x = int(round(target_x - cx))
            shift_y = int(round(target_y - cy))
            final_img = Image.new('L', (28, 28), color=0)
            final_img.paste(new_img, (shift_x, shift_y))
            new_img = final_img

    tensor = transform_tensor(new_img).unsqueeze(0)
    return tensor, new_img

#  --------- 予測関数 ----------
def predict_with_options(image, invert=True, binarize=True, crop=True, scale20=True, center_mass=True):
    """UI用ラッパー: オプションの前処理ステップを使って予測し、(probs_dict, display_img)を返す。"""

    img_data = image.get('composite')
    if img_data is None:
        return {str(i): 0.0 for i in range(10)}, Image.new('L', (28, 28), color=0)

    tensor, display_img = preprocess_with_options(img_data, invert=invert, binarize=binarize, crop=crop, scale20=scale20, center_mass=center_mass)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    return {str(i): float(probs[i]) for i in range(10)}, display_img

#  --------- モデル再訓練関数 ----------
def retrain_model(epochs):
    global model
    print(f"\nモデル再訓練開始、エポック数: {epochs}")
    model, log_output = train_model("model.pth", epochs=epochs, capture_log=True)
    model = load_model()
    return log_output


#  --------- 評価関数 ----------
def evaluate_model(model_obj=None, batch_size=256):
    """MNISTテストセットでモデルを評価し、精度と簡単なレポート文字列を返す"""
    model_eval = model_obj if model_obj is not None else load_model()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    model_eval.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model_eval(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    acc = correct / total if total > 0 else 0.0
    report = f"テストセットの正確度: {acc*100:.2f}% ({correct}/{total})"
    # テストセットの正確度レポート文字列を返す
    return report

#  --------- Gradio 界面 ----------
with gr.Blocks(title="手書き数字認識") as interface:
    gr.Markdown("# 🎨 手書き数字識別システム")
    gr.Markdown("下のキャンバスに数字（0～9）を描くと、モデルがリアルタイムで認識します。モデルの精度を向上させるために再訓練も可能です。")
    
    #  ブラシサイズを動的に更新するためのコールバック
    def update_brush(size):
        try:
            return gr.update(brush=gr.Brush(default_size=int(size), default_color="#000000", colors=["#000000"], color_mode='fixed'))
        except Exception:
            return gr.update()

    with gr.Row():
        
        with gr.Column(scale=2):
            gr.Markdown("### 数字識別")
            # ブラシ初期サイズ8のスケッチパッド
            canvas = gr.Sketchpad(
                canvas_size=(280, 280),
                image_mode="L",
                label="数字を描くキャンバス",
                type="pil",
                brush=gr.Brush(default_size=8, default_color="#000000", colors=["#000000"], color_mode='fixed')
            )
            # ブラシサイズコントローラー
            brush_size = gr.Slider(minimum=1, maximum=40, value=8, step=1, label="ブラシサイズ")
            # 前処理オプションおよびボタン
            clear_btn = gr.Button("🧹 キャンバスをクリア", variant="secondary")
            with gr.Row():
                invert_cb = gr.Checkbox(value=True, label="自動反転")
                binarize_cb = gr.Checkbox(value=True, label="二値化")
                crop_cb = gr.Checkbox(value=True, label="前景にトリミング")
                scale20_cb = gr.Checkbox(value=True, label="20にスケーリングして中央に配置")
                center_mass_cb = gr.Checkbox(value=True, label="重心を中央に配置")
            predict_btn = gr.Button("識別", variant="primary", scale=1)
            with gr.Row():
                output = gr.Label(label="識別結果")
                processed_img_out = gr.Image(type="pil", height=400, label="モデル入力 (28x28)", interactive=False)
                

        with gr.Column(scale=1):
            gr.Markdown("### モデル訓練")
            gr.Markdown("エポック数を調整してモデルを再訓練")
            epochs_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                label="訓練エポック数",
                info="訓練回数が多いほど精度が向上する可能性がありますが、時間もかかります"
            )
            train_btn = gr.Button("🔄 モデルを再訓練", variant="secondary")
            train_output = gr.Textbox(label="訓練状況", lines=6, interactive=False)
            eva_model_btn = gr.Button("モデル評価", variant="primary", scale=1)
            eva_output = gr.Textbox(label="評価結果", lines=2, interactive=False)

    # ボタンのクリックイベントを設定
    predict_btn.click(
        fn=predict_with_options,
        inputs=[canvas, invert_cb, binarize_cb, crop_cb, scale20_cb, center_mass_cb],
        outputs=[output, processed_img_out]
    )
    train_btn.click(retrain_model, inputs=epochs_slider, outputs=train_output)
    eva_model_btn.click(fn=evaluate_model, inputs=None, outputs=eva_output)
    clear_btn.click(fn=lambda: None, inputs=None, outputs=canvas)

    # ブラシサイズが変わったときにキャンバスのブラシ設定を動的に更新
    brush_size.change(fn=update_brush, inputs=brush_size, outputs=canvas)

interface.launch(theme=gr.themes.Soft())
