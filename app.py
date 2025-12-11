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
from PIL import ImageOps
import sys
from contextlib import redirect_stdout

# ---------- 模型定义 ----------
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

# ---------- 模型训练 ----------
def train_model(model_path="model.pth", epochs=1, capture_log=False):
    """训练模型，可选择捕获日志"""
    if capture_log:
        log_capture = io.StringIO()
        stdout_backup = sys.stdout
        sys.stdout = log_capture
    
    print(f"开始训练模型，训练轮数：{epochs}...")
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
    print(f"模型训练完成并保存为 {model_path}")
    
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
        print("已加载已有模型。")
    else:
        model = train_model(model_path)
    model.eval()
    return model

model = load_model()

# ---------- 图像预处理 ----------
transform_input = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 用于将已经是 28x28 灰度 PIL 图转换为 tensor + normalize（避免重复 resize）
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess_image(pil_img):
    # delegate to the options-based preprocessing with default options
    return preprocess_with_options(pil_img, invert=True, binarize=True, crop=True, scale20=True, center_mass=True)


def preprocess_with_options(pil_img, invert=True, binarize=True, crop=True, scale20=True, center_mass=True):
    """同 preprocess_image，但每一步可开关，便于做消融实验。
    返回 (tensor, display_img)
    步骤对应含义：
      - invert: 自动反相背景
      - binarize: 二值化（阈值分割）
      - crop: 裁剪到前景 bbox
      - scale20: 把裁剪结果等比缩放到 20x(<=20) 并居中到28x28；否则直接缩放到28x28
      - center_mass: 质心居中平移
    """
    img = pil_img.convert('L')
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
        # 直接缩放到28x28
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


def ablation_predict(image):
    """对传入的画布/图像做消融：逐个关闭 preprocess 步骤并返回每种设置的预测与展示图。
    返回 dict: {label: (probs_dict, display_img)}
    可用于比较哪些预处理是关键。
    """
    # 统一把 sketchpad dict 转为 PIL image
    if isinstance(image, dict):
        img_data = image.get('composite') if image.get('composite') is not None else image.get('background')
        if img_data is None:
            raise ValueError('no image in sketchpad dict')
        pil = Image.fromarray(img_data)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(image)

    variants = {
        'all_steps': dict(invert=True, binarize=True, crop=True, scale20=True, center_mass=True),
        'no_invert': dict(invert=False, binarize=True, crop=True, scale20=True, center_mass=True),
        'no_binarize': dict(invert=True, binarize=False, crop=True, scale20=True, center_mass=True),
        'no_crop': dict(invert=True, binarize=True, crop=False, scale20=True, center_mass=True),
        'no_scale20': dict(invert=True, binarize=True, crop=True, scale20=False, center_mass=True),
        'no_center_mass': dict(invert=True, binarize=True, crop=True, scale20=True, center_mass=False),
    }

    results = {}
    for name, opts in variants.items():
        tensor, disp = preprocess_with_options(pil, **opts)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
        results[name] = ({str(i): float(probs[i]) for i in range(10)}, disp)

    return results

# ---------- 预测函数 ----------
def predict(image):
    # Sketchpad 返回字典格式，需要提取 'composite' 或 'background'
    if isinstance(image, dict):
        # 优先使用 composite（合成层），其次使用 background
        img_data = image.get('composite') if image.get('composite') is not None else image.get('background')
        if img_data is None:
            return {str(i): 0.0 for i in range(10)}
        image = img_data
    
    # 处理 PIL Image 对象
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # 使用稳健预处理，返回 tensor 和展示图
    tensor, display_img = preprocess_image(image)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    # 返回概率字典和用于展示的 PIL 图像
    return ({str(i): float(probs[i]) for i in range(10)}, display_img)


def predict_with_options(image, invert=True, binarize=True, crop=True, scale20=True, center_mass=True):
    """Wrapper for UI: 使用可选预处理步骤进行预测并返回 (probs_dict, display_img)。"""
    # 统一把 sketchpad dict 转为 PIL image，支持多种输入类型
    def to_pil(obj):
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray):
            try:
                return Image.fromarray(obj)
            except Exception:
                # sometimes shape or dtype unexpected
                return Image.fromarray(obj.astype('uint8'))
        if isinstance(obj, bytes):
            from io import BytesIO
            try:
                return Image.open(BytesIO(obj))
            except Exception:
                return None
        if isinstance(obj, str):
            # filepath or base64? try open file
            if os.path.exists(obj):
                try:
                    return Image.open(obj)
                except Exception:
                    return None
        return None

    if isinstance(image, dict):
        img_data = image.get('composite') if image.get('composite') is not None else image.get('background')
        if img_data is None:
            return {str(i): 0.0 for i in range(10)}, Image.new('L', (28, 28), color=0)
        pil = to_pil(img_data)
    else:
        pil = to_pil(image)

    if pil is None:
        # 无法解析输入，返回空白图和零概率，避免 UI 崩溃
        return {str(i): 0.0 for i in range(10)}, Image.new('L', (28, 28), color=0)

    tensor, display_img = preprocess_with_options(pil, invert=invert, binarize=binarize, crop=crop, scale20=scale20, center_mass=center_mass)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    return {str(i): float(probs[i]) for i in range(10)}, display_img

# ---------- 重新训练函数 ----------
def retrain_model(epochs):
    global model
    print(f"\n开始重新训练模型，Epoch: {epochs}")
    model, log_output = train_model("model.pth", epochs=epochs, capture_log=True)
    model = load_model()
    return log_output


# ---------- 评估函数 ----------
def evaluate_model(model_obj=None, batch_size=256):
    """在 MNIST 测试集上评估模型，返回准确率和简单报告字符串"""
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
    report = f"测试集准确率: {acc*100:.2f}% ({correct}/{total})"
    return acc, report

# ---------- Gradio 界面 ----------
with gr.Blocks(title="手写数字识别") as interface:
    gr.Markdown("# 🎨 手写数字识别系统")
    gr.Markdown("在下方画布上绘制数字（0～9），模型会实时识别。你也可以重新训练模型来提高准确率。")
    
    # 用于动态更新画笔的回调
    def update_brush(size):
        try:
            return gr.update(brush=gr.Brush(default_size=int(size), default_color="#000000", colors=["#000000"], color_mode='fixed'))
        except Exception:
            return gr.update()

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 数字识别")
            # 初始画笔大小设为 8（较细）
            canvas = gr.Sketchpad(
                canvas_size=(280, 280),
                image_mode="L",
                label="绘制数字",
                type="pil",
                brush=gr.Brush(default_size=8, default_color="#000000", colors=["#000000"], color_mode='fixed')
            )
            # 画笔大小控制器
            brush_size = gr.Slider(minimum=1, maximum=40, value=8, step=1, label="画笔大小")
            # 预处理选项
            invert_cb = gr.Checkbox(value=True, label="自动反相")
            binarize_cb = gr.Checkbox(value=True, label="二值化")
            crop_cb = gr.Checkbox(value=True, label="裁剪到前景")
            scale20_cb = gr.Checkbox(value=True, label="缩放到20并居中")
            center_mass_cb = gr.Checkbox(value=True, label="质心居中")
            predict_btn = gr.Button("识别", variant="primary", scale=1)
            output = gr.Label(label="识别结果")
            processed_img_out = gr.Image(type="pil", label="模型输入 (28x28)", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 模型训练")
            gr.Markdown("调整 epoch 数量并重新训练模型")
            epochs_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="训练 Epoch 数",
                info="训练轮数越多，可能精度越高但耗时越长"
            )
            train_btn = gr.Button("🔄 重新训练模型", variant="secondary")
            train_output = gr.Textbox(label="训练状态", interactive=False)
    
    # 绑定事件（使用带选项的预测函数）
    predict_btn.click(
        fn=predict_with_options,
        inputs=[canvas, invert_cb, binarize_cb, crop_cb, scale20_cb, center_mass_cb],
        outputs=[output, processed_img_out]
    )
    train_btn.click(retrain_model, inputs=epochs_slider, outputs=train_output)

    # 画笔大小变化时动态更新画布的 brush 设置
    brush_size.change(fn=update_brush, inputs=brush_size, outputs=canvas)

if __name__ == "__main__":
    interface.launch(theme=gr.themes.Soft())
