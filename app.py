# app.py - 垃圾分类识别网页应用
import torch
from torchvision import transforms, models
from PIL import Image
import io
import json
from flask import Flask, request, jsonify, render_template_string
import os
import glob
# ========== 1. 初始化Flask应用 ==========
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 限制为8MB

MODEL_FILE = 'best_garbage_resnet50.pth'
PART_PATTERN = 'best_garbage_resnet50.pth_part*'

# 如果完整模型不存在，但分割文件存在，则合并
if not os.path.exists(MODEL_FILE):
    part_files = sorted(glob.glob(PART_PATTERN))
    if part_files:
        print(f"检测到 {len(part_files)} 个模型分片，开始合并...")
        with open(MODEL_FILE, 'wb') as outfile:
            for part_file in part_files:
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"✅ 模型合并完成: {MODEL_FILE}")
    else:
        print(f"⚠️  未找到模型文件 {MODEL_FILE} 或其分片")

# ========== 然后正常加载模型 ==========
if os.path.exists(MODEL_FILE):
    checkpoint = torch.load(MODEL_FILE, map_location='cpu')
    # ... 后续您的加载代码
else:
    raise FileNotFoundError(f"模型文件 {MODEL_FILE} 不存在")
# ========== 2. 加载训练好的模型 ==========
# 设置设备（优先使用GPU，训练时用的GPU，推理时用CPU也可以）
device = torch.device('cpu')
print(f"使用设备: {device} 进行推理")

# ！！！重要：请确保模型路径与训练代码中保存的路径一致 ！！！
MODEL_PATH = 'best_garbage_resnet50.pth'

# 2.1 构建模型结构（必须与训练时完全一致）
model = models.resnet50(weights=None)  # 不加载预训练权重
num_classes = 6  # ！！！务必与训练时设定的类别数一致（你的数据集是5类）！！！
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 2.2 加载保存的权重和映射
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # 设置为评估模式
print("模型加载完毕！")

# 2.3 获取类别名称映射
# 从检查点中加载类别到索引的映射，并反转得到索引到类别名的映射
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("类别映射:", idx_to_class)

# ========== 3. 定义图像预处理（必须与训练时验证集的预处理一致） ==========
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 4. 定义预测函数 ==========
def predict_image(image_bytes):
    """接收图片的二进制数据，返回预测结果"""
    # 将二进制数据转换为PIL图像
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 应用预处理
    input_tensor = preprocess(image)
    # 增加一个批次维度 [C, H, W] -> [1, C, H, W]
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 执行预测，不计算梯度以节省内存
    with torch.no_grad():
        output = model(input_batch)
    
    # 获取预测结果（概率和类别索引）
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probabilities, 0)
    
    # 将结果转换为Python原生类型，便于JSON序列化
    predicted_class = idx_to_class[top_idx.item()]
    confidence = top_prob.item()
    
    # 获取所有类别的概率（可选，用于前端显示详细结果）
    all_probs = {idx_to_class[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    return predicted_class, confidence, all_probs

# ========== 5. 定义Flask路由 ==========
# 5.1 主页：显示一个简单的上传表单
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>垃圾分类识别器</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
        .upload-box { border: 2px dashed #ccc; padding: 30px; text-align: center; margin: 20px 0; }
        #preview { max-width: 300px; max-height: 300px; margin-top: 15px; }
        .result { margin-top: 25px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .class-item { margin: 5px 0; }
        .progress-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 5px 0; }
        .progress-fill { height: 100%; background: #4CAF50; }
        button { background: #4CAF50; color: white; border: none; padding: 12px 24px; cursor: pointer; border-radius: 4px; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>🚮 垃圾分类识别系统</h1>
    <p>上传一张垃圾图片，AI将识别其类别。支持类别：cardboard, glass, metal, paper, plastic。</p>
    
    <form id="uploadForm" enctype="multipart/form-data">
        <div class="upload-box">
            <input type="file" id="imageInput" name="image" accept="image/*" required>
            <p>或将图片拖拽至此区域</p>
            <img id="preview" style="display:none;">
        </div>
        <button type="submit">识别垃圾类别</button>
    </form>
    
    <div id="result" class="result" style="display:none;">
        <h3>识别结果：</h3>
        <p><strong>类别：</strong><span id="predClass"></span></p>
        <p><strong>置信度：</strong><span id="confidence"></span></p>
        <div id="allClasses"></div>
    </div>
    
    <script>
        // 图片预览功能
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const preview = document.getElementById('preview');
                    preview.src = event.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        // 处理表单提交（使用AJAX，避免页面刷新）
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            formData.append('image', document.getElementById('imageInput').files[0]);
            
            const button = this.querySelector('button');
            button.textContent = '识别中...';
            button.disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                // 显示结果
                document.getElementById('predClass').textContent = result.class;
                document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';
                
                // 显示所有类别的概率
                const allClassesDiv = document.getElementById('allClasses');
                allClassesDiv.innerHTML = '<h4>所有类别概率：</h4>';
                for (const [cls, prob] of Object.entries(result.all_probs || {})) {
                    const percent = (prob * 100).toFixed(1);
                    allClassesDiv.innerHTML += `
                        <div class="class-item">
                            <div>${cls}: ${percent}%</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: ${percent}%"></div></div>
                        </div>
                    `;
                }
                
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                alert('识别失败：' + error.message);
            } finally {
                button.textContent = '识别垃圾类别';
                button.disabled = false;
            }
        });
        
        // 拖拽上传功能
        const uploadBox = document.querySelector('.upload-box');
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#4CAF50';
        });
        uploadBox.addEventListener('dragleave', () => {
            uploadBox.style.borderColor = '#ccc';
        });
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#ccc';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                document.getElementById('imageInput').files = e.dataTransfer.files;
                // 触发change事件以显示预览
                document.getElementById('imageInput').dispatchEvent(new Event('change'));
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """返回上传表单页面"""
    return render_template_string(HTML_FORM)

# 5.2 预测接口：接收上传的图片并返回JSON格式的预测结果
@app.route('/predict', methods=['POST'])
def predict():
    """处理图片上传和预测"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400
    
    try:
        # 读取图片字节数据
        img_bytes = file.read()
        # 调用预测函数
        predicted_class, confidence, all_probs = predict_image(img_bytes)
        
        # 返回JSON结果
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_probs': all_probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== 6. 启动应用 ==========
if __name__ == '__main__':
    # 获取环境变量中的端口（Render会自动设置）
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
