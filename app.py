from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
from torchvision import transforms
import torch
from model import EDSR_baseline
from utils import tensor_to_img

app = Flask(__name__)
UPLOAD_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR_baseline()
model.load_state_dict(torch.load("student_sr_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file).convert('RGB')

            # Save input (Low-Res) image
            input_path = os.path.join(UPLOAD_FOLDER, 'input.png')
            img.save(input_path)

            # Super-resolve
            lr_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            # Save output (SR) image
            sr_img = tensor_to_img(sr_tensor.squeeze(0))
            output_path = os.path.join(UPLOAD_FOLDER, 'output.png')
            Image.fromarray(sr_img).save(output_path)

            return render_template('result.html',
                                   lr_path=input_path,
                                   sr_path=output_path)
    return render_template('index.html')

@app.route('/download')
def download_file():
    return send_from_directory(UPLOAD_FOLDER, 'output.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
