pip install inference
pip install inference-gpu

export ROBOFLOW_API_KEY=API_KEY

import inference
model = inference.get_model("brain-tumor-detection-wfagt/1")
model.infer(image="YOUR_IMAGE.jpg")