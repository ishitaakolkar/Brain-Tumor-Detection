pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("brain-tumor-detection-wfagt")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "YOUR_VIDEO.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)