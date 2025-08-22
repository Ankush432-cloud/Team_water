import os
import json
import streamlit as st

def _safe_len(path: str) -> int:
	try:
		return len([p for p in os.listdir(path) if os.path.isfile(os.path.join(path, p))])
	except Exception:
		return 0

def _dataset_overview():
	train_normal = _safe_len('Datasets/train/NORMAL')
	train_pneumonia = _safe_len('Datasets/train/PNEUMONIA')
	test_normal = _safe_len('Datasets/test/NORMAL')
	test_pneumonia = _safe_len('Datasets/test/PNEUMONIA')

	cols = st.columns(4)
	with cols[0]:
		st.metric("Train NORMAL", train_normal)
	with cols[1]:
		st.metric("Train PNEUMONIA", train_pneumonia)
	with cols[2]:
		st.metric("Test NORMAL", test_normal)
	with cols[3]:
		st.metric("Test PNEUMONIA", test_pneumonia)

def _model_status():
	model_path = 'model/chest_anomaly_model.h5'
	class_file = 'class_names.json'

	model_exists = os.path.exists(model_path)
	class_exists = os.path.exists(class_file)

	cols = st.columns(2)
	with cols[0]:
		st.subheader("Model File")
		if model_exists:
			st.success(f"Found: {model_path}")
		else:
			st.error("Model file not found. Train the model first.")
	with cols[1]:
		st.subheader("Class Names")
		if class_exists:
			try:
				with open(class_file, 'r') as f:
					classes = json.load(f)
				st.success(f"Loaded: {classes}")
			except Exception as e:
				st.error(f"Error reading class_names.json: {e}")
		else:
			st.error("class_names.json not found.")

def home_page():
	st.title("Chest X-Ray Anomaly Detection")
	st.caption("Classify chest X-rays as NORMAL or PNEUMONIA using a CNN model.")

	st.markdown("""
	**How it works**
	- Upload a chest X-ray image in the Image Prediction mode
	- The image is resized to 224×224 and normalized
	- The trained model outputs probabilities for each class
	- The top class and confidence are shown with the image
	""")

	st.divider()
	st.subheader("Project Quick Stats")
	_dataset_overview()

	st.divider()
	st.subheader("Environment & Assets")
	_model_status()

	st.divider()
	st.subheader("Instructions")
	st.markdown("""
	1. Go to the sidebar and choose "🔍 Image Prediction".
	2. Upload a JPG/PNG chest X-ray image.
	3. Click "Predict Disease" to get the result.
	4. Use the Chatbot mode for general Q&A about the task.
	""")