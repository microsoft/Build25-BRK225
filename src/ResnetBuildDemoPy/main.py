from pathlib import Path

from PIL import Image
import numpy as np
import onnxruntime as ort


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    means = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    stds = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - means) / stds    
    img_array = img_array.transpose((2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip().split(',')[1] for line in f.readlines()]
    return labels

def print_results(lables, results, is_logit=False):
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    results = results.flatten()
    if is_logit:
        results = softmax(results)
    top_k = 5
    top_indices = np.argsort(results)[-top_k:][::-1]
    print("Top Predictions:")
    print("-"*50)
    print(f"{'Label':<32} {'Confidence':>10}")
    print("-"*50)
    
    for i in top_indices:
        print(f"{lables[i]:<32} {results[i]*100:>10.2f}%")
    
    print("-"*50)

def is_compiled_model_available():
    return compiled_model_path.exists()


print("Creating session ...")

model_path = Path(__file__).parent / "SqueezeNet.onnx"
compiled_model_path = Path(__file__).parent / "SqueezeNet_ctx.onnx"
session_options = ort.SessionOptions()
session_options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_NPU)
# session_options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.MIN_OVERALL_POWER)
assert session_options.has_providers()

if is_compiled_model_available():
    print("Using compiled model")
else:
    print("No compiled model found, attempting to create compiled model at ", compiled_model_path)  
    model_compiler = ort.ModelCompiler(session_options, model_path)
    print("Starting compile, this may take a few moments..." )
    try:
        model_compiler.compile_to_file(compiled_model_path)
        print("Model compiled successfully")
    except Exception as e:
        print("Model compilation failed:", e)
        print("Falling back to uncompiled model")

model_path_to_use = compiled_model_path if is_compiled_model_available() else model_path

session = ort.InferenceSession(
    model_path_to_use,
    sess_options=session_options,
)

labels = load_labels(Path(__file__).parent / "SqueezeNet.Labels.txt")

for im_file in ["cat.jpg", "dog.jpg"]:
    im_file = Path(__file__).parent / im_file
    print(f"Running inference on image: {im_file}")
    print("Preparing input ...")
    img_array = load_and_preprocess_image(im_file)
    print("Running inference ...")
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: img_array})[0]
    print_results(labels, results, is_logit=False)