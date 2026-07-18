"""Minimal test: Paddle Inference with ASCII path vs Chinese path"""
import os
import json

# Step 1: Check if inference.json is valid JSON
json_path = r"C:\paddle_test\inference.json"
print(f"Checking {json_path}...")
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  JSON is valid, keys: {list(data.keys())[:5]}")
    print(f"  File size: {os.path.getsize(json_path)} bytes")
except Exception as e:
    print(f"  FAILED to parse: {e}")

# Step 2: Try Paddle Inference Config with ASCII path
print("\nCreating paddle_inference.Config with ASCII path...")
try:
    import paddle.inference as paddle_inference
    model_file = r"C:\paddle_test\inference.json"
    params_file = r"C:\paddle_test\inference.pdiparams"
    print(f"  model_file: {model_file}")
    print(f"  params_file: {params_file}")
    print(f"  model_file exists: {os.path.exists(model_file)}")
    print(f"  params_file exists: {os.path.exists(params_file)}")
    config = paddle_inference.Config(str(model_file), str(params_file))
    config.disable_gpu()
    config.disable_glog_info()
    print("  Config created OK")
    predictor = paddle_inference.create_predictor(config)
    print("  Predictor created OK!")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Try with Chinese path
print("\nCreating paddle_inference.Config with Chinese path...")
try:
    model_file_cn = r"C:\Users\马龙\.paddlex\official_models\PP-LCNet_x1_0_doc_ori\inference.json"
    params_file_cn = r"C:\Users\马龙\.paddlex\official_models\PP-LCNet_x1_0_doc_ori\inference.pdiparams"
    print(f"  model_file: {model_file_cn}")
    print(f"  model_file exists: {os.path.exists(model_file_cn)}")
    config2 = paddle_inference.Config(str(model_file_cn), str(params_file_cn))
    config2.disable_gpu()
    config2.disable_glog_info()
    print("  Config created OK")
    predictor2 = paddle_inference.create_predictor(config2)
    print("  Predictor created OK!")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
