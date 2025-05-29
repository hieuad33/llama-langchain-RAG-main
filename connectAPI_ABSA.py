import requests
import json

# Địa chỉ cơ bản của API của bạn
# Đảm bảo địa chỉ này khớp với nơi Uvicorn đang chạy API
BASE_URL = "http://26.176.201.242:8000"

# 1. Kết nối đến endpoint /analyze_product/
def analyze_product_api(product_url: str):
    """
    Gửi yêu cầu phân tích sản phẩm đến API.
    """
    endpoint = f"{BASE_URL}/analyze_product/"
    payload = {"product_url": product_url}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status() # Ném một ngoại lệ cho các mã trạng thái lỗi (4xx hoặc 5xx)

        data = response.json()
        print("\n--- Kết quả Phân tích Sản phẩm ---")
        print(json.dumps(data, indent=4, ensure_ascii=False)) # In kết quả dưới dạng JSON đẹp
        return data
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi kết nối hoặc nhận phản hồi từ API analyze_product: {e}")
        return {'error':f"Lỗi khi kết nối hoặc nhận phản hồi từ API analyze_product: {e}"}
    except json.JSONDecodeError:
        print(f"Lỗi giải mã JSON từ phản hồi analyze_product. Phản hồi thô: {response.text}")
        return {'error':f"Lỗi giải mã JSON từ phản hồi analyze_product. Phản hồi thô: {response.text}"}

# 2. Kết nối đến endpoint /predict_sentiment/
def predict_sentiment_api(text: str):
    """
    Gửi yêu cầu dự đoán cảm xúc cho văn bản đến API.
    """
    endpoint = f"{BASE_URL}/predict_sentiment/"
    payload = {"text": text}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status() # Ném một ngoại lệ cho các mã trạng thái lỗi (4xx hoặc 5xx)

        data = response.json()['prediction']
        print("\n--- Kết quả Dự đoán Cảm xúc ---")
        print(json.dumps(data, indent=4, ensure_ascii=False)) # In kết quả dưới dạng JSON đẹp
        return data
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi kết nối hoặc nhận phản hồi từ API predict_sentiment: {e}")
        return {'error':f"Lỗi giải mã JSON từ phản hồi analyze_product. Phản hồi thô: {response.text}"}
    except json.JSONDecodeError:
        print(f"Lỗi giải mã JSON từ phản hồi predict_sentiment. Phản hồi thô: {response.text}")
        return {'error':f"Lỗi giải mã JSON từ phản hồi analyze_product. Phản hồi thô: {response.text}"}


# 3. Kết nối đến endpoint /predict_multilingual/
def predict_multilingual_api(text: str):
    """
    Gửi yêu cầu dự đoán đa ngôn ngữ cho văn bản đến API.
    """
    endpoint = f"{BASE_URL}/predict_multilingual/"
    payload = {"text": text}

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status() # Ném một ngoại lệ cho các mã trạng thái lỗi (4xx hoặc 5xx)

        data = response.json()
        print("\n--- Kết quả Dự đoán Đa ngôn ngữ ---")
        print(json.dumps(data, indent=4, ensure_ascii=False)) # In kết quả dưới dạng JSON đẹp

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi kết nối hoặc nhận phản hồi từ API predict_multilingual: {e}")
    except json.JSONDecodeError:
        print(f"Lỗi giải mã JSON từ phản hồi predict_multilingual. Phản hồi thô: {response.text}")
