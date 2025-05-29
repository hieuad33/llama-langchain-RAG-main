import os
import csv
import chromadb


CHROMA_PATH = "./choma_pro"

def main():
    query_text="iphone 13"
    print(get_product_forchat(query_text))
def get_product_forchat(chat):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    product_collection=client.get_collection(name="product_collect")
    query_results = product_collection.query(
        query_texts=chat,
        n_results=4)
        #using the 'where' condition for filtering based on category and price
        # where={
        #     '$and': [
        #         {'name': {'$eq': 'Electronics'}},
        #         {'price': {'$gte': 100}},
        #         {'price': {'$lte': 500}}
        #     ]
        # 
        
        # Extract metadata from query results
    data = query_results.get('metadatas', [])
    return data
         

import re

def remove_vietnamese_stopwords_concise(text):
    """
    Loại bỏ các từ dừng tiếng Việt khỏi văn bản một cách gọn gàng.

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        str: Chuỗi văn bản sau khi đã loại bỏ từ dừng.
    """
    # Danh sách từ dừng tiếng Việt (đã chuyển về set để tìm kiếm nhanh hơn)
   
    # Chuẩn hóa văn bản: chuyển về chữ thường, loại bỏ ký tự không phải chữ/số/dấu cách, rồi tách từ
    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()
    
    # Lọc bỏ từ dừng và nối lại thành chuỗi
    return ' '.join(word for word in cleaned_text.split() if word not in stopwords)
       
    
def product_question(que):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    product_collection=client.get_collection(name="product_collect")
    query_results = product_collection.query(
        query_texts=que,
        n_results=4,
        #using the 'where' condition for filtering based on category and price
        # where={
        #     '$and': [
        #         {'name': {'$eq': 'Electronics'}},
        #         {'price': {'$gte': 100}},
        #         {'price': {'$lte': 500}}
        #     ]
        # }
    )
    data = query_results.get('metadatas', [])
    return data
if __name__ == "__main__":
    main()