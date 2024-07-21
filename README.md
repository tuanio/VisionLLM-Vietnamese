## Hướng dẫn chạy mã VisionLLM-Vietnamese để Evaluation

Tệp tin này hướng dẫn cách chạy mã nguồn trong VisionLLM-Vietnamese để Evaluation hiệu suất

Gồm 2 phương pháp đánh giá mô hình LLM:
- Evaluate using Prompt and In-Context Learning: API của Google Gemini và OpenAI.
- Evaluate using Embedding → Compare cosine similarity

### Cài đặt

* **Bước 1:** Clone repository từ GitHub:
```bash
!git clone https://github.com/tuanio/VisionLLM-Vietnamese.git
```

* **Bước 2:** Di chuyển vào thư mục:
```bash
cd /kaggle/working/VisionLLM-Vietnamese
```

* **Bước 3:** Cài đặt các thư viện cần thiết:
```bash
!pip install -r requirements.txt
```
Hoặc cài đặt từng thư viện cụ thể:
```bash
!pip -q install sentence-transformers==3.0.1 openai==1.13.3 google==3.0.0
```

### Run code 

* **Bước 1:** Chuẩn bị dữ liệu đầu vào: Đảm bảo bạn có tệp tin JSON chứa dữ liệu đầu vào theo định dạng sau :
```json
[
    {
        "image": "/kaggle/input/vlsp-dataset/dev-images/dev-images/000000003442.jpg",
        "id_image": 3442,
        "question": "Hai bên đường người ta bày trí những gì?",
        "answer": "Những bức tường với các bài viết lịch sử",
        "prediction": "Những gian hàng lưu niệm, đồ ăn, thức uống, trò chơi, ..."
    },
    ....
]
```


* **Bước 2:** Chạy mã đánh giá bằng một trong các lệnh sau:
    * Evaluate using Embedding:
    ```bash
    python evaluate_input_json/evaluate_using_embedding.py --input_json_file <đường_dẫn_tệp_json_đầu_vào> --output_json_file <đường_dẫn_tệp_json_đầu_ra> --model_name <tên_mô_hình_embedding>
    ```
    * Evaluate using Prompt and In-Context Learning with Google Gemini:
    ```bash
    python evaluate_input_json/evaluate_gemini_vision.py --input_json_file <đường_dẫn_tệp_json_đầu_vào> --output_json_file <đường_dẫn_tệp_json_đầu_ra --google_api_key <GOOGLE_API_KEY> 
    ```
    * Evaluate using Prompt and In-Context Learning with OpenAI Vision:
    ```bash
    python evaluate_input_json/evaluate_openai_vision.py --input_json_file <đường_dẫn_tệp_json_đầu_vào> --output_json_file <đường_dẫn_tệp_json_đầu_ra> --openai_api_key <OPENAI_API_KEY>
    ```
    * **Lưu ý:** 
    - Thay thế `<...>` bằng đường dẫn và thông tin tương ứng.
    - Các api key nếu đã lưu trong biến môi trường thì có thể không cần truyền vào lệnh chạy.
    - output_json_file: sẽ default là `output.json` nếu không truyền vào.
    - name_model: tên mô hình embedding, mặc định là `hiieu/halong_embedding`. Có thể tham khảo thêm `dangvantuan/vietnamese-embedding`, `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` hoặc các mô hình embedding khác.

###  Kết quả

Kết quả đánh giá sẽ được lưu vào tệp tin JSON đầu ra được chỉ định trong lệnh chạy theo định dạng sau: 
```json
[
    {
        "id_image": 3442,
        "list_score": [
            0.25294792652130127
        ],
        "score": 0.25294792652130127
    },
    ......
]
```
