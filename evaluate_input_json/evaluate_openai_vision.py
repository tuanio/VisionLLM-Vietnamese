'''
python evaluate_input_json/evaluate_openai_vision.py --input_json_file eval-data/test_eval.json --output_json_file path/to/result.json --openai_api_key YOUR_API_KEY --model_name gpt-4o-mini
'''
import os
import time
from pathlib import Path
from openai import OpenAI
import argparse
from tqdm.auto import tqdm
import numpy as np
import base64
import requests
import json

# prompt = """Dựa vào thông tin ảnh được cung cấp. So sánh kết quả thực tế và dự đoán từ các mô hình AI, để đưa ra điểm chính xác cho dự đoán. Điểm chính xác là 0.0 (hoàn toàn sai), 0.1, 0.2, 0.3, 0.4, 0.5 (nửa đúng nửa sai), 0.6, 0.7, 0.8, 0.9, hoặc 1.0 (hoàn toàn đúng). Nếu làm tốt tôi sẽ bo 1000$, hãy kiểm tra đáp án thật chính xác. Chỉ cần điền vào khoảng trống cuối cùng của điểm chính xác.

# Câu hỏi | Kết quả thực tế | Dự đoán | Độ chính xác
# --- | --- | --- | ---
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme nói về Iceland và Greenland. Nó chỉ ra rằng mặc dù có tên như vậy, Iceland không quá băng giá và Greenland không quá xanh. | 0.4
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme này sử dụng hài hước để chỉ ra bản chất gây hiểu lầm của tên Iceland và Greenland. Iceland, mặc dù có tên như vậy, lại có phong cảnh xanh tươi trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Dòng chữ 'Đây là lý do tại sao tôi có vấn đề về lòng tin' là một cách hài hước để gợi ý rằng những mâu thuẫn này có thể dẫn đến sự không tin tưởng hoặc nhầm lẫn. Sự hài hước trong meme này xuất phát từ sự tương phản bất ngờ giữa tên của các quốc gia và đặc điểm vật lý thực tế của chúng. | 1.0
# {} | {} | {} | """

# prompt = """Dựa vào thông tin ảnh được cung cấp, hãy so sánh kết quả thực tế và dự đoán từ mô hình AI, rồi đưa ra điểm chính xác cho dự đoán (mức độ tương đồng với kết quả thực tế). Điểm chính xác là 0.0 (hoàn toàn sai), 0.1, 0.2, 0.3, 0.4, 0.5 (nửa đúng nửa sai), 0.6, 0.7, 0.8, 0.9, hoặc 1.0 (hoàn toàn đúng). Nếu làm tốt tôi sẽ bo 1000$, hãy kiểm tra đáp án thật chính xác. Chỉ cần điền vào khoảng trống cuối cùng của điểm chính xác.

# Câu hỏi | Kết quả thực tế | Dự đoán | Độ chính xác
# --- | --- | --- | ---
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme nói về Iceland và Greenland. Nó chỉ ra rằng mặc dù có tên như vậy, Iceland không quá băng giá và Greenland không quá xanh. | 0.4
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme này sử dụng hài hước để chỉ ra bản chất gây hiểu lầm của tên Iceland và Greenland. Iceland, mặc dù có tên như vậy, lại có phong cảnh xanh tươi trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Dòng chữ 'Đây là lý do tại sao tôi có vấn đề về lòng tin' là một cách hài hước để gợi ý rằng những mâu thuẫn này có thể dẫn đến sự không tin tưởng hoặc nhầm lẫn. Sự hài hước trong meme này xuất phát từ sự tương phản bất ngờ giữa tên của các quốc gia và đặc điểm vật lý thực tế của chúng. | 1.0
# {} | {} | {} | """

# prompt = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

# Question | Ground truth | Prediction | Correctness
# --- | --- | --- | ---
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme nói về Iceland và Greenland. Nó chỉ ra rằng mặc dù có tên như vậy, Iceland không quá băng giá và Greenland không quá xanh. | 0.4
# Bạn có thể giải thích meme này không? | Meme này đang chế giễu việc tên của các quốc gia Iceland và Greenland gây hiểu lầm. Mặc dù có tên như vậy, Iceland nổi tiếng với phong cảnh xanh tươi đẹp, trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Meme này nói rằng người đó có vấn đề về lòng tin vì tên của những quốc gia này không phản ánh chính xác phong cảnh của chúng. | Meme này sử dụng hài hước để chỉ ra bản chất gây hiểu lầm của tên Iceland và Greenland. Iceland, mặc dù có tên như vậy, lại có phong cảnh xanh tươi trong khi Greenland phần lớn được bao phủ bởi băng và tuyết. Dòng chữ 'Đây là lý do tại sao tôi có vấn đề về lòng tin' là một cách hài hước để gợi ý rằng những mâu thuẫn này có thể dẫn đến sự không tin tưởng hoặc nhầm lẫn. Sự hài hước trong meme này xuất phát từ sự tương phản bất ngờ giữa tên của các quốc gia và đặc điểm vật lý thực tế của chúng. | 1.0
# {} | {} | {} | """

sys_prompt = """Compare the ground truth and prediction from AI model, to give a correctness score for the prediction (how similar the prediction compare to ground truth). The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5 (half similar), 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). OUTPUT ONLY ONE SCORE FLOAT NUMBER.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 and -5 | x = 3 | 0.0
What is x in the equation? | -1 and -5 | x = -5 | 0.5
What is x in the equation? | -1 and -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
{} | {} | {} | """

def get_prompt(question, ground_truth, prediction):
    return sys_prompt.format(question, ground_truth, prediction)


class OpenAI:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def encode_image(self, image_path: str) -> str:
        """Encode the image as base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_response(self, image_path: str, prompt: str) -> str:
        """Get a response from the OpenAI API using an image and a prompt."""
        text = ""
        while len(text) < 1:
            try:
                base64_image = self.encode_image(image_path)
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                response_json = response.json()
                try:
                    text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                except KeyError:
                    text = ""
                
            except Exception as error:
                print(f"Error: {error}")
                print("Sleeping for 10 seconds")
                time.sleep(10)
                
            try:
                float(text)  # This may not be necessary depending on the response structure
            except ValueError:
                print("Could not get score! Try again...")
                text = ""
                
        return text.strip()



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="input.json",
        help="Path to input json file",
    )
    parser.add_argument(
        "--output_json_file", 
        type=str,
        default="output.json",
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None,
        help="https://platform.openai.com/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name",
    )
    parser.add_argument(
        "-f", required=False,  # Thêm đối số -f không bắt buộc
        help="Dummy argument for Jupyter compatibility",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    input_json_file = args.input_json_file
    output_json_file = args.output_json_file

    if args.openai_api_key is not None:
        OPENAI_API_KEY = args.openai_api_key
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_API_KEY is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable or pass it as an argument")

    model = OpenAI(model=args.model_name, api_key= OPENAI_API_KEY)

    with open(input_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result = []
    for item in data:
        id_image = item['id_image']
        image_path = item['image']
        prompt = get_prompt(question=item['question'], ground_truth=item['answer'], prediction=item['prediction'])
        list_scores = [model.get_response(image_path=image_path, prompt=prompt) for _ in range(5)]
        score = np.mean(list(map(float, list_scores)))
        result.append({
            'id_image': id_image,
            'list_scores': list_scores,
            'score': score
        })

    with open(output_json_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4)

    print("Evaluation done!")

