import os
import time
from pathlib import Path
from openai import OpenAI
import argparse
from tqdm.auto import tqdm
import numpy as np
import base64
import requests

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
        "--img-path",
        type=str,
        default="path/to/image.jpg",
        help="Path to image file",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    if args.openai_api_key is not None:
        OPENAI_API_KEY = args.openai_api_key
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_API_KEY is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable or pass it as an argument")

    model = OpenAI(model=args.model_name, api_key= OPENAI_API_KEY)

    image_path = 'eval-data/images/000000397133.jpg'
    prompt = get_prompt(
        question='Mô tả chi tiết của hình ảnh',
        ground_truth='Một người đàn ông mặc tạp dề đầu bếp đang đứng trong bếp, trước một chiếc lò nướng màu đen. Ông đang cầm một chiếc chảo trong một tay và một chiếc thìa trong tay kia. Sau lưng ông là một chiếc bàn có nhiều loại dụng cụ nấu ăn khác nhau. Trên bàn là một số bát, cốc và dụng cụ. Ở phía bên trái của bàn là một chiếc lò nướng khác và một cái chảo lớn hơn.\n\nCạnh bàn là một người khác, mặc áo sơ mi và quần tây. Người này dường như đang quan sát người đầu bếp làm bánh pizza. Trong góc bên phải của khung cảnh, có một bồn rửa lớn với vòi nước.',
        prediction="Hình ảnh cho thấy một người đàn ông mặc tạp dề đang đứng trong bếp. Anh ta đang đứng trước một cái bàn, trên đó có nhiều dụng cụ nấu ăn khác nhau. Người đàn ông đang cầm một cái thìa và một cái nĩa, có vẻ như anh ta đang chuẩn bị nấu ăn.\nCó một số vật dụng nhà bếp khác trong bếp, bao gồm một cái bát, một cái cốc và một cái thìa khác. Một cái bát khác có thể được nhìn thấy trên một cái bàn gần đó.\nCó một số vật dụng khác trong bếp, chẳng hạn như một cái lò nướng, một cái lò vi sóng và một số tủ."
    ) # => 0.6
    # image_path = 'eval-data/images/000000270244.jpg'
    # prompt = get_prompt(
    #     question='Mô tả chi tiết của hình ảnh',
    #     ground_truth='Đây là hình ảnh một con ngựa vằn đang đứng một mình trong khu rừng xanh. Con ngựa vằn đang đứng trên bãi cỏ, và xung quanh nó là những tán cây xanh tươi.',
    #     prediction="Hình ảnh cho thấy một cánh đồng cỏ rộng lớn, xanh tươi với một con ngựa vằn đang đứng một mình. Con ngựa vằn có bộ lông sọc đen trắng đặc trưng, ​​với những sọc đen dày hơn ở phần đầu và cổ. Nó đang đứng trên một bãi cỏ cao, xung quanh là những cây xanh tươi tốt. Bầu trời phía trên con ngựa vằn có màu xanh lam nhạt, với một vài đám mây trắng lơ lửng."
    # ) # => 0.9

    # image_path = 'eval-data/images/000000034873.jpg'
    # prompt = get_prompt(
    #     question='Mô tả chi tiết của hình ảnh',
    #     ground_truth='Đây là một nhà bếp rộng rãi, hiện đại với nhiều đồ nội thất và đồ gia dụng. Ở trung tâm nhà bếp là một quầy đảo lớn có bồn rửa nông trại. Một bồn rửa đôi khác có thể được nhìn thấy ở bức tường bên phải, gần một bếp nấu ăn màu đen.\n\nBên cạnh khu vực quầy bếp là một bàn ăn lớn với mặt bàn bằng đá cẩm thạch và tủ gỗ bao quanh. Có hai chiếc ghế gỗ màu đen xung quanh bàn ăn. Hai chiếc ghế khác có thể được nhìn thấy ở bức tường bên trái của khung cảnh, bên dưới một chiếc tivi gắn trên tường.\n\nTrên quầy đảo có một bát trái cây và một chiếc cốc.',
    #     prediction="Đây là một nhà bếp hiện đại với một hòn đảo trung tâm lớn ở giữa. Hòn đảo có một bồn rửa đôi bằng thép không gỉ ở một bên và một số tủ bếp ở bên kia. Trên bồn rửa có một số chai lọ và một số đồ dùng nhà bếp khác.\nCó một số ghế trong bếp, bao gồm một chiếc ghế dài ở bên phải hòn đảo và một số chiếc ghế khác ở phía xa hơn của căn phòng. Một chiếc tivi treo trên tường ở phía bên trái của hòn đảo.\nCó một cửa sổ lớn ở phía sau bếp, cung cấp nhiều ánh sáng tự nhiên. Sàn nhà được lát gạch màu sáng và tường được sơn màu trắng."
    # ) # => 0.7

    # image_path = 'eval-data/images/000000397133.jpg'
    # prompt = get_prompt(
    
    #     question='Người đầu bếp này đang chuẩn bị món gì?',
    #     ground_truth='Trong ảnh, người đầu bếp đứng trước lò nướng và đang làm bánh. Người đó đang sử dụng các dụng cụ khác nhau, chẳng hạn như chảo và đồ đựng nướng, hỗ trợ cho quá trình này. Bàn cạnh đó bày la liệt các loại bột và nguyên liệu khác, cho thấy người đầu bếp có thể đang làm nhiều món cùng lúc.',
    #     prediction="Người đầu bếp đang chuẩn bị một món ăn trên một chiếc bàn gỗ. Trên bàn có một cái tô lớn, một cái thìa và một số dụng cụ khác. Người đầu bếp đang đứng trước một cái bếp, nơi có một cái nồi lớn trên bếp. Có vẻ như người đầu bếp đang nấu một món ăn phức tạp, vì có nhiều nguyên liệu và dụng cụ khác nhau trên bàn. Người đầu bếp mặc đồng phục đầu bếp màu trắng và đang tập trung vào công việc của mình."
    # ) # 0.6

    # image_path = 'eval-data/images/000000270244.jpg'
    # prompt = get_prompt(
    #     question='Tại sao cảnh này lại kỳ lạ?',
    #     ground_truth='Trong ảnh, có một con ngựa vằn đứng một mình trong khu rừng xanh. Điều này kỳ lạ vì ngựa vằn thường được tìm thấy ở đồng cỏ hoặc thảo nguyên, chứ không phải trong rừng. Ngựa vằn cũng là loài động vật sống theo bầy đàn, nên việc nhìn thấy một cá thể ngựa vằn đơn độc cũng rất bất thường. Hơn nữa, ngựa vằn trong ảnh có vẻ như đang đứng rất thoải mái, không sợ hãi trước môi trường xung quanh. Điều này cho thấy rằng con ngựa vằn không phải là một động vật hoang dã, mà có thể là một con ngựa được thuần hóa hoặc thậm chí là một con vật nuôi đi lạc.',
    #     prediction="Trong ảnh, một con ngựa vằn đang đứng trên một cánh đồng cỏ xanh tươi, xung quanh là những hàng cây. Tuy nhiên, điều kỳ lạ là con ngựa vằn có vẻ như đang đứng trên một cánh đồng cỏ khô, không có cỏ xanh tươi. Điều này tạo nên sự tương phản bất thường giữa con ngựa vằn và môi trường xung quanh, khiến người xem phải suy nghĩ về lý do tại sao con vật lại ở trong tình trạng như vậy."
    # ) # 0.3

    # image_path = 'eval-data/images/000000034873.jpg'
    # prompt = get_prompt(
    #     question='Trong căn phòng này, có bao nhiêu người ngồi được thoải mái?',
    #     ground_truth='Có hai chiếc ghế được đặt ở phía trước bàn và hai chiếc ghế khác đặt ở hai đầu của bàn. Mỗi chiếc ghế có thể thoải mái chứa một người nên căn phòng này có thể chứa tối đa bốn người ngồi thoải mái.',
    #     prediction="Trong ảnh, có một người ngồi trên ghế ở phía bên phải của quầy bar. Người này có thể đang ngồi ở đó để thưởng thức đồ uống hoặc đồ ăn nhẹ, hoặc họ có thể đang chờ đợi ai đó. Không có người nào khác được nhìn thấy trong ảnh, vì vậy không thể xác định được có bao nhiêu người ngồi được thoải mái trong phòng."
    # ) # 0.3

    
    # image_path = 'eval-data/images/000000270244.jpg'
    # prompt = get_prompt(
    #     question='Tại sao cảnh này lại kỳ lạ?',
    #     ground_truth='Trong ảnh, có một con ngựa vằn đứng một mình trong khu rừng xanh. Điều này kỳ lạ vì ngựa vằn thường được tìm thấy ở đồng cỏ hoặc thảo nguyên, chứ không phải trong rừng. Ngựa vằn cũng là loài động vật sống theo bầy đàn, nên việc nhìn thấy một cá thể ngựa vằn đơn độc cũng rất bất thường. Hơn nữa, ngựa vằn trong ảnh có vẻ như đang đứng rất thoải mái, không sợ hãi trước môi trường xung quanh. Điều này cho thấy rằng con ngựa vằn không phải là một động vật hoang dã, mà có thể là một con ngựa được thuần hóa hoặc thậm chí là một con vật nuôi đi lạc.',
    #     prediction="Trong ảnh, có một con ngựa vằn đứng một mình trong khu rừng xanh. Điều này kỳ lạ vì ngựa vằn thường được tìm thấy ở đồng cỏ hoặc thảo nguyên, chứ không phải trong rừng. Ngựa vằn cũng là loài động vật sống theo bầy đàn, nên việc nhìn thấy một cá thể ngựa vằn đơn độc cũng rất bất thường. Hơn nữa, ngựa vằn trong ảnh có vẻ như đang đứng rất thoải mái, không sợ hãi trước môi trường xung quanh. Điều này cho thấy rằng con ngựa vằn không phải là một động vật hoang dã, mà có thể là một con ngựa được thuần hóa hoặc thậm chí là một con vật nuôi đi lạc."
    # ) # 0.9

    output = model.get_response(image_path, prompt)
    print("Evaluate result:", output)

    # evaluate on mm-vet
    # evaluate_on_mmvet(args, model)