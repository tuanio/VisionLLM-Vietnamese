'''
python evaluate_using_embedding.py --input_json_file path/to/input.json --output_json_file path/to/result.json -model_name hiieu/halong_embedding
'''
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import json

def compare_gt_with_pred(gt, pred, model_name="hiieu/halong_embedding"):
    # Khởi tạo model
    model = SentenceTransformer(model_name)
    
    # Encode ground_truth và prediction
    gt_embedding = model.encode(gt)
    pred_embedding = model.encode(pred)
    
    # Tính toán similarity
    similaritiy = model.similarity(gt_embedding, pred_embedding)
    
    return similaritiy.item()


def arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate embedding similarity')
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
    parser.add_argument('--model_name', type=str, default="hiieu/halong_embedding", help='Model name : dangvantuan/vietnamese-embedding, VoVanPhuc/sup-SimCSE-VietNamese-phobert-base, hiieu/halong_embedding')
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
    model_name = args.model_name
    
    # read json file format same as below
    '''
    [
        {
            "image": "eval-data/images/000000397133.jpg",
            "id_image": "000000397133",
            "question": "Người đầu bếp này đang chuẩn bị món gì?",
            "answer": "Trong ảnh, người đầu bếp đứng trước lò nướng và đang làm bánh. Người đó đang sử dụng các dụng cụ khác nhau, chẳng hạn như chảo và đồ đựng nướng, hỗ trợ cho quá trình này. Bàn cạnh đó bày la liệt các loại bột và nguyên liệu khác, cho thấy người đầu bếp có thể đang làm nhiều món cùng lúc.",
            "prediction": "Trong ảnh, người đầu bếp đang làm bánh. Người đó đang sử dụng các dụng cụ khác nhau, chẳng hạn như chảo và đồ đựng nướng, hỗ trợ cho quá trình này. Bàn cạnh đó bày la liệt các loại bột và nguyên liệu khác, cho thấy người đầu bếp có thể đang làm nhiều món cùng lúc."
        },
        {
            "image": "eval-data/images/000000270244.jpg",
            "id_image": "000000270244",
            "question": "Tại sao cảnh này lại kỳ lạ?",
            "answer": "Trong ảnh, có một con ngựa vằn đứng một mình trong khu rừng xanh. Điều này kỳ lạ vì ngựa vằn thường được tìm thấy ở đồng cỏ hoặc thảo nguyên, chứ không phải trong rừng. Ngựa vằn cũng là loài động vật sống theo bầy đàn, nên việc nhìn thấy một cá thể ngựa vằn đơn độc cũng rất bất thường. Hơn nữa, ngựa vằn trong ảnh có vẻ như đang đứng rất thoải mái, không sợ hãi trước môi trường xung quanh. Điều này cho thấy rằng con ngựa vằn không phải là một động vật hoang dã, mà có thể là một con ngựa được thuần hóa hoặc thậm chí là một con vật nuôi đi lạc.",
            "prediction": "Trong ảnh, có một con ngựa vằn đứng một mình trong khu rừng xanh. Điều này kỳ lạ vì ngựa vằn thường được tìm thấy ở đồng cỏ hoặc thảo nguyên, chứ không phải trong rừng. "
        }
    ]
    '''
    with open(input_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result = []
    for item in data:
        id_image = item['id_image']
        score = compare_gt_with_pred(item['answer'], item['prediction'], model_name)
        list_score = [score]
        result.append({'id_image': id_image, 'list_score': list_score, 'score': score})

    with open(output_json_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
    
    print("Evaluation done!")


