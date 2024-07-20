import os
try :
    from sentence_transformers import SentenceTransformer
except:
    os.system('pip install -U sentence-transformers')
    from sentence_transformers import SentenceTransformer
    

from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import argparse

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
    parser.add_argument('--model_name', type=str, default="hiieu/halong_embedding", help='Model name : dangvantuan/vietnamese-embedding, VoVanPhuc/sup-SimCSE-VietNamese-phobert-base, hiieu/halong_embedding')
    args = parser.parse_args()
    return args

ground_truth='Một người đàn ông mặc tạp dề đầu bếp đang đứng trong bếp, trước một chiếc lò nướng màu đen. Ông đang cầm một chiếc chảo trong một tay và một chiếc thìa trong tay kia. Sau lưng ông là một chiếc bàn có nhiều loại dụng cụ nấu ăn khác nhau. Trên bàn là một số bát, cốc và dụng cụ. Ở phía bên trái của bàn là một chiếc lò nướng khác và một cái chảo lớn hơn.\n\nCạnh bàn là một người khác, mặc áo sơ mi và quần tây. Người này dường như đang quan sát người đầu bếp làm bánh pizza. Trong góc bên phải của khung cảnh, có một bồn rửa lớn với vòi nước.'
prediction="Hình ảnh cho thấy một người đàn ông mặc tạp dề đang đứng trong bếp. Anh ta đang đứng trước một cái bàn, trên đó có nhiều dụng cụ nấu ăn khác nhau. Người đàn ông đang cầm một cái thìa và một cái nĩa, có vẻ như anh ta đang chuẩn bị nấu ăn.\nCó một số vật dụng nhà bếp khác trong bếp, bao gồm một cái bát, một cái cốc và một cái thìa khác. Một cái bát khác có thể được nhìn thấy trên một cái bàn gần đó.\nCó một số vật dụng khác trong bếp, chẳng hạn như một cái lò nướng, một cái lò vi sóng và một số tủ."

args = arg_parser()
similarity = compare_gt_with_pred(ground_truth, prediction, args.model_name)
print(similarity)