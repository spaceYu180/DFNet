import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
import cv2

from main_model import EnsembleModel
from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
running_metrics =  RunningMetrics(2)
full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (255, 0, 0)}

np.seterr(divide='ignore', invalid='ignore')

def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()
    save_path, result_save_path = check_eval_dirs()
    save_results = SaveResult(result_save_path)
    save_results.prepare()

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)
    
    
    if model.models_list[0].head2 is None:
        opt.dual_label = False
    else:
        opt.dual_label = True
    eval_loader = get_eval_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(model, eval_loader, tta=opt.tta)


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=448):   # 评估,得到指标(metric)
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    tn_fp_fn_tp = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, name) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()
            #print(name)

            
            b, _, h, w = batch_img1.size()
            
            arr = []

            if criterion is not None:
                batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            outs = model(batch_img1, batch_img2, tta)

            if not isinstance(outs, tuple):
                outs = (outs, outs)
            labels = (batch_label1, batch_label2)

            if criterion is not None:
                outs = scale.scale_output(outs)
                val_loss = criterion(outs, labels)
                _, cd_pred1 = torch.max(outs[0], 1)
                _, cd_pred2 = torch.max(outs[1], 1)
            else:
                cd_pred1 = outs[0]
                cd_pred2 = outs[1]

            cd_preds = (cd_pred1, cd_pred2)
            running_metrics.update(labels[0].data.cpu().numpy(),cd_preds[0].data.cpu().numpy())
            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)
            
            
            count = 0
            for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
            
                if(count == 1):
                    count = 0
                    continue
                
                if(count == 0):
                    label = label.data.cpu().numpy()
                    cd_pred = cd_pred.data.cpu().numpy()
                    
                    name = str(name)
                    name = name[2:]
                    name = name[:-3]
                    tp = np.array ((cd_pred == 1) & (label == 1)).astype(np.int8)
                    tn = np.array ((cd_pred == 0) & (label == 0)).astype(np.int8)
                    fp = np.array ((cd_pred == 1) & (label == 0)).astype(np.int8)
                    fn = np.array ((cd_pred == 0) & (label == 1)).astype(np.int8)
                    img = tp*1 + tn*2 + fp*3 + fn*4
                    img_colour = torch.zeros(b, 3, h, w)
                    img_r = torch.zeros(1, h, w)
                    img_g = torch.zeros(1, h, w)
                    img_b = torch.zeros(1, h, w)
                    img = img.reshape(1, 1, h, -1)
    
                    for k, v in full_to_colour.items():
                        img_r[(img == k)] = v[0]
                        img_g[(img == k)] = v[1]
                        img_b[(img == k)] = v[2]
                        img_colour = torch.cat((img_r, img_g, img_b), 0)
                        img_colour=img_colour.data.cpu().numpy()
                        img_colour = np.transpose(img_colour,(1,2,0))

                    cv2.imwrite("LEVIR/"+name, img_colour.astype(np.uint8))
                count += 1

    # plt.show()
    return 1, 2, 3, 4, 5, 6


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')
    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "/mnt/Disk1/DFNet/runs/train/1/best_ckp",
                        ])
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/LEVIR-CD/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--tta", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
