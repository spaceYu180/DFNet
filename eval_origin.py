import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data

from main_model import EnsembleModel
from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
running_metrics =  RunningMetrics(2)
import csv

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

np.seterr(divide='ignore', invalid='ignore')
col = 'Ours'
csv_path = 'LEVIR-CD.csv'

data_list = [col]
fp_csv = open(csv_path, 'w', newline='', encoding='utf-8-sig')
writer = csv.writer(fp_csv)
writer.writerow(data_list)

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
    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=448):
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
            print(name)
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
                tn = ((cd_pred == 0) & (label == 0)).int().sum().cpu().numpy()
                fp = ((cd_pred == 1) & (label == 0)).int().sum().cpu().numpy()
                fn = ((cd_pred == 0) & (label == 1)).int().sum().cpu().numpy()
                tp = ((cd_pred == 1) & (label == 1)).int().sum().cpu().numpy()
                
                c_matrix['tn'] += tn
                c_matrix['fp'] += fp
                c_matrix['fn'] += fn
                c_matrix['tp'] += tp
                tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
 
                if(count == 1):
                    count = 0
                    continue
                
                if(count == 0):
                    P = tp / (tp + fp)
                    R = tp / (tp + fn)
                    F1 = 2 * P * R / (R + P)
                    IOU_0 = tn/(tn+fp+fn)
                    IOU_1 = tp/(tp+fp+fn)
                    mIOU = (IOU_0+IOU_1)/2
                    OA = (tp+tn)/(tp+fp+tn+fn)
                    p0 = OA
                    pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn)**2
                    Kappa = (p0-pe)/(1-pe)
                    arr.append(P)
                    # arr.append(name)
                    writer.writerow(arr)
                count += 1

    p, r, f1, miou, iou_0, iou_1, oa, kappa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)
    score = running_metrics.get_scores()
    fp_csv.close()
    return p, r, f1, miou, oa, avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')

    # 配置测试参数
    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "./runs/train/1/best_ckp/",
                        ])

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/LEVIR-CD/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--tta", type=bool, default=False)
    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
