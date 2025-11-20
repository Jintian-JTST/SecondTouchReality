
# hand_head_cli.py
# 训练 / 推理命令行工具
# 用法：
#   1) 训练：
#       python hand_head_cli.py train data.csv --epochs 500 --lr 0.01 --out model.json
#   2) 推理（单条）：
#       python hand_head_cli.py infer model.json --s_w 0.8 --s_l 0.7 --curl 0.2 --Zw 35 --Zl 33
#   3) 生成 CSV 模板：
#       python hand_head_cli.py template data_template.csv

import argparse, json, os, sys, csv
import numpy as np

from hand_head_model import TrainConfig, train_fusion_head, save_model, load_model, predict

def cmd_train(args):
    cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, val_split=args.val_split, z_max_cm=args.zmax)
    model, metrics = train_fusion_head(args.csv, cfg)
    print("\n== Metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    extra = {'metrics': metrics, 'cfg': cfg.__dict__}
    save_model(model, args.out, extra)
    print(f"\nSaved model to: {args.out}")

def cmd_infer(args):
    model = load_model(args.model)
    z, a = predict(model, args.s_w, args.s_l, args.curl, args.Zw, args.Zl, z_max_cm=args.zmax)
    print(json.dumps({'Z_pred_cm': float(z), 'alpha_on_Zw': float(a)}, ensure_ascii=False))

def cmd_template(args):
    header = ['s_w','s_l','curl','Zw','Zl','z_gt']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        # 写几行演示（随便填一些）
        w.writerow([0.90, 0.85, 0.10, 35.2, 36.0, 35.8])
        w.writerow([0.70, 0.90, 0.40, 40.0, 34.5, 36.0])
        w.writerow([0.60, 0.75, 0.80, 48.0, 33.0, 39.5])
    print(f"Wrote template CSV with header to: {args.out}")

def main():
    ap = argparse.ArgumentParser(description='Tiny fusion head trainer/inferencer')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_tr = sub.add_parser('train', help='Train from CSV')
    ap_tr.add_argument('csv')
    ap_tr.add_argument('--epochs', type=int, default=400)
    ap_tr.add_argument('--lr', type=float, default=1e-2)
    ap_tr.add_argument('--weight_decay', type=float, default=1e-4)
    ap_tr.add_argument('--val_split', type=float, default=0.2)
    ap_tr.add_argument('--zmax', type=float, default=200.0)
    ap_tr.add_argument('--out', type=str, default='fusion_head_model.json')
    ap_tr.set_defaults(func=cmd_train)

    ap_inf = sub.add_parser('infer', help='Single example inference')
    ap_inf.add_argument('model')
    ap_inf.add_argument('--s_w', type=float, required=True)
    ap_inf.add_argument('--s_l', type=float, required=True)
    ap_inf.add_argument('--curl', type=float, required=True)
    ap_inf.add_argument('--Zw', type=float, required=True)
    ap_inf.add_argument('--Zl', type=float, required=True)
    ap_inf.add_argument('--zmax', type=float, default=200.0)
    ap_inf.set_defaults(func=cmd_infer)

    ap_tpl = sub.add_parser('template', help='Write a CSV template with header rows')
    ap_tpl.add_argument('out')
    ap_tpl.set_defaults(func=cmd_template)

    args = ap.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
