import json
import cv2
import os
import argparse
import numpy as np
from glob import glob
from torchvision import transforms

import torch
from torch.backends import cudnn

from backbone import HybridNetsBackbone
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from utils.constants import *


def parse_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')

    parser.add_argument('-p', '--project_config', type=str, default='project_config',
                        help='Project file that contains parameters and file format must yaml/yml.')
    parser.add_argument('-t', '--task', type=str, default='boxy', help='tian chi')
    parser.add_argument('-d', '--debug', type=boolean_string, default=True, help='debug project')
    parser.add_argument('-bb', '--backbone', type=str,
                        help='Use timm to create another backbone replacing efficientnet. '
                             'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
    parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
    parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
    parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
    parser.add_argument('--imshow', type=boolean_string, default=False,
                        help="Show result onscreen (unusable on colab, jupyter...)")
    parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
    parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
    parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
    parser.add_argument('--speed_test', type=boolean_string, default=False,
                        help='Measure inference latency')
    args = parser.parse_args()
    return args


def load_images(args):
    root = args.source
    # args.task values are "boxy" and "llamas"
    task_root = os.path.join(root, args.task)

    imgs_dict = {}  # {"folder_name": [xxx/xx.png]}
    for folder_name in os.listdir(task_root):
        dir = os.path.join(task_root, folder_name)
        imgs = glob(f'{dir}/*.jpg') + glob(f'{dir}/*.png')
        imgs_dict[folder_name] = imgs

    return imgs_dict


def get_output_dir(args):
    output = args.output
    if output.endswith("/"):
        output = output[:-1]
    os.makedirs(output, exist_ok=True)

    return output


def get_params(args):
    if not args.project_config:
        return Params(f'projects/{args.project}.yml')

    return Params(args.project_config)


def get_color_list_seg(seg_list):
    color_list_seg = {}
    for seg_class in seg_list:
        # edit your color here if you wanna fix to your liking
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))


def get_color_list():
    return standard_to_bgr(STANDARD_COLORS)


def get_seg_mode(params, weight):
    weight_last_layer_seg = weight.get('model', weight)['segmentation_head.0.weight']
    if weight_last_layer_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        if params.seg_multilabel:
            seg_mode = MULTILABEL_MODE
        else:
            seg_mode = MULTICLASS_MODE

    return seg_mode


def get_model(args, params, weight):
    seg_mode = get_seg_mode(params, weight)
    print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)

    model = HybridNetsBackbone(compound_coef=args.compound_coef, num_classes=len(params.obj_list),
                               ratios=eval(params.anchors_ratios),
                               scales=eval(params.anchors_scales), seg_classes=len(params.seg_list),
                               backbone_name=args.backbone,
                               seg_mode=seg_mode)

    model.load_state_dict(weight.get('model', weight))

    model.requires_grad_(False)
    model.eval()

    if args.use_cuda:
        model = model.cuda()
        if args.use_float16:
            model = model.half()

    return model


def set_cudnn():
    cudnn.fastest = True
    cudnn.benchmark = True


def boxy_task(args, params, model, folder_name, img_paths, ori_imgs, shapes, input_tensor):
    result = {}

    with torch.no_grad():
        features, regression, classification, anchors, seg = model(input_tensor)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(input_tensor, anchors, regression, classification, regressBoxes, clipBoxes,
                          args.threshold, args.ariou_threshold)

        for i in range(len(ori_imgs)):
            img_result = {}

            bboxes = []
            scores = []
            out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(int)

                ori_y = shapes[i][0][0]
                ori_x = shapes[i][0][1]
                bboxes.append([y1 / ori_y, x1 / ori_x, y2 / ori_y, x2 / ori_x])

                score = float(out[i]['scores'][j])
                scores.append(score)

                if args.debug:
                    obj = params.obj_list[out[i]['class_ids'][j]]
                    color_list = get_color_list()
                    plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                                 color=color_list[get_index_label(obj, params.obj_list)])

            img_result["detection_boxes"] = bboxes
            img_result["detection_scores"] = scores

            result[f"./{folder_name}/{os.path.basename(img_paths[i])}"] = img_result

            if args.debug and args.imwrite:
                img_path = os.path.join(get_output_dir(args),
                                        f"boxy_predict/{folder_name}/{os.path.basename(img_paths[i])}")
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                cv2.imwrite(img_path, cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))

    return result


def save_lines_to_file(lanes_array, lanes_info_file):
    pass


def llamas_task(args, params, model, folder_name, img_paths, ori_imgs, shapes, input_tensor, seg_mode):
    output_dir = os.path.join(get_output_dir(args), "llamas", folder_name)
    with torch.no_grad():
        features, regression, classification, anchors, seg = model(input_tensor)
        get_seg_mode()

        seg_mask_list = []
        # (B, C, W, H) -> (B, W, H)
        if seg_mode == BINARY_MODE:
            seg_mask = torch.where(seg >= 0.5, 1, 0)
            seg_mask.squeeze_(1)
            seg_mask_list.append(seg_mask)
        elif seg_mode == MULTICLASS_MODE:
            _, seg_mask = torch.max(seg, 1)
            seg_mask_list.append(seg_mask)
        else:
            seg_mask_list = [torch.where(seg[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
            # but remove background class from the list
            seg_mask_list.pop(0)

        # (B, W, H) -> (W, H)
        for i in range(seg.size(0)):
            for seg_class_index, seg_mask in enumerate(seg_mask_list):
                seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                pad_h = int(shapes[i][1][1][1])
                pad_w = int(shapes[i][1][1][0])
                seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0] - pad_h, pad_w:seg_mask_.shape[1] - pad_w]
                seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)

                lanes_array = np.where(seg_mask_ == 2)
                lanes_array = np.dstack((lanes_array[0], lanes_array[1])).squeeze()

                img_name = os.path.basename(img_paths[i])
                lanes_info_file = os.path.join(output_dir, f'{img_name.split("_color")[0]}.lines.txt')
                save_lines_to_file(lanes_array, lanes_info_file)

                if args.debug:
                    color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                    color_list_seg = get_color_list_seg(params.seg_list)

                    for index, seg_class in enumerate(params.seg_list):
                        color_seg[seg_mask_ == index + 1] = color_list_seg[seg_class]
                    color_seg = color_seg[..., ::-1]  # RGB -> BGR
                    color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background

                    seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[
                        i]  # do not work on original images if MULTILABEL_MODE
                    seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                    seg_img = seg_img.astype(np.uint8)

                    img_path = os.path.join(get_output_dir(args),
                                            f"llamas_predict/{folder_name}/{os.path.basename(img_paths[i])}")
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)

                    if args.show_seg or seg_mode == MULTILABEL_MODE:
                        cv2.imwrite(img_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))


def get_input_tensor(args, params, input_imgs):
    normalize = transforms.Normalize(
        mean=params.mean, std=params.std
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.use_cuda:
        input_tensor = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
    else:
        input_tensor = torch.stack([transform(fi) for fi in input_imgs], 0)
    input_tensor = input_tensor.to(torch.float16 if args.use_cuda and args.use_float16 else torch.float32)

    return input_tensor


def preprocess_imgs(params, ori_imgs):
    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)

    shapes = []
    input_imgs = []
    for ori_img in ori_imgs:
        h0, w0 = ori_img.shape[:2]
        r = resized_shape / max(h0, w0)
        input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]
        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                               scaleup=False)
        input_imgs.append(input_img)
        shapes.append(((h0, w0), ((h / h0, w / w0), pad)))

    return input_imgs, shapes


def predict():
    args = parse_args()
    params = get_params(args)

    output = args.output
    if output.endswith("/"):
        output = output[:-1]
    os.makedirs(output, exist_ok=True)

    imgs_dict = load_images(args)

    set_cudnn()
    weight = torch.load(args.load_weights, map_location='cuda' if args.use_cuda else 'cpu')
    model = get_model(args, params, weight)

    if args.task == "boxy":
        all_result = {}
        for folder_name, img_paths in imgs_dict.items():
            ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_paths]
            ori_imgs = [cv2.resize(i, (1280, 720), interpolation=cv2.INTER_AREA) for i in ori_imgs]
            ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
            print(f"Found boxy/{folder_name} {len(ori_imgs)} images")

            input_imgs, shapes = preprocess_imgs(params, ori_imgs)
            input_tensor = get_input_tensor(args, params, input_imgs)

            result = boxy_task(args, params, model, folder_name, img_paths, ori_imgs, shapes, input_tensor)
            all_result.update(result)

        boxy_json_file = os.path.join(get_output_dir(args), "boxy.json")
        with open(boxy_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_result, f, indent=4, sort_keys=True, ensure_ascii=False)

    if args.task == "llamas":
        for folder_name, img_paths in imgs_dict.items():
            ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_paths]
            ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
            print(f"Found llamas/{folder_name} {len(ori_imgs)} images")

            input_imgs, shapes = preprocess_imgs(params, ori_imgs)
            input_tensor = get_input_tensor(args, params, input_imgs)
            seg_mode = get_seg_mode(params, weight)
            llamas_task(args, params, model, folder_name, img_paths, ori_imgs, shapes, input_tensor, seg_mode)


if __name__ == "__main__":
    predict()
