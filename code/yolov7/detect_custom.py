import argparse
import time
from pathlib import Path

import cv2
import torch
import easyocr
from numpy import random, shape, copy, ascontiguousarray

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



def show_img(winname, img):
    # cv2.namedWindow(winname, cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(winname, img)
    k = 0xff & cv2.waitKey() # wait for keyboard input
    if k == 27: # terminate execution with 'escape'
        exit()
    cv2.destroyAllWindows()


def remove_contained_dets(detections):
    # remove eventual detections which are entirely contained inside another detection
    rows_to_remove = []
    rows = shape(detections)[0]
    for i in range(rows):
        xm, ym, xM, yM = int(detections[i, 0]), int(detections[i, 1]), int(detections[i, 2]), int(detections[i, 3])
        for j in range(shape(detections)[0]):
            if xm > int(detections[j, 0]) and ym > int(detections[j, 1]) and xM < int(detections[j, 2]) and yM < int(detections[j, 3]):
                rows_to_remove.append(i)
                break
    # if there are detections inside other detections, remove them from "detections"
    if len(rows_to_remove):
        for idx in reversed(rows_to_remove):
            detections = torch.cat((detections[:idx, :], detections[idx+1:, :]), axis=0)
    return detections


def prep_img(img, device, half):
    # process the input image to make it a valid input for the .pt model
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def detect(save_img=False):
    source, weights_s, weights_a, view_img, save_txt, imgsz, trace = opt.source, opt.weights_s, opt.weights_a, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize OCR
    reader = easyocr.Reader(['en', 'it'])

    # Initialize detectors
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    model_signs  = attempt_load(weights_s, map_location=device)
    model_arrows = attempt_load(weights_a, map_location=device)
    stride = int(model_signs.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model_signs = TracedModel(model_signs, device, opt.img_size)
        model_arrows = TracedModel(model_arrows, device, opt.img_size)

    if half:
        model_signs.half()  # to FP16
        model_arrows.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset_signs = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names_signs = model_signs.module.names if hasattr(model_signs, 'module') else model_signs.names
    if opt.fixed_colors:
        colors_signs = [[60, 220, 0], [15, 45, 240]]
        color_arrows = [5, 200, 180]
        color_text = [250, 5, 250]
    else:
        colors_signs = [[random.randint(0, 255) for _ in range(3)] for _ in names_signs]
        color_arrows = [random.randint(0, 255) for _ in range(3)]
        color_text = [random.randint(0, 255) for _ in range(3)]

    # Run inferences
    if device.type != 'cpu':
        model_signs(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_signs.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset_signs:
        img = prep_img(img, device, half)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model_signs(img, augment=opt.augment)[0]

        # Run inference to detect signs in input image(s)
        # t1 = time_synchronized()
        pred_signs = model_signs(img, augment=opt.augment)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred_signs = non_max_suppression(pred_signs, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t3 = time_synchronized()

        # Process signs detections
        for i, det_signs in enumerate(pred_signs):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset_signs, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset_signs.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det_signs):
                # Remove eventual signs detections which are completely contained inside other signs detections
                det_signs = remove_contained_dets(det_signs)
                # Rescale boxes from img_size to im0 size
                det_signs[:, :4] = scale_coords(img.shape[2:], det_signs[:, :4], im0.shape).round()

                # Perform arrows detection on these regions of the input image to verify the presence or absence of arrows
                rows = shape(det_signs)[0]
                for i in range(rows):
                    # Get the detected sign patch from the original image and process it to make it a valid input for the model
                    xm, ym, xM, yM = int(det_signs[i, 0]), int(det_signs[i, 1]), int(det_signs[i, 2]), int(det_signs[i, 3])
                    img_sign0 = im0[ym:yM, xm:xM, :] # original sign patch
                    img_sign = letterbox(img_sign0, imgsz, stride)[0]
                    img_sign = img_sign[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HxWxCH to CHxHxW
                    img_sign = ascontiguousarray(img_sign)
                    img_sign = prep_img(img_sign, device, half)

                    # Run inference to detect eventual arrows
                    pred_arrows = model_arrows(img_sign, augment=opt.augment)[0]
                    pred_arrows = non_max_suppression(pred_arrows, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    # Run inference to detect eventual writings
                    pred_writings = reader.readtext(img_sign0)

                    """
                    If one or more arrows and one or more writings are detected:
                        -If the input image patch was detected as a directional sign:
                            OK! Move on;
                        -If the input image patch was detected as a "other" sign:
                            Change the label "other" to "direction-or-information";
                    If no arrows and/or writings are detected:
                        -If the input image patch was detected as a directional sign:
                            Change the label "direction-or-information" to "other";
                        -If the input image patch was detected as a "other" sign:
                            Ok! Move on.
                    """
                    is_directional = not(int(det_signs[i, 5])) # '0' = direction-or-information, '1' = other
                    arrows_check = False
                    writings_check = False

                    # Check for arrows
                    temp = 0 # TODO: remove this line ⚠️
                    for _, det_arrows in enumerate(pred_arrows):
                        if len(det_arrows):
                            arrows_check = True
                            # Remove eventual arrows detections which are completely contained inside other arrows detections
                            # det_arrows = remove_contained_dets(det_arrows)
                            # Rescale boxes from img_sign to img_sign0 size
                            det_arrows[:, :4] = scale_coords(img_sign.shape[2:], det_arrows[:, :4], img_sign0.shape).round()
                            # Draw detection bbox on image and show it
                            for *xyxy, conf, cls in reversed(det_arrows):
                                label = f'arrow {conf:.2f}'
                                # plt_one_box(xyxy, img_sign0, label=label, color=color_arrows, line_thickness=1)
                                arrow_img = img_sign0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :] # TODO: remove this line ⚠️
                                cv2.imwrite(save_path[:-4]+str(temp)+'.jpg', arrow_img) # TODO: remove this line ⚠️
                                temp += 1 # TODO: remove this line ⚠️

                    # Check for writings
                    for writing in pred_writings:
                        if len(writing):
                            writings_check = True
                            text_coords = [writing[0][0][0], writing[0][0][1], writing[0][2][0], writing[0][2][1]] # [xm, ym, xM, yM] of the text detected
                            label = f'{writing[1]} {writing[2]:.2f}'
                            # plot_one_box(text_coords, img_sign0, label=label, color=color_text, line_thickness=1)

                    # Adjust the labels if necessary
                    if not(is_directional) and arrows_check and writings_check:
                        det_signs[i, 5] = 0.0 # 'other' -> 'direction-or-information'
                    elif is_directional and (not(arrows_check) or not(writings_check)):
                        det_signs[i, 5] = 1.0 # 'direction-or-information' -> 'other'


                print(f" Image {p.name} done.\n")

                # Print results
                for c in det_signs[:, -1].unique():
                    n = (det_signs[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names_signs[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det_signs):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names_signs[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors_signs[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                show_img(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset_signs.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')
    print("\nDone.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_s', type=str, default='yolov7.pt', help='model.pt path(s)') # signs detection
    parser.add_argument('--weights_a', type=str, default='yolov7.pt', help='model.pt path(s)') # arrows detection
    parser.add_argument('--fixed-colors', action='store_true', help='use fixed colors instead of random ones')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
