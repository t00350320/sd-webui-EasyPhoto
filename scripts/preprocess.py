import argparse
import json
import logging
import math
import os
import platform
import sys
from shutil import copyfile

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "easyphoto_utils"))

import cv2
import numpy as np
import torch
from face_process_utils import call_face_crop
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=("The validation_prompt of the user."),
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help=("The ref_image_path."),
    )
    parser.add_argument(
        "--images_save_path",
        type=str,
        default=None,
        help=("The images_save_path."),
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default=None,
        help=("The json_save_path."),
    )
    parser.add_argument(
        "--inputs_dir",
        type=str,
        default=None,
        help=("The inputs dir of the data for preprocessing."),
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=3,
        help=("The crop ratio of the data for scene lora preprocessing."),
    )
    parser.add_argument(
        "--skin_retouching_bool",
        action="store_true",
        help=("Whether to use beauty"),
    )
    parser.add_argument(
        "--train_scene_lora_bool",
        action="store_true",
        help=("Whether to train scene lora"),
    )
    args = parser.parse_args()
    return args


def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # Take mean from the user image to obtain the average features of the real person image
    pivot_feature = np.mean(embedding_array, axis=0)
    pivot_feature = np.reshape(pivot_feature, [512, 1])

    # Sort the images in a folder that are closest to the median value
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores

def get_mask_head(result, need_hair = True):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    h, w = np.shape(masks[0])
    mask_hair = np.zeros((h, w))
    mask_face = np.zeros((h, w))
    mask_human = np.zeros((h, w))
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    
    #crop hair eara
    #print(mask_face)
    indices = np.transpose(np.where(mask_face == 1))
    points = indices.tolist()
    #print(points)
    y_values = [point[0] for point in points]
    min_y = min(y_values)
    max_y = max(y_values)
    expand_y = (max_y - min_y) // 20
    #mask_hair[(max_y + expand_y):, :] = 0
    #mask_hair[max_y:, :] = 0

    if need_hair:
        mask_head = np.clip(mask_hair + mask_face, 0, 1)
    else:
        mask_head = mask_face
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    mask_head = np.zeros((h, w)).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2)
    return mask_head

if __name__ == "__main__":
    args = parse_args()
    images_save_path = args.images_save_path
    json_save_path = args.json_save_path
    validation_prompt = args.validation_prompt
    inputs_dir = args.inputs_dir
    ref_image_path = args.ref_image_path
    skin_retouching_bool = args.skin_retouching_bool
	facecrop_method     = "fc"
    train_scene_lora_bool = args.train_scene_lora_bool

    logging.info(
        f"""
        preprocess params:
        images_save_path     = {images_save_path}
        json_save_path       = {json_save_path}
        validation_prompt    = {validation_prompt}
        inputs_dir           = {inputs_dir}
        ref_image_path       = {ref_image_path}
        skin_retouching_bool = {skin_retouching_bool}
        train_scene_lora_bool = {train_scene_lora_bool}
        """
    )
    # embedding
    face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")
    # face detection
    retinaface_detection = pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface", model_revision="v2.0.2")
    # semantic segmentation
    # salient_detect          = pipeline(Tasks.semantic_segmentation, 'damo/cv_u2net_salient-detection', model_revision='v1.0.0')
    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing', model_revision='v1.0.1')
    fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition, 'damo/cv_resnet34_face-attribute-recognition_fairface', model_revision='v2.0.2')
    facial_landmark_confidence_func = pipeline(Tasks.face_2d_keypoints, 'damo/cv_manual_facial-landmark-confidence_flcm', model_revision='v2.5')

    # skin retouching
    try:
        skin_retouching = pipeline("skin-retouching-torch", model="damo/cv_unet_skin_retouching_torch", model_revision="v1.0.2")
    except Exception as e:
        skin_retouching = None
        logging.info(f"Skin Retouching model load error, but pass. Error info {e}")
    # portrait enhancement
    try:
        portrait_enhancement = pipeline(
            Tasks.image_portrait_enhancement, model="damo/cv_gpen_image-portrait-enhancement", model_revision="v1.0.0"
        )
    except Exception as e:
        portrait_enhancement = None
        logging.info(f"Portrait Enhancement model load error, but pass. Error info {e}")

    if not train_scene_lora_bool:
        # jpg list
        jpgs = os.listdir(inputs_dir)
        # ---------------------------FaceID score calculate-------------------------- #
        face_id_scores = []
        face_angles = []
        copy_jpgs = []
        selected_paths = []
        sub_images = []
        genders = []
        for index, jpg in enumerate(tqdm(jpgs)):
            try:
                if not jpg.lower().endswith((".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                    continue
                _image_path = os.path.join(inputs_dir, jpg)
                image = Image.open(_image_path)

                h, w, c = np.shape(image)

                retinaface_boxes, retinaface_keypoints, _ = call_face_crop(retinaface_detection, image, 3, prefix="tmp")
                retinaface_box = retinaface_boxes[0]
                retinaface_keypoint = retinaface_keypoints[0]

                # get key point
                retinaface_keypoint = np.reshape(retinaface_keypoint, [5, 2])
                # get angle
                x = retinaface_keypoint[0, 0] - retinaface_keypoint[1, 0]
                y = retinaface_keypoint[0, 1] - retinaface_keypoint[1, 1]
                angle = 0 if x == 0 else abs(math.atan(y / x) * 180 / math.pi)
                angle = (90 - angle) / 90

                # face size judge
                face_width = (retinaface_box[2] - retinaface_box[0]) / (3 - 1)
                face_height = (retinaface_box[3] - retinaface_box[1]) / (3 - 1)
                if min(face_width, face_height) < 128:
                    print("Face size in {} is small than 128. Ignore it.".format(jpg))
                    continue

                # face crop
                sub_image = image.crop(retinaface_box)
	            # sub image too large
	            w, h = sub_image.size
	            if max(w ,h) > 3072:
	                sub_image = sub_image.resize((w//2,h//2),Image.ANTIALIAS)
	            tmp_path = os.path.join(images_save_path, 'tmp.png')
	            sub_image.save(tmp_path)
	            # gender detect
	            attribute_result = fair_face_attribute_func(tmp_path)
	            score_gender = np.array(attribute_result['scores'][0])
	            gender = np.argmax(score_gender)
	            genders.append(gender)
	            if gender == 0 or skin_retouching_bool:
                    try:
                        sub_image = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        torch.cuda.empty_cache()
                        logging.error(f"Photo skin_retouching error, error info: {e}")

                # get embedding
                embedding = face_recognition(dict(user=image))[OutputKeys.IMG_EMBEDDING]

                face_id_scores.append(embedding)
                face_angles.append(angle)

                copy_jpgs.append(jpg)
                selected_paths.append(_image_path)
                sub_images.append(sub_image)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Photo detect and count score error, error info: {e}")

        # Filter reference faces based on scores, considering quality scores, similarity scores, and angle scores
        face_id_scores = compare_jpg_with_face_id(face_id_scores)
        ref_total_scores = np.array(face_angles) * np.array(face_id_scores)
        ref_indexes = np.argsort(ref_total_scores)[::-1]
        for index in ref_indexes:
            print("selected paths:", selected_paths[index], "total scores: ", ref_total_scores[index], "face angles", face_angles[index])
        copyfile(selected_paths[ref_indexes[0]], ref_image_path)

        # Select faces based on scores, considering similarity scores
        total_scores = np.array(face_id_scores)
        indexes = np.argsort(total_scores)[::-1][:15]

        selected_jpgs = []
        selected_scores = []
        selected_sub_images = []
        for index in indexes:
            selected_jpgs.append(copy_jpgs[index])
            selected_scores.append(ref_total_scores[index])
            selected_sub_images.append(sub_images[index])
            print("jpg:", copy_jpgs[index], "face_id_scores", ref_total_scores[index])

        images = []
        enhancement_num = 0
        max_enhancement_num = len(selected_jpgs) // 2
        for index, jpg in tqdm(enumerate(selected_jpgs[::-1])):
            try:
                sub_image = selected_sub_images[index]
                try:
                    if (np.shape(sub_image)[0] < 512 or np.shape(sub_image)[1] < 512) and enhancement_num < max_enhancement_num:
                        sub_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                        enhancement_num += 1
                except Exception as e:
                    torch.cuda.empty_cache()
                    logging.error(f"Photo enhance error, error info: {e}")

	            if facecrop_method == "ep":
	                # Correct the mask area of the face
	                sub_boxes, _, sub_masks = call_face_crop(retinaface_detection, sub_image, 1, prefix="tmp")
	                sub_box = sub_boxes[0]
	                sub_mask = sub_masks[0]

	                h, w, c = np.shape(sub_mask)
	                face_width = sub_box[2] - sub_box[0]
	                face_height = sub_box[3] - sub_box[1]
	                sub_box[0] = np.clip(np.array(sub_box[0], np.int32) - face_width * 0.3, 1, w - 1)
	                sub_box[2] = np.clip(np.array(sub_box[2], np.int32) + face_width * 0.3, 1, w - 1)
	                sub_box[1] = np.clip(np.array(sub_box[1], np.int32) + face_height * 0.15, 1, h - 1)
	                sub_box[3] = np.clip(np.array(sub_box[3], np.int32) + face_height * 0.15, 1, h - 1)
	                sub_mask = np.zeros_like(np.array(sub_mask, np.uint8))
	                sub_mask[sub_box[1] : sub_box[3], sub_box[0] : sub_box[2]] = 1

	                # Significance detection, merging facial masks
	                result = salient_detect(sub_image)[OutputKeys.MASKS]
	                mask = np.float32(np.expand_dims(result > 128, -1)) * sub_mask

	                # Obtain the image after the mask
	                mask_sub_image = np.array(sub_image) * np.array(mask) + np.ones_like(sub_image) * 255 * (1 - np.array(mask))
	                mask_sub_image = Image.fromarray(np.uint8(mask_sub_image))
	            elif facecrop_method == "fc":
	                sub_image.save(tmp_path)
	                tmp2_path = os.path.join(images_save_path, 'tmp2.png') 
	                result = segmentation_pipeline(tmp_path)
	                #print("result is", result)
	                #print(np.shape(result["masks"][0]))
	                need_hair = True if genders[index] == 0 else False
	                mask = get_mask_head(result, need_hair)
	                im = cv2.imread(tmp_path)
	                mask_image = im * mask + 255 * (1 - mask)
	                raw_result = facial_landmark_confidence_func(mask_image)
	                if raw_result is None:
	                    print('landmark quality fail...')
	                    continue

	                #print("score is", raw_result['scores'][0])
	                if float(raw_result['scores'][0]) < (1 - 0.145):
	                    print('landmark quality fail...')
	                    continue
	                #cv2.imwrite(tmp2_path, mask_image)
	                #print(mask_image.shape)
	                mask_sub_image = Image.fromarray(cv2.cvtColor(np.uint8(mask_image),cv2.COLOR_BGR2RGB))
	            if np.sum(np.array(mask)) != 0:
	                    images.append(mask_sub_image)
	    	except Exception as e:
	                torch.cuda.empty_cache()
					logging.error(f"Photo face crop and salient_detect error, error info: {e}")

	    try:
	        os.remove(tmp_path)
	    except OSError as e:
	        print(f"Failed to remove path {tmp_path}: {e}")
    else:
        # jpg list
        jpgs = os.listdir(inputs_dir)
        images = []
        for index, jpg in enumerate(tqdm(jpgs)):
            try:
                if not jpg.lower().endswith((".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                    continue
                _image_path = os.path.join(inputs_dir, jpg)
                image = Image.open(_image_path)

                h, w, c = np.shape(image)

                retinaface_boxes, retinaface_keypoints, _ = call_face_crop(retinaface_detection, image, 1, prefix="tmp")
                retinaface_box = retinaface_boxes[0]
                retinaface_keypoint = retinaface_keypoints[0]

                face_width = retinaface_box[2] - retinaface_box[0]
                face_height = retinaface_box[3] - retinaface_box[1]

                crop_ratio = float(args.crop_ratio)
                retinaface_box[0] = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
                retinaface_box[1] = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 4, 0, h - 1)
                retinaface_box[2] = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
                retinaface_box[3] = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 4 * 3, 0, h - 1)

                # Calculate the left, top, right, bottom of all faces now
                left, top, right, bottom = retinaface_box
                # Calculate the width, height, center_x, and center_y of all faces, and get the long side for rec
                width, height, center_x, center_y = [right - left, bottom - top, (left + right) / 2, (top + bottom) / 2]
                long_side = min(width, height)

                # Calculate the new left, top, right, bottom of all faces for clipping
                # Pad the box to square for saving GPU memomry
                left, top = int(np.clip(center_x - long_side // 2, 0, w - 1)), int(np.clip(center_y - long_side // 2, 0, h - 1))
                right, bottom = int(np.clip(left + long_side, 0, w - 1)), int(np.clip(top + long_side, 0, h - 1))
                retinaface_box = [left, top, right, bottom]

                # face crop
                sub_image = image.crop(retinaface_box)
                if skin_retouching_bool:
                    try:
                        sub_image = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        torch.cuda.empty_cache()
                        logging.error(f"Photo skin_retouching error, error info: {e}")

                try:
                    if np.shape(sub_image)[0] < 768 or np.shape(sub_image)[1] < 768:
                        sub_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                except Exception as e:
                    torch.cuda.empty_cache()
                    logging.error(f"Photo enhance error, error info: {e}")

                images.append(sub_image)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Photo detect and count score error, error info: {e}")

        if len(images) > 0:
            target_size = (224, 224)
            image = images[0]

            # calculate resize border
            width, height = image.size
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width = int(target_size[0] * aspect_ratio)
                image = image.resize((new_width, target_size[1]))
            else:
                new_height = int(target_size[1] / aspect_ratio)
                image = image.resize((target_size[0], new_height))

            # crop a 224x224 photo for goodlook
            width, height = image.size
            left = (width - target_size[0]) / 2
            top = (height - target_size[1]) / 2
            right = (width + target_size[0]) / 2
            bottom = (height + target_size[1]) / 2

            cropped_image = image.crop((left, top, right, bottom))
            image.save(ref_image_path)

    # write results
    for index, base64_pilimage in enumerate(images):
        image = base64_pilimage.convert("RGB")
        image.save(os.path.join(images_save_path, str(index) + ".jpg"))
        print("save processed image to " + os.path.join(images_save_path, str(index) + ".jpg"))
        with open(os.path.join(images_save_path, str(index) + ".txt"), "w") as f:
            f.write(validation_prompt)

    with open(json_save_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(images_save_path, topdown=False):
            for file in files:
                path = os.path.join(root, file)
                if not file.endswith("txt"):
                    txt_path = ".".join(path.split(".")[:-1]) + ".txt"
                    if os.path.exists(txt_path):
                        prompt = open(txt_path, "r").readline().strip()
                        if platform.system() == "Windows":
                            path = path.replace("\\", "/")
                        jpg_path_split = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {"file_name": file_name, "text": prompt}
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")

    del retinaface_detection
    # del salient_detect
    del skin_retouching
    del portrait_enhancement
    del face_recognition
    torch.cuda.empty_cache()
