import os
import cv2
from utils.head_pose_proc import PoseEstimator
import numpy as np
from utils.proc_vid import parse_vid


def exam_video(args, vid_path, face_inst, classifier, scaler):

    imgs, frame_num, fps, width, height = parse_vid(vid_path)
    pose_estimate = PoseEstimator([height, width])
    cumulative_prob = 0.0

    video_optout = False
    optout_counter = 0
    img_counter = 0
    for im in imgs:
        proba, optout = examine_a_frame(args, im, face_inst, classifier, scaler, pose_estimate)
        if optout:
            optout_counter += 1
        else:
            cumulative_prob += proba
            img_counter += 1
    return cumulative_prob/(img_counter * 1.0), video_optout



def exam_img(args, path, face_inst, classifier, scaler):
    # load image and get image height and width
    img = cv2.imread(os.path.join(path))
    height, width = img.shape[0:2]
    # initialize head pose estimator
    pose_estimate = PoseEstimator([height, width])
    prob = examine_a_frame(args, img, face_inst, classifier, scaler, pose_estimate)

    return prob




def examine_a_frame(args, img, face_inst, classifier, scaler, pose_estimator):
    face_boxes = face_inst.get_all_face_rects(img)
    optout = False
    if face_boxes is None:
        # fail to detect a face, opt out
        prob = 0
        optout = True
        return prob, optout

    all_landmarks = face_inst.get_landmarks_all_faces(img, face_boxes)

    if len(face_boxes) != len(all_landmarks):
        # if there's a face detected, but cannot find landmarks on that face, return 1.0
        return 1.0
    max_prob = 0.0
    for landmark in all_landmarks:
        prob = examine_a_face(args, landmark, classifier, scaler, pose_estimator)
        max_prob = max(prob, max_prob)
    return max_prob, optout

def examine_a_face(args, landmarks, classifier, scaler, pose_estimator):
    # extract head pose
    R_c, t_c = None, None
    R_a, t_a = None, None
    R_c_matrix, R_a_matrix = None, None


    R_c, t_c = pose_estimator.solve_single_pose(landmarks, args.markID_c)
    R_a, t_a = pose_estimator.solve_single_pose(landmarks, args.markID_a)
    R_c_matrix = pose_estimator.Rodrigues_convert(R_c)
    R_a_matrix = pose_estimator.Rodrigues_convert(R_a)

    rotation_matrix_feature = (R_c_matrix - R_a_matrix).flatten()
    translation_vector_feature = (t_c - t_a)[:, -1]
    feature = np.concatenate([rotation_matrix_feature, translation_vector_feature]).reshape(1, -1)
    scaled_feature = scaler.transform(feature)
    score = classifier.predict_proba(scaled_feature)

    return score[0][-1]