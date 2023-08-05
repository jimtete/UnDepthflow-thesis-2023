import numpy as np
import tensorflow as tf
import scipy.misc as sm

from eval.evaluate_depth import load_depths, eval_depth
from eval.evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg
from eval.evaluate_mask import eval_mask
from enum import Enum

from eval.evaluate_disp import eval_disp_avg
from eval.pose_evaluation_utils import pred_pose
from eval.eval_pose import eval_snippet, kittiEvalOdom
from PIL import Image

import re, os
import sys

from tensorflow.python.platform import flags
opt = flags.FLAGS

class model_type(Enum):
    hitnet = -1
    undepthflow = -2

class train_test(Enum):
    train = -202
    test = -205


def test(sess, eval_model, itr, gt_flows_2012, noc_masks_2012, gt_flows_2015,
         noc_masks_2015, gt_masks):

    # Setup evaluation pipeline.
    custom = 0
    model = model_type.undepthflow
    data_type = train_test.train


    print('Entered test file')
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        if opt.eval_pose != "":
            seqs = opt.eval_pose.split(",")
            odom_eval = kittiEvalOdom("./pose_gt_data/")
            odom_eval.eval_seqs = seqs
            pred_pose(eval_model, opt, sess, seqs)

            for seq_no in seqs:
                sys.stderr.write("pose seq %s: \n" % seq_no)
                eval_snippet(
                    os.path.join(opt.trace, "pred_poses", seq_no),
                    os.path.join("./pose_gt_data/", seq_no))
            odom_eval.eval(opt.trace + "/pred_poses/")
            sys.stderr.write("pose_prediction_finished \n")
        for eval_data in ["kitti_2012", "kitti_2015"]:
            test_result_disp, test_result_flow_rigid, test_result_flow_optical, \
            test_result_mask, test_result_disp2, test_image1 = [], [], [], [], [], []

            if eval_data == "kitti_2012":
                total_img_num = 194
                gt_dir = opt.gt_2012_dir
            else:
                total_img_num = 200
                gt_dir = opt.gt_2015_dir

            for i in range(total_img_num):

                    # print("saved image: " + title)
                    # im.save(title)

                    # if (custom == 1):
                    #     print("Press Ctrl + C")
                    #     continue

                    test_result_flow_rigid.append(np.squeeze(pred_flow_rigid))
                    test_result_flow_optical.append(np.squeeze(pred_flow_optical))
                    test_result_disp.append(np.squeeze(pred_disp))
                    test_result_disp2.append(np.squeeze(pred_disp2))
                    test_result_mask.append(np.squeeze(pred_mask))
                    test_image1.append(img1_orig)

                if (eval_data == "kitti_2012"):
                    eval_kitti_year = "2012"
                else:
                    eval_kitti_year = "2015"

                if (model == model_type.hitnet):
                    img_path = "predictions/hitnet/training/"+eval_kitti_year+"/"+str(i).zfill(6) + ".png"
                    img = Image.open(img_path).resize((opt.img_height, opt.img_width))
                    img = img.convert('L')
                    image_array = np.array(img).astype(np.float32)
                    numpy_image = image_array / 256.0

                    test_result_disp.append(np.squeeze(numpy_image))

            if (data_type == train_test.test):
                continue

            ## depth evaluation
            if opt.eval_depth and eval_data == "kitti_2015":
                print("Evaluate depth at iter [" + str(itr) + "] " + eval_data)
                gt_depths, pred_depths, gt_disparities, pred_disp_resized = load_depths(
                    test_result_disp, gt_dir, eval_occ=True)
                abs_rel, sq_rel, rms, log_rms, a1, a2, a3, d1_all = eval_depth(
                    gt_depths, pred_depths, gt_disparities, pred_disp_resized)
                sys.stderr.write(
                    "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".
                    format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all',
                           'a1', 'a2', 'a3'))
                sys.stderr.write(
                    "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".
                    format(abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3))

                disp_err = eval_disp_avg(
                    test_result_disp,
                    gt_dir,
                    disp_num=0,
                    moving_masks=gt_masks)
                sys.stderr.write("disp err 2015 is \n")
                sys.stderr.write(disp_err + "\n")

                if opt.mode == "depthflow":
                    disp_err = eval_disp_avg(
                        test_result_disp2,
                        gt_dir,
                        disp_num=1,
                        moving_masks=gt_masks)
                    sys.stderr.write("disp2 err 2015 is \n")
                    sys.stderr.write(disp_err + "\n")

            if opt.eval_depth and eval_data == "kitti_2012":
                disp_err = eval_disp_avg(test_result_disp, gt_dir)
                sys.stderr.write("disp err 2012 is \n")
                sys.stderr.write(disp_err + "\n")

            # flow evaluation
            if model == model_type.undepthflow and opt.eval_flow and eval_data == "kitti_2012":
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                        test_result_flow_rigid, opt)
                    sys.stderr.write("epe 2012 rigid is \n")
                    sys.stderr.write(epe + "\n")

                epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                    test_result_flow_optical, opt)
                sys.stderr.write("epe 2012 optical is \n")
                sys.stderr.write(epe + "\n")

            if model == model_type.undepthflow and opt.eval_flow and eval_data == "kitti_2015":
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(
                        gt_flows_2015,
                        noc_masks_2015,
                        test_result_flow_rigid,
                        opt,
                        moving_masks=gt_masks)
                    sys.stderr.write("epe 2015 rigid is \n")
                    sys.stderr.write(epe + "\n")

                epe = eval_flow_avg(
                    gt_flows_2015,
                    noc_masks_2015,
                    test_result_flow_optical,
                    opt,
                    moving_masks=gt_masks)
                sys.stderr.write("epe 2015 optical is \n")
                sys.stderr.write(epe + "\n")

            # mask evaluation
            if model == model_type.undepthflow and  opt.eval_mask and eval_data == "kitti_2015":
                mask_err = eval_mask(test_result_mask, gt_masks, opt)
                sys.stderr.write("mask_err is %s \n" % str(mask_err))
