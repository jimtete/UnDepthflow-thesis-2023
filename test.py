import numpy as np
import sys
import tensorflow as tf
import scipy.misc as sm

from eval.evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg
from eval.evaluate_mask import eval_mask

from eval.evaluate_disp import eval_disp_avg
from eval.pose_evaluation_utils import pred_pose
#from eval.eval_pose import eval_snippet, kittiEvalOdom

import re, os

from tensorflow.python.platform import flags
opt = flags.FLAGS


def test(sess, eval_model, itr, gt_flows_2012, noc_masks_2012, gt_flows_2015,
         noc_masks_2015, gt_masks):
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
                img1 = sm.imread(
                    os.path.join(gt_dir, "image_2",
                                 str(i).zfill(6) + "_10.png"))
                img1_orig = img1
                orig_H, orig_W = img1.shape[0:2]
                img1 = sm.imresize(img1, (opt.img_height, opt.img_width))

                img2 = sm.imread(
                    os.path.join(gt_dir, "image_2",
                                 str(i).zfill(6) + "_11.png"))
                img2 = sm.imresize(img2, (opt.img_height, opt.img_width))

                imgr = sm.imread(
                    os.path.join(gt_dir, "image_3",
                                 str(i).zfill(6) + "_10.png"))
                imgr = sm.imresize(imgr, (opt.img_height, opt.img_width))

                img2r = sm.imread(
                    os.path.join(gt_dir, "image_3",
                                 str(i).zfill(6) + "_11.png"))
                img2r = sm.imresize(img2r, (opt.img_height, opt.img_width))

                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                imgr = np.expand_dims(imgr, axis=0)
                img2r = np.expand_dims(img2r, axis=0)

                calib_file = os.path.join(gt_dir, "calib_cam_to_cam",
                                          str(i).zfill(6) + ".txt")

                input_intrinsic = get_scaled_intrinsic_matrix(
                    calib_file,
                    zoom_x=1.0 * opt.img_width / orig_W,
                    zoom_y=1.0 * opt.img_height / orig_H)

                pred_flow_rigid, pred_flow_optical, \
                pred_disp, pred_disp2, pred_mask= sess.run([eval_model.pred_flow_rigid,
                                                         eval_model.pred_flow_optical,
                                                         eval_model.pred_disp,
                                                         eval_model.pred_disp2,
                                                         eval_model.pred_mask],
                                                          feed_dict = {eval_model.input_1: img1,
                                                                       eval_model.input_2: img2,
                                                                       eval_model.input_r: imgr,
                                                                       eval_model.input_2r:img2r,
                                                                       eval_model.input_intrinsic: input_intrinsic})

                test_result_flow_rigid.append(np.squeeze(pred_flow_rigid))
                test_result_flow_optical.append(np.squeeze(pred_flow_optical))
                test_result_disp.append(np.squeeze(pred_disp))
                test_result_disp2.append(np.squeeze(pred_disp2))
                test_result_mask.append(np.squeeze(pred_mask))
                test_image1.append(img1_orig)

            # TODO: Changing eval depth into a hardoded value
            # Old line :             if opt.eval_depth and eval_data == "kitti_2015":
            ## depth evaluation
            if eval_data == "kitti_2015":
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

            # TODO also here:
            # old line:             if opt.eval_depth and eval_data == "kitti_2012":

            if eval_data == "kitti_2012":
                disp_err = eval_disp_avg(test_result_disp, gt_dir)
                sys.stderr.write("disp err 2012 is \n")
                sys.stderr.write(disp_err + "\n")

            # TODO also here:
            # old line: if opt.eval_flow and eval_data == "kitti_2012":
            
            # flow evaluation
            if eval_data == "kitti_2012":
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                        test_result_flow_rigid, opt)
                    sys.stderr.write("epe 2012 rigid is \n")
                    sys.stderr.write(epe + "\n")

                epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                    test_result_flow_optical, opt)
                sys.stderr.write("epe 2012 optical is \n")
                sys.stderr.write(epe + "\n")

            # TODO also here:
            # old line: if opt.eval_flow and eval_data == "kitti_2015":

            if eval_data == "kitti_2015":
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

            # TODO also here:
            # if opt.eval_mask and eval_data == "kitti_2015":
            
            # mask evaluation
            if  eval_data == "kitti_2015":
                mask_err = eval_mask(test_result_mask, gt_masks, opt)
                sys.stderr.write("mask_err is %s \n" % str(mask_err))

def load_depths(pred_disp_org, gt_path, eval_occ):
    gt_disparities = load_gt_disp_kitti(gt_path, eval_occ)
    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(
        gt_disparities, pred_disp_org)

    return gt_depths, pred_depths, gt_disparities, pred_disparities_resized

def load_gt_disp_kitti(path, eval_occ):
    gt_disparities = []
    for i in range(200):
        if eval_occ:
            disp = sm.imread(
                path + "/disp_occ_0/" + str(i).zfill(6) + "_10.png", -1)
        else:
            disp = sm.imread(
                path + "/disp_noc_0/" + str(i).zfill(6) + "_10.png", -1)
        disp = disp.astype(np.float32) / 256.0
        gt_disparities.append(disp)
    return gt_disparities

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / (gt))

    sq_rel = np.mean(((gt - pred)**2) / (gt))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
def eval_depth(gt_depths,
               pred_depths,
               gt_disparities,
               pred_disparities_resized,
               min_depth=1e-3,
               max_depth=80):
    num_samples = len(pred_depths)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_depth, pred_depth, mask = process_depth(
            gt_depth, pred_depth, gt_disparities[i], min_depth, max_depth)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
            i] = compute_errors(gt_depth[mask], pred_depth[mask])

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3,
                                    (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(
    ), a2.mean(), a3.mean(), d1_all.mean()

def eval_snippet(pred_dir, gt_dir):
    pred_files = glob(pred_dir + '/*.txt')
    ate_all = []
    for i in range(len(pred_files)):
        gtruth_file = gt_dir + "/" + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    sys.stderr.write("ATE mean: %.4f, std: %.4f \n" %
                     (np.mean(ate_all), np.std(ate_all)))

class kittiEvalOdom():
    # ----------------------------------------------------------------------
    # poses: N,4,4
    # pose: 4,4
    # ----------------------------------------------------------------------
    def __init__(self, gt_dir):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        self.gt_dir = gt_dir

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
        # Each line in the file should follow one of the following structures
        # (1) idx pose(3x4 matrix in terms of 12 numbers)
        # (2) pose(3x4 matrix in terms of 12 numbers)
        # ----------------------------------------------------------------------
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split) == 13)
            for row in xrange(3):
                for col in xrange(4):
                    P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectoryDistances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in xrange(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translationError(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in xrange(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10

        for first_frame in xrange(9, len(poses_gt), self.step_size):
            for i in xrange(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame,
                                                             len_)

                # ----------------------------------------------------------------------
                # Continue if sequence not long enough
                # ----------------------------------------------------------------------
                if last_frame == -1 or not (
                        last_frame in poses_result.keys()) or not (
                            first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
                # compute rotational and translational errors
                # ----------------------------------------------------------------------
                pose_delta_gt = np.dot(
                    np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(
                    np.linalg.inv(poses_result[first_frame]),
                    poses_result[last_frame])
                pose_error = np.dot(
                    np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # ----------------------------------------------------------------------
                # compute speed 
                # ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append(
                    [first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write + "\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num = -1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{:02}".format(seq)
        plt.savefig(
            self.plot_path_dir + "/" + png_title + ".pdf",
            bbox_inches='tight',
            pad_inches=0)
        # plt.show()

    def plotError(self, avg_segment_errs):
        # ----------------------------------------------------------------------
        # avg_segment_errs: dict [100: err, 200: err...]
        # ----------------------------------------------------------------------
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            plot_y.append(avg_segment_errs[len_][0])
        fig = plt.figure()
        plt.plot(plot_x, plot_y)
        plt.show()

    def computeSegmentErr(self, seq_errs):
        # ----------------------------------------------------------------------
        # This function calculates average errors for different segment.
        # ----------------------------------------------------------------------

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []
        # ----------------------------------------------------------------------
        # Get errors
        # ----------------------------------------------------------------------
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        # ----------------------------------------------------------------------
        # Compute average
        # ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def eval(self, result_dir):
        error_dir = result_dir + "/errors"
        self.plot_path_dir = result_dir + "/plot_path"
        plot_error_dir = result_dir + "/plot_error"

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)
        if not os.path.exists(plot_error_dir):
            os.makedirs(plot_error_dir)

        total_err = []

        ave_t_errs = []
        ave_r_errs = []

        for seq in self.eval_seqs:
            self.cur_seq = seq
            file_name = seq + ".txt"

            poses_result = self.loadPoses(result_dir + "/" + file_name)
            poses_gt = self.loadPoses(self.gt_dir + "/" + file_name)
            self.result_file_name = result_dir + file_name

            # ----------------------------------------------------------------------
            # compute sequence errors
            # ----------------------------------------------------------------------
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, error_dir + "/" + file_name)

            # ----------------------------------------------------------------------
            # Compute segment errors
            # ----------------------------------------------------------------------
            avg_segment_errs = self.computeSegmentErr(seq_err)

            # ----------------------------------------------------------------------
            # compute overall error
            # ----------------------------------------------------------------------
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
            sys.stderr.write("Sequence: " + seq + "\n")
            sys.stderr.write("Average translational RMSE (%%): %.2f \n" %
                             (ave_t_err * 100))
            sys.stderr.write("Average rotational error (deg/100m): %.2f \n" %
                             (ave_r_err / np.pi * 180 * 100))
            ave_t_errs.append(ave_t_err)
            ave_r_errs.append(ave_r_err)