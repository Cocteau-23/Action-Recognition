import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d


skeleton_path = "../sample_datas"
skeleton_file = "S003A001.txt"
gif_save_path = "vis/action1.gif"
file_path = osp.join(skeleton_path, skeleton_file)
num_joints = 32
align_joints = 27
ema_alpha = 1
mean_alpha = 3
align_frame = 500
window_size = 40    #原40
window_speed = 10   #原10

joint_connections = [[0, 1], [0, 18], [0, 22],
    [1, 2], [2, 3], [2, 4], [2, 11], [3, 26], [4, 5], [5, 6], [6, 7], [7, 8], [7, 10], [8, 9],
    [11, 12], [12, 13], [13, 14], [14, 15], [14, 17], [15, 16], [18, 19], [19, 20], [20, 21],
    [22, 23], [23, 24], [24, 25]]

def get_coordinate_from_file(skeleton_file_path):
    with open(skeleton_file_path, 'r') as file:
        lines = file.readlines()

    num_frames = int(lines[-1].split()[0])
    skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    current_line = 0
    for frame in range(num_frames):
        num_bodies = int(lines[current_line].strip('\r\n'))
        if num_bodies == 2:
            print("Error: num_bodies is 2")
            break
        current_line += 1
        for j in range(num_joints):
            tem_str = lines[current_line].strip('\r\n').split()
            skeleton_data[frame, j, :] = np.array(tem_str[:3], dtype=np.float32)
            current_line += 1
    return skeleton_data

def visualize(skeleton_data):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.view_init(elev=-90, azim=-90)
    xmin,xmax = skeleton_data[:,:,0].min(),skeleton_data[:,:,0].max()
    ymin,ymax = skeleton_data[:,:,1].min(),skeleton_data[:,:,1].max()
    zmin,zmax = skeleton_data[:,:,2].min(),skeleton_data[:,:,2].max()

    def update(frame):  # 帧
        ax.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        x = skeleton_data[frame, :, 0]
        y = skeleton_data[frame, :, 1]
        z = skeleton_data[frame, :, 2]
        ax.scatter(x, y, z, s=10)
        for connection in joint_connections:
            ax.plot(x[connection], y[connection], z[connection])



    ani = animation.FuncAnimation(fig,
                                  func=update,
                                  frames=list(range(skeleton_data.shape[0])),
                                  init_func=None)

    ani.save(gif_save_path, writer='pillow')

    plt.show()
    return

def visualize_two_body(skeleton_data, skeleton_data2):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.view_init(elev=-90, azim=-90)
    xmin,xmax = skeleton_data[:,:,0].min(),skeleton_data[:,:,0].max()
    ymin,ymax = skeleton_data[:,:,1].min(),skeleton_data[:,:,1].max()
    zmin,zmax = skeleton_data[:,:,2].min(),skeleton_data[:,:,2].max()
    xmin2, xmax2 = skeleton_data2[:, :, 0].min(), skeleton_data2[:, :, 0].max()
    ymin2, ymax2 = skeleton_data2[:, :, 1].min(), skeleton_data2[:, :, 1].max()
    zmin2, zmax2 = skeleton_data2[:, :, 2].min(), skeleton_data2[:, :, 2].max()
    xmin, xmax = max(xmin, xmin2), max(xmax, xmax2)
    ymin, ymax = max(ymin, ymin2), max(ymax, ymax2)
    zmin, zmax = max(zmin, zmin2), max(zmax, zmax2)

    def update(frame):  # 帧
        ax.cla()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        x = skeleton_data[frame, :, 0]
        y = skeleton_data[frame, :, 1]
        z = skeleton_data[frame, :, 2]
        x2 = skeleton_data2[frame, :, 0]
        y2 = skeleton_data2[frame, :, 1]
        z2 = skeleton_data2[frame, :, 2]
        ax.scatter(x, y, z, s=10,color='red',marker='*')
        ax.scatter(x2, y2, z2, s=10,color='blue',alpha=0.2)
        for connection in joint_connections:
            ax.plot(x[connection], y[connection], z[connection],color='red')
            ax.plot(x2[connection], y2[connection], z2[connection],color='blue',alpha=0.2)
        ax.text(xmax, ymax, 1, f"{frame}", fontsize=12, color='red')



    ani = animation.FuncAnimation(fig,
                                  func=update,
                                  frames=list(range(skeleton_data.shape[0])),
                                  init_func=None)

    ani.save("vis/ani_two_body.gif", writer='pillow')

    plt.show()
    return

def analyze_action_scores(scores):
    x = list(range(len(scores)))
    y1 = [scores[i] for i in x]

    plt.plot(x, y1, color='blue')
    plt.show()

def analyze_two_action_scores(scores1, scores2):
    x = list(range(len(scores1)))
    y1 = [scores1[i] for i in x]
    y2 = [scores2[i] for i in x]


    plt.plot(x, y1, color='blue')
    plt.plot(x, y2, color='red')
    plt.show()

# def data_alignment(skeleton_data, aling_num_frames):
#     num_frames = len(skeleton_data)
#     a = num_frames / aling_num_frames
#     b = []
#     for i in range(aling_num_frames):
#         b.append(int((i + 1) * a) - 1)
#     return skeleton_data[b]

def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha*data[i] + (1-alpha) * ema[i-1])
    return ema

def data_smoothing(skeleton_data, alpha):
    smoothed_data = np.zeros_like(skeleton_data)
    for joint in range(skeleton_data.shape[1]):
        for dimension in range(skeleton_data.shape[2]):
            smoothed_data[:, joint, dimension] = exponential_moving_average(skeleton_data[:, joint, dimension],alpha)
    return skeleton_data

def data_smooting_by_mean(skeleton_data, alpha):
    skeleton_data = skeleton_data[:skeleton_data.shape[0] // alpha * alpha]
    split_arr = np.split(skeleton_data, skeleton_data.shape[0] // alpha)
    mean_arr = [np.mean(split, axis=0) for split in split_arr]
    return np.array(mean_arr)

def shift(skeleton_data):
    origin = np.copy(skeleton_data[0, 0, :])
    length = np.linalg.norm(skeleton_data[0][0] - skeleton_data[0][3])
    for frame in range(len(skeleton_data)):
        # 数据平移，以0号点作为坐标系原点
        skeleton_data[frame] -= np.array([origin] * skeleton_data.shape[1])
        # skeleton_data[frame] -= np.array(skeleton_data[frame][0])

        # 数据放缩，统一除以0号点pelvis到3号点neck的长度
        skeleton_data[frame] /= length

    return skeleton_data

# def dtw_alignment(ske_data1, ske_data2):
#     def distance(x, y):
#         return euclidean(x.reshape(-1), y.reshape(-1))

#     # window_size = max(len(ske_data1), len(ske_data2)) // 5
#     dis, path = fastdtw(ske_data1, ske_data2, dist= distance)
#     best_match_indices = {}
#     for idx1, idx2 in path:
#         if idx1 in best_match_indices:
#             if distance(ske_data1[idx1], ske_data2[idx2]) < distance(ske_data1[idx1], ske_data2[best_match_indices[idx1]]):
#                 best_match_indices[idx1] = idx2
#         else:
#             best_match_indices[idx1] = idx2
#     seq1_aligned = np.array([ske_data1[idx1] for idx1 in best_match_indices.keys()])
#     seq2_aligned = np.array([ske_data2[idx2] for idx2 in best_match_indices.values()])

#     return seq1_aligned, seq2_aligned

def interpolate_skeleton_data(skeleton_data, target_num_frames):
    num_frames, num_joints, _ = skeleton_data.shape
    original_indices = np.arange(num_frames)
    target_indices = np.linspace(0, num_frames - 1, target_num_frames)
    
    interpolated_data = np.zeros((target_num_frames, num_joints, 3))
    
    for joint in range(num_joints):
        for dim in range(3):
            interp_func = interp1d(original_indices, skeleton_data[:, joint, dim], kind='linear')
            interpolated_data[:, joint, dim] = interp_func(target_indices)
    
    return interpolated_data

def align_skeleton_data(skeleton_data1, skeleton_data2):
    num_frames1 = skeleton_data1.shape[0]
    num_frames2 = skeleton_data2.shape[0]
    
    target_num_frames = max(num_frames1, num_frames2)
    
    aligned_data1 = interpolate_skeleton_data(skeleton_data1, target_num_frames)
    aligned_data2 = interpolate_skeleton_data(skeleton_data2, target_num_frames)
    
    return aligned_data1, aligned_data2

def align_skeleton_sequences(seq1, seq2):
    num_frames1, num_joints, _ = seq1.shape
    num_frames2 = seq2.shape[0]

    if num_frames1 == num_frames2:
        return seq1, seq2
    
    if num_frames1 < num_frames2:
        new_frames = np.linspace(0, num_frames1 - 1, num_frames2)
        aligned_seq1 = interp1d(np.arange(num_frames1), seq1, axis=0, kind='linear')(new_frames)
        aligned_seq2 = seq2
    else:
        new_frames = np.linspace(0, num_frames2 - 1, num_frames1)
        aligned_seq2 = interp1d(np.arange(num_frames2), seq2, axis=0, kind='linear')(new_frames)
        aligned_seq1 = seq1
    
    return aligned_seq1, aligned_seq2

def get_action_sequence(skeleton_data):
    # 假定为15 / 3 = 5frame/s, 以3秒为一个滑动窗口区间也即15frame, 滑动速度为5frame
    num_action_sequence = (skeleton_data.shape[0] - window_size) // window_speed + 1
    output_shape = (num_action_sequence, window_size * skeleton_data.shape[1] * skeleton_data.shape[2])
    output = np.zeros(output_shape)

    for i in range(num_action_sequence):
        start = i * window_speed
        end = start + window_size
        output[i] = skeleton_data[start:end].reshape(-1)

    return output

def get_action_score(test_action, master_action):
    scores = []
    index = 0

    while index < len(test_action):
        cos_sim = 1 - cosine(test_action[index], master_action[index])
        scores.append(cos_sim)
        index += 1

    for score in scores:
        print("%.4f" %score, end=', ')

    return scores, np.mean(scores)

def get_relative_distance(skeleton_data):
    num_frames, num_joints, _ = skeleton_data.shape
    relative_skeleton_data = np.zeros((num_frames, num_joints, 3))

    for frame in range(num_frames):
        reference_joint = skeleton_data[frame, 0, :]
        for joint in range(num_joints):
            relative_skeleton_data[frame, joint, :] = skeleton_data[frame, joint, :] - reference_joint

    return relative_skeleton_data


def main():
    master_data = get_coordinate_from_file(file_path)[:, :27, :]
    # master_data = data_smoothing(master_data, ema_alpha)
    # master_data = data_smooting_by_mean(master_data, mean_alpha)

    scores = []
    compare_scores = []

    for file in range(20, 40):
        skeleton_tester_file = "../skeleton_datas/A" + str(file+1) + ".txt"
        if not os.path.exists(skeleton_tester_file):
            continue
        test_data = get_coordinate_from_file(skeleton_tester_file)[:, :27, :]
        # test_data = data_smoothing(test_data, ema_alpha)
        # test_data = data_smooting_by_mean(test_data, mean_alpha)
        aligned_test_data, aligned_master_data = align_skeleton_sequences(test_data, master_data)
        relative_test_data = get_relative_distance(aligned_test_data)
        relative_master_data = get_relative_distance(aligned_master_data)

        master_data_sequence = get_action_sequence(relative_master_data)
        test_data_sequence = get_action_sequence(relative_test_data)

        compare_score, score = get_action_score(test_data_sequence, master_data_sequence)
        scores.append(score)
        compare_scores.append(compare_score)
        print(skeleton_tester_file + " : " + str(score))

    # 将scores保存到一个TXT文件中
    scores_file_path = "scores_with_relative_distance.txt"
    with open(scores_file_path, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")

    for i in range(0, len(compare_scores)):
        analyze_action_scores(compare_scores[i])
    
    # analyze_two_action_scores(compare_scores[0], compare_scores[1])


if __name__ == '__main__':
    main()