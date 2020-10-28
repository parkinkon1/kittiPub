
import numpy as np
import pykitti
import sys

np.random.seed(777)

sys.path.append('./src')

# kitti dataset
kitti_root_dir = '/datasets/kitti/raw'
# kitti_date = '2011_09_30'
# kitti_drive = '0033'
kitti_date = '2011_09_30'
kitti_drive = '0020'
dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)

print("kitti dataset: date {}, drive {}".format(kitti_date, kitti_drive))

gt_trajectory_lla = []  # [longitude(deg), latitude(deg), altitude(meter)] x N
gt_yaws = []  # [yaw_angle(rad),] x N
gt_yaw_rates= []  # [vehicle_yaw_rate(rad/s),] x N
gt_forward_velocities = []  # [vehicle_forward_velocity(m/s),] x N

for oxts_data in dataset.oxts:
    packet = oxts_data.packet
    gt_trajectory_lla.append([
        packet.lon,
        packet.lat,
        packet.alt
    ])
    gt_yaws.append(packet.yaw)
    gt_yaw_rates.append(packet.wz)
    gt_forward_velocities.append(packet.vf)

gt_trajectory_lla = np.array(gt_trajectory_lla).T
gt_yaws = np.array(gt_yaws)
gt_yaw_rates = np.array(gt_yaw_rates)
gt_forward_velocities = np.array(gt_forward_velocities)

timestamps = np.array(dataset.timestamps)
elapsed = np.array(timestamps) - timestamps[0]
ts = [t.total_seconds() for t in elapsed]




num_candidates = 10
threshold = 0.15
max_length = 80  # recommended but other (e.g., 100m) is also ok.

ENOUGH_LARGE = 15000  # capable of up to ENOUGH_LARGE number of nodes
ptclouds = [None] * ENOUGH_LARGE
scancontexts = [None] * ENOUGH_LARGE
ringkeys = [None] * ENOUGH_LARGE
curr_node_idx = 0

def ptcloud2sc(ptcloud, sc_shape, max_length):
    num_ring, num_sector = sc_shape

    gap_ring = max_length / num_ring
    gap_sector = 360 / num_sector

    enough_large = 500
    sc_storage = np.zeros([enough_large, num_ring, num_sector])
    sc_counter = np.zeros([num_ring, num_sector])

    num_points = ptcloud.shape[0]

    def xy2theta(x, y):
        theta = 0
        if (x >= 0 and y >= 0):
            theta = 180 / np.pi * np.arctan(y / x)
        if (x < 0 and y >= 0):
            theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)))
        if (x < 0 and y < 0):
            theta = 180 + ((180 / np.pi) * np.arctan(y / x))
        if (x >= 0 and y < 0):
            theta = 360 - ((180 / np.pi) * np.arctan((-y) / x))
        return theta

    def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        # z = point[2]

        if x == 0.0:
            x = 0.001
        if y == 0.0:
            y = 0.001

        theta = xy2theta(x, y)
        faraway = np.sqrt(x * x + y * y)

        idx_ring = np.divmod(faraway, gap_ring)[0]
        idx_sector = np.divmod(theta, gap_sector)[0]

        if (idx_ring >= num_ring):
            idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

        return int(idx_ring), int(idx_sector)

    for pt_idx in range(num_points):
        point = ptcloud[pt_idx, :]
        point_height = point[2] + 2.0  # for setting ground is roughly zero

        idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

        if sc_counter[idx_ring, idx_sector] >= enough_large:
            continue
        sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
        sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

    sc = np.amax(sc_storage, axis=0)

    return sc




from tqdm import tqdm
shape=[120,540] # [ring, sector]
max_length = 80

scs = []

for idx, cloud in tqdm(enumerate(dataset.velo)):
    sc = ptcloud2sc(np.array(cloud[:,:3]), shape, max_length)
    scs.append(sc)

scs_ = np.array(scs)
print(scs_.shape)
np.save('./experiments_sc/'+kitti_date+'_'+kitti_drive+'_scs_120_540', scs_)





