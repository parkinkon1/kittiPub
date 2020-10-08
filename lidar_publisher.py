import os

import rospy
# import tf2_ros
# import tf_conversions
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg

from cv_bridge import CvBridge
import numpy as np

import pykitti
import sys

np.random.seed(777)


kitti_root_dir = '/datasets/kitti/raw'
kitti_date = '2011_09_30'
kitti_drive = '0033'

dataset = pykitti.raw(kitti_root_dir, kitti_date, kitti_drive)


def makePointCloud2Msg(points, frame_time, parent_frame, pcd_format):
    ros_dtype = sensor_msgs.PointField.FLOAT32

    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(pcd_format)]

    # header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.from_sec(frame_time))

    num_field = 3
    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * num_field),
        row_step=(itemsize * num_field * points.shape[0]),
        data=data
    )


if __name__ == '__main__':

    rospy.init_node('KittiPublisher')  # don't have blank (space) in the name
    # r = rospy.Rate(2)
    r = rospy.Rate(20)


    bridge = CvBridge()

    # scan_publisher = rospy.Publisher('velodyne_points', sensor_msgs.PointCloud2, queue_size=10)
    scan_publisher = rospy.Publisher('lidar_top', sensor_msgs.PointCloud2, queue_size=10)

    for idx, cloud in enumerate(dataset.velo):
        scan_publisher.publish(makePointCloud2Msg(cloud[:, :3], idx, "KITTI", 'xyz'))
        # scan_publisher.publish(makePointCloud2Msg(cloud[:, :4], idx, "KITTI", 'xyzi'))
        print(idx)
        r.sleep()







