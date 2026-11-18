#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from collections import deque
import math


def cost_function(actual, command, state, covariance, process_cov,
                  measurement_cov, huber_delta, sigmoid_slope,
                  sigmoid_cutoff, max_velocity):
    # WVN planar velocity MSE, robust scalar Kalman filter, then 1 - traversability.
    error = ((actual.x - command.x) ** 2 + (actual.y - command.y) ** 2)
    error /= 2.0 * max_velocity
    if not math.isfinite(error):
        raise ValueError('Velocity error must be finite')
    prior_cov = covariance + process_cov
    innovation = error - state
    residual = abs(innovation) / math.sqrt(measurement_cov)
    weight = 1.0 if residual <= huber_delta else huber_delta / residual
    gain = weight * prior_cov / (prior_cov + measurement_cov)
    state += gain * innovation
    covariance = (1.0 - gain) * prior_cov
    x = sigmoid_slope * (state - sigmoid_cutoff)
    if x >= 0:
        traversability = math.exp(-x) / (1.0 + math.exp(-x))
    else:
        traversability = 1.0 / (1.0 + math.exp(x))
    cost = 1.0 - max(0.001, min(1.0, traversability))
    return cost, state, covariance


def velocity_to_base(velocity, orientation, velocity_frame, base_pitch_deg):
    """Coincident origins. Base axes are IMU axes rotated about Y by base_pitch_deg.

    Passive component conversion is Ry(-base_pitch_deg). Super-LIO currently
    publishes world-frame linear velocity despite the /lio/imu/odom name.
    """
    x, y, z = velocity.x, velocity.y, velocity.z
    if not all(math.isfinite(v) for v in (x, y, z, base_pitch_deg)):
        raise ValueError('Velocity and mounting angle must be finite')
    if velocity_frame == 'world':
        q = (orientation.x, orientation.y, orientation.z, orientation.w)
        if not all(math.isfinite(v) for v in q):
            raise ValueError('Invalid orientation')
        norm = math.hypot(*q)
        if norm < 1e-12:
            raise ValueError('World velocity requires a valid IMU orientation')
        qx, qy, qz, qw = (v / norm for v in q)
        # R_world_imu transpose times world velocity.
        x, y, z = (
            (1 - 2*(qy*qy + qz*qz))*x + 2*(qx*qy + qw*qz)*y + 2*(qx*qz - qw*qy)*z,
            2*(qx*qy - qw*qz)*x + (1 - 2*(qx*qx + qz*qz))*y + 2*(qy*qz + qw*qx)*z,
            2*(qx*qz + qw*qy)*x + 2*(qy*qz - qw*qx)*y + (1 - 2*(qx*qx + qy*qy))*z)
    elif velocity_frame != 'imu':
        raise ValueError('velocity_frame must be world or imu')
    angle = math.radians(-base_pitch_deg)
    c, s = math.cos(angle), math.sin(angle)
    return Vector3(x=c*x + s*z, y=y, z=-s*x + c*z)


class TraversabilityCostNode(Node):
    def __init__(self):
        super().__init__('quadruped_cost_publisher')

        self.declare_parameter('state_topic', '/wild_visual_navigation_node/robot_state')
        self.declare_parameter('command_topic', '/motion_reference/command_twist')
        self.declare_parameter('state_type', 'RobotState')
        self.declare_parameter('command_type', 'TwistStamped')
        self.declare_parameter('velocity_frame', 'world')
        self.declare_parameter('base_pitch_deg', -21.0)
        self.declare_parameter('body_frame', 'base')
        self.declare_parameter('input_timeout', 0.5)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('process_cov', 0.1)
        self.declare_parameter('measurement_cov', 10.0)
        self.declare_parameter('huber_delta', 0.5)
        self.declare_parameter('sigmoid_slope', 20.0)
        self.declare_parameter('sigmoid_cutoff', 0.25)
        state_topic = self.get_parameter('state_topic').value
        command_topic = self.get_parameter('command_topic').value
        self.body_frame = self.get_parameter('body_frame').value
        self.input_timeout = self.get_parameter('input_timeout').value
        self.state_type = self.get_parameter('state_type').value
        self.command_type = self.get_parameter('command_type').value
        self.velocity_frame = self.get_parameter('velocity_frame').value
        self.base_pitch_deg = self.get_parameter('base_pitch_deg').value
        if self.velocity_frame not in ('world', 'imu') or not math.isfinite(self.base_pitch_deg):
            raise ValueError('Invalid velocity frame or mounting angle')
        if not self.body_frame:
            raise ValueError('body_frame must be set')
        self.cost_params = {name: self.get_parameter(name).value for name in (
            'process_cov', 'measurement_cov', 'huber_delta',
            'sigmoid_slope', 'sigmoid_cutoff', 'max_velocity')}
        if not all(math.isfinite(v) for v in (*self.cost_params.values(), self.input_timeout)):
            raise ValueError('Cost parameters must be finite')
        if (self.input_timeout <= 0 or self.cost_params['process_cov'] < 0 or
                any(self.cost_params[name] <= 0 for name in (
                    'measurement_cov', 'huber_delta', 'sigmoid_slope', 'max_velocity'))):
            raise ValueError('Invalid cost parameters')

        self.commands = deque(maxlen=1000)
        self.state = 0.0
        self.covariance = 0.1
        self.last_stamp = None
        self.last_clock = None
        self.cost_publisher = self.create_publisher(Float32, '/traversability_cost', 10)
        self.base_velocity_publisher = self.create_publisher(
            TwistStamped, '/quadruped/base_velocity', 10)
        if self.state_type == 'RobotState':
            from wild_visual_navigation_msgs.msg import RobotState
            state_class = RobotState
        elif self.state_type == 'Odometry':
            state_class = Odometry
        else:
            raise ValueError('state_type must be RobotState or Odometry')
        command_classes = {'Twist': Twist, 'TwistStamped': TwistStamped}
        if self.command_type not in command_classes:
            raise ValueError('command_type must be Twist or TwistStamped')
        self.create_subscription(state_class, state_topic, self.handle_state, qos_profile_sensor_data)
        self.create_subscription(command_classes[self.command_type], command_topic,
                                 self.handle_command, qos_profile_sensor_data)
        self.get_logger().info('Initialized quadruped_cost_publisher node')

    def read_velocity(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_clock is not None and now < self.last_clock:
            self.commands.clear()
            self.last_stamp = None
            self.state, self.covariance = 0.0, 0.1
        self.last_clock = now
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if msg.header.frame_id != self.body_frame or not 0 <= now - stamp <= self.input_timeout:
            return None
        if not all(math.isfinite(v) for v in (msg.twist.linear.x, msg.twist.linear.y)):
            return None
        return stamp, msg.twist.linear

    def handle_command(self, msg):
        if self.command_type == 'Twist':
            stamped = TwistStamped()
            # Unstamped /cmd_vel is defined at the base origin, in base axes.
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = self.body_frame
            stamped.twist = msg
            msg = stamped
        sample = self.read_velocity(msg)
        if sample is not None:
            self.commands.append(sample)

    def handle_state(self, msg):
        if self.state_type == 'Odometry':
            converted = TwistStamped()
            converted.header.stamp = msg.header.stamp
            converted.header.frame_id = self.body_frame
            try:
                converted.twist.linear = velocity_to_base(
                    msg.twist.twist.linear, msg.pose.pose.orientation,
                    self.velocity_frame, self.base_pitch_deg)
                # Current Super-LIO angular velocity is already in IMU axes.
                converted.twist.angular = velocity_to_base(
                    msg.twist.twist.angular, msg.pose.pose.orientation,
                    'imu', self.base_pitch_deg)
            except ValueError as error:
                self.get_logger().warning(str(error), throttle_duration_sec=5.0)
                return
        else:
            # Preserve the original WVN path: RobotState twist is already base-frame.
            converted = msg.twist
        sample = self.read_velocity(converted)
        if sample is None:
            return
        stamp, actual = sample
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return
        self.base_velocity_publisher.publish(converted)
        commands = [item for item in self.commands
                    if 0 <= stamp - item[0] <= self.input_timeout]
        if not commands:
            self.state, self.covariance = 0.0, 0.1
            self.last_stamp = None
            return
        if self.last_stamp is None or stamp - self.last_stamp > self.input_timeout:
            self.state, self.covariance = 0.0, 0.1
        command = max(commands, key=lambda item: item[0])[1]
        try:
            cost, self.state, self.covariance = cost_function(
                actual, command, self.state, self.covariance, **self.cost_params)
        except (ValueError, OverflowError):
            return
        self.last_stamp = stamp
        cost_msg = Float32()
        cost_msg.data = float(cost)
        self.cost_publisher.publish(cost_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityCostNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
