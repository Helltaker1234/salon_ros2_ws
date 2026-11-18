#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from physics_atv_racepak.msg import RpControls, RpShockSensors, RpWheelEncoders
from nav_msgs.msg import Odometry
from struct import *
import serial


class RacepakNode(Node):
    def __init__(self):
        super().__init__('racepak_node')

        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        port = self.get_parameter('serial_port').value

        self.baud_rate = 57600
        parity = serial.PARITY_NONE
        data_width = serial.EIGHTBITS
        stop_bits = serial.STOPBITS_ONE
        self.racepak = serial.Serial(port, self.baud_rate, data_width, parity, stop_bits, xonxoff=False)
        self.racepak.flushInput()
        self.get_logger().info('[Racepak] Serial Port {} Opened...'.format(port))

        self.pub_rp_controls = self.create_publisher(RpControls, 'controls', 10)
        self.pub_rp_shocks = self.create_publisher(RpShockSensors, 'shock_pos', 10)
        self.pub_rp_rpm = self.create_publisher(RpWheelEncoders, 'wheel_rpm', 10)
        self.pub_speed = self.create_publisher(Odometry, '/vehicle_state/velocity', 10)

        self.channel_num = 0
        self.channel_size_list = []
        self.can_id_list = []
        self.data_type_list = []
        self.data_exponent_list = []

    def start_read(self):
        self.racepak.write(b'\x91')
        self.get_logger().info('[Racepak] Sent Start Command...')

    def stop_read(self):
        self.racepak.write(b'\x92')
        self.get_logger().info('[Racepak] Sent Stop Command...')

    def update(self):
        rx_bytes = self.racepak.read(1)
        while rx_bytes != b'\xA0' and rx_bytes != b'\xA1':
            rx_bytes = self.racepak.read(1)
            if len(rx_bytes) == 0:
                self.get_logger().warning('[Racepak] Error No Data Received while Waiting for Header!')
                return False

        if rx_bytes == b'\xA0':
            self.get_logger().info('[Racepak] Receive Telemetry Starter..')
            payload_size = int.from_bytes(self.racepak.read(2), byteorder='little')
            if payload_size != 150:
                self.get_logger().warning('[Racepak] Error Telemetry Starter Payload Size {}, should be 150!'.format(payload_size))
                return
            data_raw = self.racepak.read(payload_size)
            checksum = int.from_bytes(self.racepak.read(2), byteorder='little')
            datasum = sum(data_raw)
            if checksum != datasum:
                self.get_logger().warning('[Racepak] Error Start Telemetry Checksum {}, but got {}!'.format(checksum, datasum))
                return
            self.read_telemetry_header(data_raw[:18])
            self.read_channel_header(data_raw[18:])

        elif rx_bytes == b'\xA1':
            payload_size = int.from_bytes(self.racepak.read(2), byteorder='little')
            if payload_size != 76:
                self.get_logger().warning('[Racepak] Error Data Stream Payload Size {}, should be 76!'.format(payload_size))
                return
            data_raw = self.racepak.read(payload_size)
            checksum_data = int.from_bytes(self.racepak.read(2), byteorder='little')
            datasum = sum(data_raw)
            if checksum_data != datasum:
                self.get_logger().warning('[Racepak] Error Data Streaming Checksum {}, but got {}!'.format(checksum_data, datasum))
                return
            self.read_data_stream(data_raw)

        return True

    def read_telemetry_header(self, data_raw):
        racepak_sn = int.from_bytes(data_raw[0:4], byteorder='little')
        print('    Racepak serial number is ', racepak_sn)
        racepak_time_ms = int.from_bytes(data_raw[8:12], byteorder='little')
        print('    Racepak time alive is ', racepak_time_ms, ' ms')
        racepak_txr_2 = int.from_bytes(data_raw[12:16], byteorder='little')
        print('    Racepak TX rate is ', racepak_txr_2, ' Hz')
        self.channel_num = int(data_raw[16])
        print('    Racepak num channels is ', self.channel_num)

    def read_channel_header(self, channel_headers):
        self.channel_size_list = []
        self.can_id_list = []
        self.data_type_list = []
        self.data_exponent_list = []

        for i in range(self.channel_num):
            channel_header = channel_headers[i*6:(i+1)*6]
            can_id = channel_header[:2][::-1].hex()
            self.can_id_list.append(can_id)

            channel_headers_size_in_bytes = int(channel_header[2] / 8)
            self.channel_size_list.append(channel_headers_size_in_bytes)

            self.data_type_list.append(channel_header[3])

            data_exponent = channel_header[4]
            if data_exponent > 127:
                data_exponent = (256 - data_exponent) * (-1)
            self.data_exponent_list.append(data_exponent)

    def read_data_stream(self, data_raw):
        timestamp = self.get_clock().now().to_msg()
        start_index = 0
        shock_data = RpShockSensors()
        controls_data = RpControls()
        rpm_data = RpWheelEncoders()
        controls_data.header.stamp = timestamp
        rpm_data.header.stamp = timestamp
        shock_data.header.stamp = timestamp

        for i in range(self.channel_num):
            end_index = start_index + self.channel_size_list[i]
            channel_data_raw = data_raw[start_index:end_index]
            channel_data = int.from_bytes(channel_data_raw, byteorder='little') * (10 ** self.data_exponent_list[i])
            start_index = end_index

            if self.can_id_list[i] == '03a4':
                shock_data.front_left = channel_data
            elif self.can_id_list[i] == '03a5':
                shock_data.front_right = channel_data
            elif self.can_id_list[i] == '03a6':
                shock_data.rear_left = channel_data
            elif self.can_id_list[i] == '03a7':
                shock_data.rear_right = channel_data
            elif self.can_id_list[i] == '03b4':
                rpm_data.front_left = channel_data
            elif self.can_id_list[i] == '03b5':
                rpm_data.front_right = channel_data
            elif self.can_id_list[i] == '03b6':
                rpm_data.rear_left = channel_data
            elif self.can_id_list[i] == '03b7':
                rpm_data.rear_right = channel_data
            elif self.can_id_list[i] == '0421':
                controls_data.brake = channel_data
            elif self.can_id_list[i] == '030b':
                controls_data.throttle = channel_data

        avg_wheel_speed = (rpm_data.front_left + rpm_data.front_right) / 2
        lin_vel = 2.46 * avg_wheel_speed / 60

        speed_msg = Odometry()
        speed_msg.header.stamp = self.get_clock().now().to_msg()
        speed_msg.twist.twist.linear.y = lin_vel

        self.pub_speed.publish(speed_msg)
        self.pub_rp_controls.publish(controls_data)
        self.pub_rp_rpm.publish(rpm_data)
        self.pub_rp_shocks.publish(shock_data)


def main(args=None):
    rclpy.init(args=args)
    node = RacepakNode()
    node.start_read()
    try:
        while rclpy.ok():
            res = node.update()
            if not res:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_read()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
