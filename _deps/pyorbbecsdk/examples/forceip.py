import time

import numpy as np

from pyorbbecsdk import *

def get_ip_config():
    cfg = OBDeviceIpAddrConfig()
    cfg.dhcp = 0

    print("Please enter the network configuration information:")

    while True:
        val = input("Enter IP address: ")
        parts = val.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            cfg.address = val
            break
        print("Invalid format.")

    while True:
        val = input("Enter Subnet Mask: ")
        parts = val.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            cfg.netmask = val
            break
        print("Invalid format.")

    while True:
        val = input("Enter Gateway address: ")
        parts = val.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            cfg.gateway = val
            break
        print("Invalid format.")

    return cfg

def select_device(device_list):
    device_count = device_list.get_count()
    if device_count == 0:
        print("No devices found.")
        return -1

    index_list = []
    ethernet_dev_num = 0

    print("Device list:")
    for i in range(device_count):
        conn_type = device_list.get_device_connection_type_by_index(i)
        if conn_type != "Ethernet":
            continue

        print(f"{ethernet_dev_num}. Name: {device_list.get_device_name_by_index(i)}, "
              f"Mac: 0x{device_list.get_device_uid_by_index(i)}, "
              f"Serial Number: {device_list.get_device_serial_number_by_index(i)}, "
              f"IP: {device_list.get_device_ip_address_by_index(i)}, "
              f"Subnet Mask: {device_list.get_device_subnet_mask_by_index(i)}, "
              f"Gateway: {device_list.get_device_gateway_by_index(i)}")
        index_list.append(i)
        ethernet_dev_num += 1

    if not index_list:
        print("No network devices found.")
        return -1

    while True:
        try:
            choice = int(input("Enter your choice: "))
            if 0 <= choice < len(index_list):
                return index_list[choice]
            else:
                print("Invalid input, please enter a valid index number.")
        except ValueError:
            print("Invalid input, please enter a number.")
    return -1

def main():
    context = Context()
    device_list = context.query_devices()
    device_number = select_device(device_list)
    if device_number != -1:
        config = get_ip_config()
        device_status = context.ob_force_ip_config(device_list.get_device_uid_by_index(device_number), config)
        if device_status is not True:
            print("Failed to apply the new IP configuration.")
        else:
            print("The new IP configuration has been successfully applied to the device.")
            
if __name__ == "__main__":
    main()

