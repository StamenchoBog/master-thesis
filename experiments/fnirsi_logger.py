#!/usr/bin/env python3
# FNIRSI FNB58 power logger, reduced from
# baryluk/fnirsi-usb-power-data-logger (746e4d38515a). Exits with code 3
# when the meter is lost; the restart loop in
# experiments/run/02_instruments.sh then starts a fresh process (on
# macOS only a new process gets the meter back).
#   sudo python3 -u experiments/fnirsi_logger.py > power_fnb58.csv

import signal
import sys
import time

import usb.core
import usb.util

VID = 0x2E3C
PID = 0x5558
SPS = 100
TIME_INTERVAL = 1.0 / SPS
ALPHA = 0.9  # smoothing of the meter's temperature reading
REFRESH_S = 1.0  # the FNB58 needs a keep-alive command about every second
START = b"\xaa\x81" + b"\x00" * 61 + b"\x8e"
INIT = b"\xaa\x82" + b"\x00" * 61 + b"\x96"
KEEP_ALIVE = b"\xaa\x83" + b"\x00" * 61 + b"\x9e"

STOP_REQUESTED = False
energy = 0.0
capacity = 0.0
last_power = 0.0
sample_count = 0


def request_stop(signum, frame):
    # Finish the read loop instead of dying mid-stream: a killed logger
    # leaves the meter claimed, and the next start then often gets no
    # data.
    global STOP_REQUESTED
    STOP_REQUESTED = True


def decode(data):
    global energy, capacity, last_power, sample_count
    temp_ema = None

    # 64 bytes: 0xaa, the packet type (0x04 = data), 4 samples of 15
    # bytes, then one byte of unknown purpose and an 8-bit CRC.
    if data[1] != 0x04:
        return

    t0 = time.time() - 4 * TIME_INTERVAL
    for i in range(4):
        offset = 2 + 15 * i
        # 4 + 4 + 2 + 2 + 2 + 1 (unknown constant 1) = 15 bytes
        voltage = (
            int.from_bytes(bytes(data[offset : offset + 4]), "little") / 100000
        )
        current = (
            int.from_bytes(bytes(data[offset + 4 : offset + 8]), "little")
            / 100000
        )
        dp = (data[offset + 8] + data[offset + 9] * 256) / 1000
        dn = (data[offset + 10] + data[offset + 11] * 256) / 1000
        temp_C = (data[offset + 13] + data[offset + 14] * 256) / 10.0
        temp_ema = (
            temp_C
            if temp_ema is None
            else temp_C * (1.0 - ALPHA) + temp_ema * ALPHA
        )
        power = voltage * current
        last_power = power
        sample_count += 1
        energy += power * TIME_INTERVAL
        capacity += current * TIME_INTERVAL
        t = t0 + i * TIME_INTERVAL
        print(
            f"{t:.3f} {i} {voltage:7.5f} {current:7.5f} {dp:5.3f} {dn:5.3f} "
            f"{temp_ema:6.3f} {energy:.6f} {capacity:.6f}"
        )


def request_data(ep_out):
    ep_out.write(START)
    ep_out.write(INIT)
    ep_out.write(INIT)


def connect(attempts=10):
    """Find the meter, claim its HID interface and start the stream.
    None if that fails."""
    dev = interface_hid_num = None
    for _ in range(attempts):
        dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is not None:
            try:
                interface_hid_num = next(
                    intf.bInterfaceNumber
                    for cfg in dev
                    for intf in cfg
                    if intf.bInterfaceClass == 0x03
                )
                break
            except (usb.core.USBError, StopIteration):
                pass
        time.sleep(2.0)
    if dev is None or interface_hid_num is None:
        return None

    for cfg in dev:
        for intf in cfg:
            try:
                active = dev.is_kernel_driver_active(intf.bInterfaceNumber)
            except usb.core.USBError:
                continue
            if active:
                try:
                    dev.detach_kernel_driver(intf.bInterfaceNumber)
                except usb.core.USBError as e:
                    sys.exit(
                        f"Could not detach the kernel driver from interface "
                        f"{intf.bInterfaceNumber}: {e}"
                    )
    dev.set_configuration()

    intf = dev.get_active_configuration()[(interface_hid_num, 0)]

    def endpoint(direction):
        def match(e):
            return usb.util.endpoint_direction(e.bEndpointAddress) == direction

        return usb.util.find_descriptor(intf, custom_match=match)

    ep_out, ep_in = endpoint(usb.util.ENDPOINT_OUT), endpoint(
        usb.util.ENDPOINT_IN
    )
    assert ep_in and ep_out

    # The start command itself sometimes fails with an I/O error: clear
    # the stall and retry.
    for _ in range(10):
        try:
            request_data(ep_out)
            return dev, interface_hid_num, ep_in, ep_out
        except usb.core.USBError as e:
            print(f"Start command failed ({e}), retrying", file=sys.stderr)
            clear_stall(ep_in, ep_out)
            time.sleep(1.0)
    usb.util.dispose_resources(dev)
    return None


def clear_stall(*eps):
    for ep in eps:
        try:
            ep.clear_halt()
        except usb.core.USBError:
            pass


def release(dev, interface_hid_num):
    """Free our claim on the meter. The OS drivers are not reattached:
    macOS would remount the meter's USB drive, and a logger started
    during that remount gets no samples."""
    try:
        usb.util.release_interface(dev, interface_hid_num)
    except usb.core.USBError:
        pass
    usb.util.dispose_resources(dev)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main():
    for sig in (signal.SIGTERM, signal.SIGALRM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    # macOS mounts the meter's USB drive right after it appears; let it
    # settle.
    time.sleep(2.0)
    conn = connect()
    if conn is None:
        sys.exit("Meter not found or did not accept the start command.")
    dev, interface_hid_num, ep_in, ep_out = conn

    print()  # the original logger's format starts with a blank line
    print(
        "timestamp sample_in_packet voltage_V current_A dp_V dn_V temp_C_ema "
        "energy_Ws "
        "capacity_As"
    )

    time.sleep(0.1)
    continue_time = time.time() + REFRESH_S
    timeouts = errors = 0
    lost = False
    last_beat = time.time()
    print(
        "Logger running (this heartbeat is on stderr; sample data goes to the "
        "file).",
        file=sys.stderr,
        flush=True,
    )
    while not STOP_REQUESTED:
        try:
            data = ep_in.read(size_or_buffer=64, timeout=5000)
            decode(data)
            timeouts = errors = 0
            if time.time() >= continue_time:
                continue_time = time.time() + REFRESH_S
                ep_out.write(KEEP_ALIVE)
        except usb.core.USBTimeoutError:
            # The meter sometimes goes quiet: resend the start command,
            # and after about a minute of silence treat it as lost.
            timeouts += 1
            if timeouts >= 12:
                log("meter silent")
                lost = True
                break
            try:
                request_data(ep_out)
                ep_out.write(KEEP_ALIVE)
            except usb.core.USBError:
                pass
        except usb.core.USBError as e:
            # Single I/O errors happen on the reflashed meter; five in a
            # row mean it is gone.
            errors += 1
            log(f"USB error ({e}), restarting the stream")
            if errors >= 5:
                log("meter lost")
                lost = True
                break
            time.sleep(1.0)
            # A stalled endpoint ("Pipe error") must be cleared before
            # restarting.
            clear_stall(ep_in, ep_out)
            try:
                request_data(ep_out)
            except usb.core.USBError:
                pass
        except KeyboardInterrupt:
            break

        if time.time() - last_beat >= 5.0:
            last_beat = time.time()
            log(
                f"streaming — {sample_count} samples | {last_power:.2f} W | "
                f"{energy:.1f} Ws"
            )

    if lost:
        log("exiting for a fresh restart")
        try:
            release(dev, interface_hid_num)
        except Exception:
            pass
        sys.exit(3)
    try:
        while True:  # drain what the meter still sends; the timeout ends it
            ep_in.read(size_or_buffer=64, timeout=1000)
    except usb.core.USBTimeoutError:
        pass
    finally:
        release(dev, interface_hid_num)


if __name__ == "__main__":
    main()
