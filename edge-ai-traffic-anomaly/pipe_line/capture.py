"""
capture.py — Bắt gói tin cross-platform (Windows + Linux) qua Scapy.

Yêu cầu:
  - Linux  : sudo (quyền root để mở raw socket)
  - Windows: cài Npcap (https://npcap.com) + chạy terminal Administrator

Đẩy tuple (ts, src_ip, dst_ip, src_port, dst_port, proto,
            pkt_len, header_len, flags) vào packet_queue.

Hàm list_interfaces() để main.py và start script gọi hiển thị danh sách card mạng.
"""

import time
import logging
import platform
from queue import Queue, Full

log = logging.getLogger("capture")

IS_WINDOWS = platform.system() == "Windows"


def list_interfaces() -> list[dict]:
    """
    Trả về danh sách card mạng có thể capture.
    Mỗi phần tử: {"name": str, "description": str, "ips": list[str]}

    Hoạt động trên cả Linux và Windows (với Npcap đã cài).
    Thứ tự ưu tiên: Scapy → psutil → socket (Linux fallback)
    """
    # ── Phương án 1: dùng Scapy (chuẩn nhất, cần quyền root/admin) ──────────
    try:
        from scapy.arch import get_if_list
        from scapy.interfaces import IFACES

        scapy_names = get_if_list()
        if scapy_names:
            result = []
            for iface_name in scapy_names:
                iface_obj = IFACES.data.get(iface_name)
                desc = ""
                ips: list[str] = []

                if iface_obj:
                    # description: Windows có tên thân thiện, Linux thường trống
                    desc = getattr(iface_obj, "description", "") or ""
                    if not desc:
                        desc = getattr(iface_obj, "name", "") or iface_name

                    # Lấy danh sách địa chỉ IP
                    raw_ips = getattr(iface_obj, "ips", []) or []
                    for addr in raw_ips:
                        if addr is None:
                            continue
                        if hasattr(addr, "address"):
                            ips.append(str(addr.address))
                        elif isinstance(addr, str):
                            ips.append(addr)

                result.append({"name": iface_name, "description": desc, "ips": ips})
            return result
    except Exception as e:
        log.warning("Scapy list_interfaces thất bại (%s) — thử psutil", e)

    # ── Phương án 2: dùng psutil (cross-platform, không cần quyền root) ──────
    try:
        import psutil

        addrs_map = psutil.net_if_addrs()
        stats_map = psutil.net_if_stats()
        result = []
        import socket as _socket
        AF_INET = _socket.AF_INET

        for name, addr_list in addrs_map.items():
            ips = [a.address for a in addr_list if a.family == AF_INET and a.address]
            st = stats_map.get(name)
            status = "UP" if (st and st.isup) else "DOWN"
            result.append({
                "name": name,
                "description": f"{status} | {name}",
                "ips": ips,
            })
        if result:
            return result
    except ImportError:
        log.warning("psutil không được cài — thử socket fallback")
    except Exception as e:
        log.warning("psutil list_interfaces thất bại: %s", e)

    # ── Phương án 3: socket SIOCGIFCONF (Linux only, không cần thư viện ngoài) ─
    if not IS_WINDOWS:
        try:
            import socket
            import fcntl
            import struct
            import array

            SIOCGIFCONF = 0x8912
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            buf = array.array("B", b"\x00" * 4096)
            byteslen = struct.unpack("iL", fcntl.ioctl(
                s.fileno(), SIOCGIFCONF,
                struct.pack("iL", len(buf), buf.buffer_info()[0])
            ))[0]
            s.close()
            raw = bytes(buf.tobytes()[:byteslen])
            result = []
            for i in range(0, byteslen, 40):
                name = raw[i:i + 16].rstrip(b"\x00").decode("utf-8", errors="ignore").strip()
                ip_bytes = raw[i + 20:i + 24]
                ip = ".".join(str(b) for b in ip_bytes)
                if name:
                    result.append({"name": name, "description": name, "ips": [ip]})
            return result
        except Exception as e:
            log.warning("socket fallback thất bại: %s", e)

    return []


def _decode_flags(tcp_flags) -> str:
    """Chuyển Scapy flags object thành chuỗi ký tự (S, A, F, R, P, U)."""
    if tcp_flags is None:
        return ""
    # Scapy trả về FlagValue object, có thể cast sang str hoặc int
    flag_str = str(tcp_flags)
    # Giữ đúng ký tự mà feature_extractor đang dùng để đếm
    result = ""
    if "S" in flag_str: result += "S"
    if "A" in flag_str: result += "A"
    if "F" in flag_str: result += "F"
    if "R" in flag_str: result += "R"
    if "P" in flag_str: result += "P"
    if "U" in flag_str: result += "U"
    return result


def _make_pkt_handler(packet_queue: Queue, stats: dict, stop_event):
    """
    Trả về hàm callback cho scapy sniff().
    Tách ra để dễ test riêng lẻ hàm xử lý gói tin.
    """
    from scapy.layers.inet import IP, TCP, UDP

    def handle(pkt):
        if stop_event.is_set():
            return

        # Chỉ xử lý IPv4 TCP/UDP — bỏ qua ARP, IPv6, ICMP...
        if IP not in pkt:
            return
        if TCP not in pkt and UDP not in pkt:
            return

        ip    = pkt[IP]
        ts    = time.monotonic()
        proto = ip.proto  # 6=TCP, 17=UDP

        src_ip, dst_ip = ip.src, ip.dst
        pkt_len        = len(pkt)      # tổng kích thước gói
        ip_header_len  = ip.ihl * 4

        if TCP in pkt:
            tcp = pkt[TCP]
            src_port    = tcp.sport
            dst_port    = tcp.dport
            tcp_hdr_len = tcp.dataofs * 4
            header_len  = ip_header_len + tcp_hdr_len
            flags       = _decode_flags(tcp.flags)
        else:
            udp = pkt[UDP]
            src_port   = udp.sport
            dst_port   = udp.dport
            header_len = ip_header_len + 8   # UDP header cố định 8 byte
            flags      = ""

        pkt_tuple = (ts, src_ip, dst_ip, src_port, dst_port,
                     proto, pkt_len, header_len, flags)

        stats["pkt_total"] = stats.get("pkt_total", 0) + 1

        try:
            packet_queue.put_nowait(pkt_tuple)
        except Full:
            # Queue đầy — drop và ghi số liệu để debug
            stats["queue_drops"] = stats.get("queue_drops", 0) + 1

    return handle


def run(iface: str, packet_queue: Queue, stats: dict, stop_event):
    """
    Vòng lặp capture chính. Chạy trong thread riêng.
    Dùng scapy.sniff() — hoạt động trên cả Linux lẫn Windows (với Npcap).
    """
    try:
        from scapy.all import sniff
    except ImportError:
        log.error("Scapy chưa cài. Chạy: pip install scapy")
        raise

    handler = _make_pkt_handler(packet_queue, stats, stop_event)

    log.info("Bắt đầu capture trên interface: %s", iface)
    if IS_WINDOWS:
        log.info("Windows mode — cần Npcap (https://npcap.com) và chạy Administrator.")

    # store=0: không lưu gói vào RAM (quan trọng cho thiết bị nhúng ít RAM)
    # stop_filter: dừng khi stop_event được set (Scapy kiểm tra sau mỗi gói)
    try:
        sniff(
            iface=iface,
            prn=handler,
            store=0,
            stop_filter=lambda _: stop_event.is_set(),
        )
    except PermissionError:
        msg = ("Cần quyền root (Linux: sudo) hoặc Administrator (Windows) "
               "để capture traffic.")
        log.error(msg)
        raise
    except OSError as e:
        log.error("Lỗi mở interface '%s': %s", iface, e)
        log.info("Dùng --list-ifaces để xem danh sách interface hợp lệ.")
        raise

    log.info("Capture đã dừng.")
