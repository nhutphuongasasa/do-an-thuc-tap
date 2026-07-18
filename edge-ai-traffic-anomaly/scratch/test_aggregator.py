import sys
from pathlib import Path

sys.path.insert(0, "/home/phuong/Documents/do an thuc tpa tot nghiep/edge-ai-traffic-anomaly")

from pipeline.flow_aggregator import FlowAggregator

def main():
    emitted = []
    def on_flow_ready(flow, features):
        emitted.append(flow)
        
    agg = FlowAggregator(
        flow_timeout=30.0,
        time_window_sec=None,
        on_flow_ready=on_flow_ready
    )
    
    # Simulate a full TCP handshake + data + teardown
    # Client: 192.168.1.24:49510, Server: 20.184.175.1:443
    t = 100.0
    
    # 1. Handshake
    agg.add_packet("192.168.1.24", "20.184.175.1", 49510, 443, "TCP", t, 20, 0, "S", 64240) # Syn (Fwd)
    t += 0.01
    agg.add_packet("20.184.175.1", "192.168.1.24", 443, 49510, "TCP", t, 20, 0, "SA", 64240) # Syn-Ack (Bwd)
    t += 0.01
    agg.add_packet("192.168.1.24", "20.184.175.1", 49510, 443, "TCP", t, 20, 0, "A", 64240) # Ack (Fwd)
    t += 0.01
    
    # 2. Data
    agg.add_packet("192.168.1.24", "20.184.175.1", 49510, 443, "TCP", t, 20, 100, "PA", 64240) # Data (Fwd)
    t += 0.01
    agg.add_packet("20.184.175.1", "192.168.1.24", 443, 49510, "TCP", t, 20, 1000, "A", 64240) # Ack+Data (Bwd)
    t += 0.01
    
    # 3. Client FIN
    agg.add_packet("192.168.1.24", "20.184.175.1", 49510, 443, "TCP", t, 20, 0, "FA", 64240) # FIN (Fwd)
    
    print(f"After client FIN: {len(emitted)} flows emitted.")
    
    # 4. Server ACK to FIN
    t += 0.01
    agg.add_packet("20.184.175.1", "192.168.1.24", 443, 49510, "TCP", t, 20, 0, "A", 64240) # ACK to FIN (Bwd)
    
    print(f"After server ACK to FIN: {len(emitted)} flows emitted.")
    
    # 5. Server FIN
    t += 0.01
    agg.add_packet("20.184.175.1", "192.168.1.24", 443, 49510, "TCP", t, 20, 0, "FA", 64240) # FIN (Bwd)
    
    print(f"After server FIN: {len(emitted)} flows emitted.")
    
    # 6. Client final ACK
    t += 0.01
    agg.add_packet("192.168.1.24", "20.184.175.1", 49510, 443, "TCP", t, 20, 0, "A", 64240) # final ACK (Fwd)
    
    print(f"After client final ACK: {len(emitted)} flows emitted.")
    
    # Flush remaining
    agg.flush_all()
    print(f"After flush_all: {len(emitted)} flows emitted.")
    
    if len(emitted) > 0:
        print(f"First emitted flow packets count: {len(emitted[0].packets)}")
        if len(emitted) > 1:
            print(f"Second emitted flow packets count: {len(emitted[1].packets)}")

if __name__ == "__main__":
    main()
