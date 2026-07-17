import os
import re

directory = r"F:\Tài Liệu Học Tập\DoAnThuTap\do-an-thuc-tap\edge-ai-traffic-anomaly"

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace Path(__file__).parent.parent.parent / "TrafficGuard... 
            # with Path(__file__).parent.parent.parent / "TrafficGuard...
            new_content = re.sub(
                r'Path\(__file__\)\.parent\.parent\s*/\s*"TrafficGuard', 
                r'Path(__file__).parent.parent.parent / "TrafficGuard', 
                content
            )
            
            # Replace Path("../../TrafficGuard...
            # with Path("../../TrafficGuard...
            new_content = re.sub(
                r'Path\("\.\./TrafficGuard', 
                r'Path("../../TrafficGuard', 
                new_content
            )

            # fix the plot argument in run_drift_simulation
            new_content = re.sub(
                r'run_drift_simulation\(\s*n_epochs=n_epochs,\s*n_flows_per_epoch=n_flows,\s*plot=False,?\s*\)',
                r'run_drift_simulation(n_epochs=n_epochs, n_flows_per_epoch=n_flows)',
                new_content
            )

            if new_content != content:
                print(f"Fixing {filepath}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
