
import json
import sys
import os
import shutil

def upgrade_project(filepath):
    print(f"Processing {filepath}...")
    try:
        if not os.path.exists(filepath):
            print("File not found.")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)
            
        changed = False
        
        for prefix in ['mod1', 'mod2']:
            if prefix in data:
                m = data[prefix]
                if 'custom_mask' not in m:
                    print(f"  Adding custom_mask field to {prefix}")
                    m['custom_mask'] = {
                        'mode': 0, # Default to File mode for compatibility
                        'trans_formula': '1.0',
                        'phase_formula': '0.0',
                        'trans_vars': {},
                        'phase_vars': {}
                    }
                    changed = True
                    
        if changed:
            # Backup
            backup = filepath + ".bak"
            if not os.path.exists(backup):
                shutil.copy2(filepath, backup)
                print(f"  Backed up to {backup}")
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            print("  Upgrade successful.")
        else:
            print("  No changes needed.")
            
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/upgrade_project.py <project_file>")
        sys.exit(1)
        
    upgrade_project(sys.argv[1])
