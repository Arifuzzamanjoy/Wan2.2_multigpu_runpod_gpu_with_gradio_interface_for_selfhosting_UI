#!/usr/bin/env python3
"""
Patch to add CUDA cache clearing before VAE decode operations.
This prevents OOM errors during the decode phase after diffusion completes.

Usage: Run this once before using the multi-GPU script
"""

import os

VAE_FILE = "/workspace/Wan2.2/wan/modules/vae2_1.py"

# Read the original file
with open(VAE_FILE, 'r') as f:
    content = f.read()

# Check if already patched
if 'torch.cuda.empty_cache()' in content:
    print("✓ VAE already patched for memory optimization")
    exit(0)

# Find the decode method and add cache clearing
old_decode = """    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]"""

new_decode = """    def decode(self, zs):
        import torch
        import gc
        # Clear CUDA cache before decoding to prevent OOM
        torch.cuda.empty_cache()
        gc.collect()
        
        with amp.autocast(dtype=self.dtype):
            results = []
            for u in zs:
                # Decode one at a time with cache clearing
                decoded = self.model.decode(u.unsqueeze(0),
                                           self.scale).float().clamp_(-1, 1).squeeze(0)
                results.append(decoded)
                # Clear cache after each decode
                torch.cuda.empty_cache()
            return results"""

if old_decode in content:
    content = content.replace(old_decode, new_decode)
    
    # Backup original
    backup_file = VAE_FILE + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            f.write(content.replace(new_decode, old_decode))
        print(f"✓ Created backup: {backup_file}")
    
    # Write patched version
    with open(VAE_FILE, 'w') as f:
        f.write(content)
    
    print("✓ Successfully patched VAE decoder with memory optimization")
    print("  - Added CUDA cache clearing before/after decode")
    print("  - Sequential decoding with cache management")
else:
    print("❌ Could not find decode method to patch")
    print("   File may have been modified")
    exit(1)
