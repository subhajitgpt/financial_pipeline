#!/usr/bin/env python3
"""
Helper script to encrypt API keys for use in the web application.
Run this locally to generate encrypted versions of your keys.
"""

import base64
import hashlib

def encrypt_key(plaintext, passphrase="default_salt_2024"):
    """Encrypt a key using XOR with SHA256 hash of passphrase"""
    try:
        key_hash = hashlib.sha256(passphrase.encode()).digest()
        encrypted = bytearray()
        
        for i, char in enumerate(plaintext.encode('utf-8')):
            encrypted.append(char ^ key_hash[i % len(key_hash)])
        
        return base64.b64encode(encrypted).decode('utf-8')
    except Exception as e:
        print(f"Encryption error: {e}")
        return ""

def decrypt_key(encrypted_data, passphrase="default_salt_2024"):
    """Decrypt a key (for verification)"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        key_hash = hashlib.sha256(passphrase.encode()).digest()
        
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return ""

def main():
    print("API Key Encryption Tool")
    print("=" * 30)
    
    # Get keys from user
    gmaps_key = input("Enter your Google Maps API key (or press Enter to skip): ").strip()
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    # Optional custom passphrase
    custom_passphrase = input("Enter custom passphrase (or press Enter for default): ").strip()
    passphrase = custom_passphrase if custom_passphrase else "default_salt_2024"
    
    print("\nEncrypted Keys:")
    print("=" * 30)
    
    if gmaps_key:
        encrypted_gmaps = encrypt_key(gmaps_key, passphrase)
        print(f'const GMAPS_KEY_ENCRYPTED = "{encrypted_gmaps}";')
        
        # Verify
        decrypted = decrypt_key(encrypted_gmaps, passphrase)
        print(f"✓ Google Maps key verified: {decrypted == gmaps_key}")
    
    if openai_key:
        encrypted_openai = encrypt_key(openai_key, passphrase)
        print(f'const OPENAI_KEY_ENCRYPTED = "{encrypted_openai}";')
        
        # Verify
        decrypted = decrypt_key(encrypted_openai, passphrase)
        print(f"✓ OpenAI key verified: {decrypted == openai_key}")
    
    if custom_passphrase:
        print(f"\n⚠️  Remember to update the passphrase in your decrypt_key function!")
        print(f"   Change 'default_salt_2024' to '{passphrase}'")

if __name__ == "__main__":
    main()