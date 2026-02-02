"""
Validate Twitch Credentials
===========================

This script checks if your OAuth token and Client ID are valid
before we try to connect to IRC.

Usage:
    python scripts/validate_credentials.py
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load env
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
CHANNEL = 'paymoneywubby'

print("\n" + "=" * 80)
print("  Twitch Credentials Validator")
print("=" * 80 + "\n")

# Check if credentials exist
print("1. Checking if credentials exist...")
if not BOT_TOKEN:
    print("   ❌ TWITCH_BOT_TOKEN is missing from .env!")
    exit(1)
else:
    print(f"   ✅ BOT_TOKEN found: {BOT_TOKEN[:20]}...")

if not CLIENT_ID:
    print("   ⚠️  TWITCH_CLIENT_ID is missing (might not need it for IRC)")
else:
    print(f"   ✅ CLIENT_ID found: {CLIENT_ID[:20]}...")

if not OAUTH_TOKEN:
    print("   ⚠️  TWITCH_OAUTH_TOKEN is missing")
else:
    print(f"   ✅ OAUTH_TOKEN found: {OAUTH_TOKEN[:20]}...")

print()

# Test with Twitch API if we have CLIENT_ID
if CLIENT_ID and OAUTH_TOKEN:
    print("2. Testing credentials with Twitch API...")

    try:
        headers = {
            'Client-ID': CLIENT_ID,
            'Authorization': f'Bearer {OAUTH_TOKEN}'
        }

        # Try to get current user info
        response = requests.get(
            'https://api.twitch.tv/helix/users',
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()['data']
            if data:
                user = data[0]
                print(f"   ✅ API credentials valid!")
                print(f"      User: {user['display_name']}")
                print(f"      ID: {user['id']}")
            else:
                print(f"   ⚠️  API credentials valid but no user data")
        elif response.status_code == 401:
            print(f"   ❌ API authentication failed (401 Unauthorized)")
            print(f"      Your token might be expired or invalid")
            print(f"      Get a new token from: https://twitchtokengenerator.com/")
        else:
            print(f"   ❌ API error: {response.status_code}")
            print(f"      {response.text}")

    except Exception as e:
        print(f"   ⚠️  Could not reach Twitch API: {e}")
else:
    print("2. Skipping API test (need both CLIENT_ID and OAUTH_TOKEN)")

print()

# For IRC, the token format matters
print("3. Checking IRC token format...")
if BOT_TOKEN.startswith('oauth:'):
    print(f"   ✅ Token has correct IRC format (oauth:...)")
else:
    print(f"   ⚠️  Token does NOT have 'oauth:' prefix")
    print(f"      IRC might need: oauth:{BOT_TOKEN}")
    print(f"      Try updating .env:")
    print(f"      TWITCH_BOT_TOKEN=oauth:{BOT_TOKEN}")

print()

# Check channel exists
print(f"4. Checking if #{CHANNEL} exists...")
try:
    if CLIENT_ID and OAUTH_TOKEN:
        headers = {
            'Client-ID': CLIENT_ID,
            'Authorization': f'Bearer {OAUTH_TOKEN}'
        }

        response = requests.get(
            f'https://api.twitch.tv/helix/users?login={CHANNEL}',
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()['data']
            if data:
                print(f"   ✅ Channel exists!")
                print(f"      Channel: {data[0]['display_name']}")
                print(f"      ID: {data[0]['id']}")
            else:
                print(f"   ❌ Channel '{CHANNEL}' not found")
        else:
            print(f"   ⚠️  Could not check channel: {response.status_code}")
    else:
        print(f"   ⚠️  Cannot check channel (need API credentials)")
except Exception as e:
    print(f"   ⚠️  Could not check channel: {e}")

print()

# Summary
print("=" * 80)
print("  Summary")
print("=" * 80)
print()
print("If your credentials passed the API test, but IRC still doesn't work:")
print()
print("1. Make sure token has 'oauth:' prefix (see above)")
print("2. Make sure you're using a CHAT token, not an app token")
print("3. Generate a new token from https://twitchtokengenerator.com/")
print("   - Select: Chat Read, Chat Edit, User Read Email")
print("4. Update your .env file")
print("5. Try again")
print()
print("=" * 80 + "\n")