import socket
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
CHANNEL = 'northernlion'
NICK = 'StreamAnalysisBot'

print("\n" + "="*80)
print("  Raw IRC Debug - All Data From Twitch")
print("="*80 + "\n")

try:
    # Connect
    print(f"Connecting to irc.chat.twitch.tv:6667...\n")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('irc.chat.twitch.tv', 6667))
    sock.settimeout(2)  # 2 second timeout
    print("✅ Connected!\n")

    # Send auth
    print(f"Sending auth commands...\n")
    sock.send(f"PASS {BOT_TOKEN}\r\n".encode())
    sock.send(f"NICK {NICK}\r\n".encode())
    sock.send(f"USER {NICK} 8 * :{NICK}\r\n".encode())
    sock.send("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands\r\n".encode())
    time.sleep(1)

    # Send JOIN
    print(f"Joining #{CHANNEL}...\n")
    sock.send(f"JOIN #{CHANNEL}\r\n".encode())
    print("="*80)
    print(f"  RAW DATA FROM TWITCH (30 seconds)")
    print("="*80 + "\n")

    # Listen for 30 seconds and print EVERYTHING
    start_time = time.time()
    line_count = 0

    buffer = ""
    while time.time() - start_time < 60:
        try:
            data = sock.recv(4096)

            if not data:
                print("\nConnection closed by server")
                break

            buffer += data.decode('utf-8', errors='ignore')

            # Print each complete line
            lines = buffer.split('\r\n')
            buffer = lines[-1]  # Keep incomplete line

            for line in lines[:-1]:
                if line:
                    line_count += 1
                    print(f"{line_count:3d}: {line}")

        except socket.timeout:
            # Timeout is normal when no data
            continue
        except Exception as e:
            print(f"\nError: {e}")
            break

    print(f"\n{'='*80}")
    print(f"Debug complete! Received {line_count} lines")
    print(f"{'='*80}\n")

    sock.close()

except Exception as e:
    print(f"Connection failed: {e}")
    import traceback
    traceback.print_exc()