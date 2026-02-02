import asyncio
import logging
import sys
import signal
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# import listener and analyzer
from src.deployment.twitch_listener import SimpleTwitchChatListener
from src.deployment.realtime_analyzer import RealtimeAnalyzer, SimpleBoWClassifier

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# environment variables
load_dotenv(project_root / '.env')
BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
BOT_NICK = os.getenv('TWITCH_BOT_NICK', 'StreamAnalysisBot')

# global reference to analyzer
analyzer = None

def signal_handler(sig, frame):
    global analyzer
    if analyzer:
        asyncio.create_task(analyzer.stop())

# Message Handler
def handle_message(msg_context):
    """Display message in format: DATE TIME - USERNAME: MESSAGE"""
    dt = datetime.fromtimestamp(msg_context.timestamp)
    dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"{dt_str} - {msg_context.username}: {msg_context.text}")

async def main():
    """Run the chat display"""
    global analyzer

    print("\n" + "=" * 100)
    print("  TWITCH CHAT DISPLAY")
    print("=" * 100)
    print()

    # get channel from user
    while True:
        channel = input("  Enter Twitch channel name: ").strip().lower()
        if channel:
            break
        print("  ❌ Channel name cannot be empty. Please try again.")

    print()

    # validate credentials
    if not all([BOT_TOKEN, CLIENT_ID, OAUTH_TOKEN]):
        print("\n  ❌ ERROR: Missing Twitch credentials!")
        print("\n     Create a .env file with:")
        print("     TWITCH_BOT_TOKEN=your_token")
        print("     TWITCH_CLIENT_ID=your_client_id")
        print("     TWITCH_OAUTH_TOKEN=your_oauth_token")
        return

    try:
        print(f"  [1/3] Creating Twitch chat listener...")
        listener = SimpleTwitchChatListener(
            channel=channel,
            bot_token=BOT_TOKEN,
            nickname=BOT_NICK,
        )

        print(f"  [2/3] Creating real-time analyzer...")
        analyzer = RealtimeAnalyzer(
            chat_listener=listener,
            classifier=SimpleBoWClassifier(),
        )

        print(f"  [3/3] Starting chat listener...\n")

        # add message handler
        listener.on_message_callback = handle_message

        print("  ✅ Connected! Listening for messages...\n")
        print("  " + "=" * 96)
        print("  Messages:\n")

        # signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        await analyzer.start()

    except asyncio.CancelledError:
        print("\n\n  " + "=" * 96)
        print("  ⏹️  Shutting down...")
        print("  " + "=" * 96)
        print("  ✅ Shutdown complete\n")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())