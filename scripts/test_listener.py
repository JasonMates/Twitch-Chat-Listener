"""
Test Script: Chat Listener - Using Simplified IRC Implementation
================================================================

This uses SimpleTwitchChatListener which connects directly to Twitch IRC
without requiring twitchio.commands.Bot and its strict requirements.

Usage:
    python scripts/test_listener.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"✅ Using simplified IRC listener...")

# Import simplified listener
from src.deployment.twitch_listener import SimpleTwitchChatListener
from src.deployment.realtime_analyzer import RealtimeAnalyzer, SimpleBoWClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(project_root / '.env')
BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
BOT_NICK = os.getenv('TWITCH_BOT_NICK', 'StreamAnalysisBot')

# Configuration
CHANNEL = 'vinesauce'  # ← CHANGE THIS TO YOUR TEST CHANNEL


# ============================================================================
# Message Handler
# ============================================================================

def handle_message(msg_context):
    """Print each classified message"""
    if msg_context.sentiment == 'Noise':
        return

    sentiment_colors = {
        'Positive': '\033[92m',  # Green
        'Negative': '\033[91m',  # Red
        'Enthusiastic': '\033[93m',  # Yellow
        'Anxious': '\033[94m',  # Blue
        'Neutral': '\033[37m',  # White
    }
    reset_color = '\033[0m'

    color = sentiment_colors.get(msg_context.sentiment, reset_color)

    print(f"{color}[{msg_context.username:20s}] {msg_context.text:<50s}{reset_color} "
          f"→ {msg_context.sentiment:12s} ({msg_context.confidence:.2f})")


async def handle_update(update_data):
    """Print statistics every 30 seconds"""
    moment = update_data.get('moment_detected')
    all_stats = update_data.get('all_stats', {})

    print("\n" + "=" * 100)

    if moment:
        moment_icon = "🔥 HYPE!" if moment.moment_type.value == 'hype' else "💔 FAIL!"
        print(f"\n{moment_icon} {moment_icon}")
        print(f"  Type: {moment.moment_type.value.upper()}")
        print(f"  Messages: {moment.message_count}")
        print(f"  Peak velocity: {moment.velocity_peak:.1f} msg/s")
        print(f"  Positive sentiment: {moment.sentiment_positive_pct * 100:.0f}%")
        print(f"  Dominant: {moment.sentiment_dominant}")
        print(f"  Acceleration: {moment.acceleration:.2f}x")
        print()

    w30 = all_stats.get(30)
    if w30 and w30.message_count > 0:
        print(f"\n📊 LAST 30 SECONDS:")
        print(f"   Messages: {w30.message_count:4d}")
        print(f"   Rate: {w30.messages_per_second:6.1f} msg/s")
        print(f"   Dominant: {w30.dominant_sentiment}")

        if w30.sentiment_distribution:
            print(f"   Breakdown:")
            for sentiment in ['Positive', 'Negative', 'Enthusiastic', 'Anxious', 'Neutral']:
                count = w30.sentiment_distribution.get(sentiment, 0)
                ratio = w30.sentiment_ratios.get(sentiment, 0)
                if count > 0:
                    bar = '█' * int(ratio * 50)
                    print(f"      {sentiment:12s}: {count:3d} ({ratio * 100:5.1f}%) {bar}")

        if w30.top_emotes:
            emotes_str = ', '.join(f'{e[0]} ({e[1]})' for e in w30.top_emotes[:5])
            print(f"   Top emotes: {emotes_str}")

    w1m = all_stats.get(60)
    if w1m and w1m.message_count > 0:
        print(f"\n📊 LAST 1 MINUTE:")
        print(f"   Messages: {w1m.message_count:4d}")
        print(f"   Rate: {w1m.messages_per_second:6.1f} msg/s")

    w5m = all_stats.get(300)
    if w5m and w5m.message_count > 0:
        print(f"\n📊 LAST 5 MINUTES:")
        print(f"   Messages: {w5m.message_count:4d}")
        print(f"   Rate: {w5m.messages_per_second:6.1f} msg/s")

    print(f"\n📈 CUMULATIVE:")
    print(f"   Total messages: {update_data['total_messages']:6d}")
    print(f"   Total moments: {update_data['total_moments']:6d}")
    print(f"   Baseline velocity: {update_data['baseline_velocity']:6.1f} msg/s")

    print("=" * 100 + "\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the analysis pipeline"""

    print("\n" + "=" * 100)
    print("  🎮 TWITCH CHAT LISTENER - Simplified IRC Version")
    print("=" * 100)
    print(f"\n  Channel: {CHANNEL}")
    print(f"  Classifier: SimpleBoWClassifier (Bag of Words baseline)")
    print(f"  Status: Connecting...\n")

    if not all([BOT_TOKEN, CLIENT_ID, OAUTH_TOKEN]):
        print("\n❌ ERROR: Missing Twitch credentials!")
        print("\n   Create a .env file with:")
        print("   TWITCH_BOT_TOKEN=your_token")
        print("   TWITCH_CLIENT_ID=your_client_id")
        print("   TWITCH_OAUTH_TOKEN=your_oauth_token")
        return

    try:
        print("  [1/3] Creating Twitch chat listener...")
        listener = SimpleTwitchChatListener(
            channel=CHANNEL,
            bot_token=BOT_TOKEN,
            nickname=BOT_NICK,
        )

        print("  [2/3] Creating real-time analyzer...")
        analyzer = RealtimeAnalyzer(
            chat_listener=listener,
            classifier=SimpleBoWClassifier(),
            on_update_callback=handle_update,
        )

        print("  [3/3] Starting chat listener...\n")

        # Add message handler
        original_callback = listener.on_message_callback

        async def combined_callback(msg_context):
            handle_message(msg_context)
            if original_callback:
                await original_callback(msg_context)

        listener.on_message_callback = combined_callback

        print("✅ Connected! Listening for messages...\n")
        print("=" * 100)
        print("  Messages will appear below. Statistics update every 30 seconds.")
        print("  Press Ctrl+C to stop.\n")
        print("=" * 100 + "\n")

        await analyzer.start()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 100)
        print("  ⏹️  Shutting down...")
        print("=" * 100)
        await analyzer.stop()
        print("✅ Shutdown complete\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())