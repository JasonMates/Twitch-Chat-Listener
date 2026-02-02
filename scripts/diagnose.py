"""
Test Message Parser
===================

Tests if our message parser correctly handles real Twitch IRC lines.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.twitch_listener import SimpleTwitchChatListener

# Real message from Twitch
real_messages = [
    "@badge-info=;badges=;client-nonce=71575a072e594af48cddb85d02154f0d;color=#FF69B4;display-name=brutherinchrist;emote-only=1;emotes=555555558:0-1;first-msg=0;flags=;id=4bf438cf-6c52-4201-a633-7d6a1fdcae63;mod=0;returning-chatter=0;room-id=38251312;subscriber=0;tmi-sent-ts=1770002281642;turbo=0;user-id=1007702856;user-type= :brutherinchrist!brutherinchrist@brutherinchrist.tmi.twitch.tv PRIVMSG #paymoneywubby ::",
    "@badge-info=;badges=;client-nonce=ec254bcc10944557ac5eec1d513ec392;color=#8A2BE2;display-name=TinyDumDog;emotes=;first-msg=0;flags=;id=212d1e5b-02c8-4b2a-aa01-0b63c73d25fd;mod=0;returning-chatter=0;room-id=38251312;subscriber=0;tmi-sent-ts=1770002282261;turbo=0;user-id=515969109;user-type= :tinydumdog!tinydumdog@tinydumdog.tmi.twitch.tv PRIVMSG #paymoneywubby :lol",
    "@badge-info=subscriber/23;badges=subscriber/18,bits-charity/1;client-nonce=be762a3b471e47698e32a49e4c6786eb;color=#FF0000;display-name=fantakilla1;emotes=;first-msg=0;flags=;id=fe2a768e-5e4a-4f81-a462-5ff3f75dfca2;mod=0;returning-chatter=0;room-id=38251312;subscriber=1;tmi-sent-ts=1770002294642;turbo=0;user-id=58108597;user-type= :fantakilla1!fantakilla1@fantakilla1.tmi.twitch.tv PRIVMSG #paymoneywubby :he does look pretty tired ngl",
]

print("\n" + "="*80)
print("  Testing Message Parser")
print("="*80 + "\n")

async def test_parser():
    # Create a dummy listener just to access the parser methods
    listener = SimpleTwitchChatListener(
        channel='paymoneywubby',
        bot_token='oauth:test'
    )

    for i, line in enumerate(real_messages, 1):
        print(f"Test {i}:")
        print(f"  Raw: {line[:80]}...\n")

        try:
            msg = await listener._parse_message(line)
            if msg:
                print(f"  ✅ Parsed!")
                print(f"     Username: {msg.username}")
                print(f"     Text: {msg.text}")
                print(f"     Sentiment: {msg.sentiment}")
                print(f"     Emotes: {msg.emotes}")
            else:
                print(f"  ❌ Parser returned None")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

asyncio.run(test_parser())

print("="*80)