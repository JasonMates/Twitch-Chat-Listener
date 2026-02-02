import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = DATA_DIR / 'models'

# Twitch credentials
TWITCH_BOT_TOKEN = os.getenv('TWITCH_BOT_TOKEN')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_OAUTH_TOKEN = os.getenv('TWITCH_OAUTH_TOKEN')
TWITCH_BOT_NICK = os.getenv('TWITCH_BOT_NICK', 'StreamAnalysisBot')

# Sentiment config
SENTIMENT_CLASSES = ['Positive', 'Negative', 'Neutral', 'Enthusiastic', 'Anxious', 'Noise']
NUM_CLASSES = len(SENTIMENT_CLASSES)

# Moment detection thresholds
HYPE_SENTIMENT_THRESHOLD = 0.70        # 70% positive/enthusiastic
FAIL_SENTIMENT_THRESHOLD = 0.60        # 60% negative
VELOCITY_SPIKE_THRESHOLD = 2.0         # 2x baseline
MIN_MOMENT_DURATION = 20               # seconds

# Window sizes
WINDOW_SIZES = [30, 60, 300]  # 30s, 1m, 5m

# API
TWITCH_API_BASE = 'https://api.twitch.tv/helix'
CONTEXT_UPDATE_INTERVAL = 300          # 5 minutes