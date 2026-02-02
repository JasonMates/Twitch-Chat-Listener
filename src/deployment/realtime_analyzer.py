import asyncio
import logging
import time
from dataclasses import dataclass, asdict, field
from collections import deque, Counter
from typing import Optional, List, Dict, Callable
from enum import Enum
import math

logger = logging.getLogger(__name__)


# Data Classes
class MomentType(Enum):
    """Types of detected moments"""
    HYPE = "hype"
    FAIL = "fail"
    NEUTRAL = "neutral"


@dataclass
class DetectedMoment:
    """A detected hype or fail moment in chat"""
    moment_type: MomentType
    timestamp: float
    duration: float  # How long it lasted (seconds)
    sentiment_positive_pct: float  # % of messages that were positive
    sentiment_dominant: str  # Most common sentiment
    message_count: int  # How many messages in the moment
    velocity_peak: float  # Peak messages/second
    acceleration: float  # How quickly did it accelerate
    top_messages: List[str] = field(default_factory=list)  # Top 5 representative messages

    def to_dict(self):
        return {
            **asdict(self),
            'moment_type': self.moment_type.value,
        }


@dataclass
class WindowStatistics:
    """statistics for a rolling time window"""
    timestamp: float  # Window timestamp
    duration: int  # Window size (30, 60, 300)
    message_count: int
    sentiment_distribution: Dict[str, int]  # {'Positive': 45, 'Negative': 12, ...}
    sentiment_ratios: Dict[str, float]  # {'Positive': 0.75, ...}
    dominant_sentiment: str
    messages_per_second: float
    velocity_ratio: float
    top_emotes: List[tuple] = field(default_factory=list)  # [('POGGERS', 15), ...]


# ============================================================================
# Sentiment Classifier Interface
# ============================================================================

class SentimentClassifier:
    """
    Interface for sentiment classification.
    simple BoW baseline is included.
    """

    def __init__(self):
        self.model = None

    def predict(self, text: str) -> tuple[str, float]:
        """
        Classify text sentiment.

        Args:
            text: Input message text

        Returns:
            (sentiment_label, confidence_score)
            Example: ('Positive', 0.95)
        """
        raise NotImplementedError


class SimpleBoWClassifier(SentimentClassifier):
    """
    Simple bag of words baseline classifier for testing for prototyping
    """

    def __init__(self):
        super().__init__()

        # simple word lists for each sentiment
        self.sentiment_keywords = {
            'Positive': {
                'pog', 'poggers', 'yes', 'yep', 'yeah', 'wicked', 'based',
                'omegalul', 'lul', 'lulw', 'kekw', 'noice', 'nice', 'good',
                'great', 'amazing', 'insane', 'lit', 'fire', 'sick', '+2',
            },
            'Negative': {
                'no', 'nah', 'nooo', 'bad', 'terrible', 'awful', 'eww', 'gross',
                'hate', 'sucks', 'copium', 'sadge', 'ban', 'gg', 'loss', 'fail',
                'trash', 'awful', 'pepeluh', 'clueless', '-2',
            },
            'Enthusiastic': {
                'poggers', 'pog', 'omegalul', 'yes', 'yesss', 'www', 'hype',
                'let\'s go', 'go', 'whoa', 'wow', 'omg', 'wait', 'what',
                'clutch', 'carry', 'pentakill', 'mega', 'gigachad', 'icant'
            },
            'Anxious': {
                'monkas', 'monka', 'ono', 'oh no', 'uh oh', 'hmm', 'uhh',
                'scared', 'worried', 'nervous', 'tension', 'close',
            },
            'Neutral': {
                'ok', 'noted', 'sure', 'got it', 'thanks', 'bye', 'hi',
                'hello', 'hey', 'what', 'huh', 'pause', 'wait', 'download',
            },
        }

        # emote sentiment mapping
        self.emote_sentiment = {
            'POGGERS': ('Enthusiastic', 0.9),
            'Pog': ('Positive', 0.8),
            'OMEGALUL': ('Positive', 0.85),
            'LUL': ('Positive', 0.8),
            'KEKW': ('Positive', 0.8),
            'Sadge': ('Negative', 0.85),
            'ResidentSleeper': ('Negative', 0.75),
            'LULW': ('Positive', 0.8),
            'Clueless': ('Negative', 0.7),
            'MonkaS': ('Anxious', 0.85),
            'MONKAS': ('Anxious', 0.85),
            'FeelsGoodMan': ('Positive', 0.8),
            'FeelsStrongMan': ('Negative', 0.75),
        }

    def predict(self, text: str) -> tuple[str, float]:
        """classify text using simple keyword matching"""
        text_lower = text.lower()
        scores = {s: 0 for s in self.sentiment_keywords.keys()}

        # score each sentiment
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment] += 1

        # check for emotes (higher weight)
        for emote, (sentiment, confidence) in self.emote_sentiment.items():
            if emote in text:
                scores[sentiment] += 3  # Higher weight for emotes

        # handle all caps (indicates enthusiasm or frustration)
        if len(text) > 3 and text.isupper():
            scores['Enthusiastic'] += 2

        # determine winner
        max_score = max(scores.values())
        if max_score == 0:
            return ('Neutral', 0.5)  # default neutral

        best_sentiment = [s for s, score in scores.items() if score == max_score][0]
        confidence = min(0.95, 0.5 + (max_score / 10))  # Cap at 0.95

        return (best_sentiment, confidence)


# Sentiment Aggregation (Rolling Windows)
class SentimentAggregator:
    """
    Aggregate sentiment over rolling time windows
    Maintains statistics windows.
    """

    def __init__(self, window_sizes: List[int] = None):
        if window_sizes is None:
            window_sizes = [30, 60, 300]  # 30s, 1m, 5m

        self.window_sizes = window_sizes

        # Messages in each window: deque of (timestamp, sentiment, confidence, text)
        self.windows = {
            size: deque() for size in window_sizes
        }

        # All messages (for backlog)
        self.all_messages = deque(maxlen=10000)

    def add(self, timestamp: float, sentiment: str, confidence: float,
            text: str, context: Dict = None):
        """
        Add a classified message to the aggregator.

        Args:
            timestamp: Unix timestamp
            sentiment: Sentiment label (e.g., 'Positive')
            confidence: Confidence score (0-1)
            text: Original message text
            context: Optional additional context
        """
        msg = {
            'timestamp': timestamp,
            'sentiment': sentiment,
            'confidence': confidence,
            'text': text,
            'context': context or {},
        }

        # add to all windows
        now = time.time()
        for window_size, window in self.windows.items():
            window.append(msg)
            # remove expired messages
            while window and window[0]['timestamp'] < now - window_size:
                window.popleft()

        self.all_messages.append(msg)

    def get_window_stats(self, window_size: int = 30) -> WindowStatistics:
        """
        Get statistics for a specific window.

        Args:
            window_size: 30, 60, or 300 (seconds)

        Returns:
            WindowStatistics with distribution, ratios, dominant sentiment, etc.
        """
        if window_size not in self.windows:
            raise ValueError(f"Unknown window size: {window_size}")

        window = self.windows[window_size]
        now = time.time()

        if not window:
            return WindowStatistics(
                timestamp=now,
                duration=window_size,
                message_count=0,
                sentiment_distribution={},
                sentiment_ratios={},
                dominant_sentiment='None',
                messages_per_second=0,
                velocity_ratio=0,
            )

        # count sentiments
        sentiment_counts = Counter(msg['sentiment'] for msg in window)
        total = len(window)

        # calculate ratios
        sentiment_ratios = {
            sentiment: count / total
            for sentiment, count in sentiment_counts.items()
        }

        # dominant sentiment
        dominant = sentiment_counts.most_common(1)[0][0]

        # messages per second
        time_span = window[-1]['timestamp'] - window[0]['timestamp']
        mps = total / max(time_span, 1)

        # top emotes
        all_emotes = []
        for msg in window:
            emotes = msg['context'].get('emotes', [])
            all_emotes.extend(emotes)
        emote_counts = Counter(all_emotes)
        top_emotes = emote_counts.most_common(5)

        return WindowStatistics(
            timestamp=now,
            duration=window_size,
            message_count=total,
            sentiment_distribution=dict(sentiment_counts),
            sentiment_ratios=sentiment_ratios,
            dominant_sentiment=dominant,
            messages_per_second=mps,
            velocity_ratio=0,  # Set by moment detector
            top_emotes=top_emotes,
        )

    def get_all_stats(self) -> Dict[int, WindowStatistics]:
        return {
            size: self.get_window_stats(size)
            for size in self.window_sizes
        }


# Moment Detection
class MomentDetector:
    """
    Detect hype moments and fail moments in chat.

    Uses a combination of signals:
    - Sentiment spike (% of positive/negative messages)
    - Velocity spike (acceleration in messages/second)
    - Duration filter (must sustain for 20+ seconds)
    """

    def __init__(self):
        self.detected_moments: deque = deque(maxlen=100)
        self.last_moment_time = 0

        # Configuration
        self.hype_sentiment_threshold = 0.70  # 70%+ positive/enthusiastic
        self.fail_sentiment_threshold = 0.60  # 60%+ negative
        self.velocity_spike_threshold = 2.0  # 2x baseline
        self.min_duration = 20  # Must last 20 seconds
        self.moment_cooldown = 10  # Don't detect moments within 10 seconds

    def check(self, current_stats: WindowStatistics,
              prev_stats: Optional[WindowStatistics] = None,
              baseline_velocity: float = None) -> Optional[DetectedMoment]:
        """
        Check if current window represents a moment.

        Args:
            current_stats: Stats from current 30-second window
            prev_stats: Stats from previous 30-second window (for acceleration)
            baseline_velocity: Average baseline velocity for velocity_ratio

        Returns:
            DetectedMoment if detected, None otherwise
        """
        now = time.time()

        # cooldown check
        if now - self.last_moment_time < self.moment_cooldown:
            return None

        if current_stats.message_count < 10:  # Need at least 10 messages
            return None

        # sentiment analysis
        positive_pct = current_stats.sentiment_ratios.get('Positive', 0) + \
                       current_stats.sentiment_ratios.get('Enthusiastic', 0)
        negative_pct = current_stats.sentiment_ratios.get('Negative', 0)

        # velocity analysis
        acceleration = 0
        if prev_stats and prev_stats.message_count > 0:
            acceleration = (current_stats.messages_per_second - prev_stats.messages_per_second) / \
                           (prev_stats.messages_per_second + 0.1)

        velocity_spike = baseline_velocity and \
                         current_stats.messages_per_second > baseline_velocity * self.velocity_spike_threshold

        # detect hype moment
        hype_conditions = [
            positive_pct > self.hype_sentiment_threshold,
            velocity_spike or acceleration > 1.0,
            current_stats.message_count > 50,
        ]

        if all(hype_conditions):
            moment = DetectedMoment(
                moment_type=MomentType.HYPE,
                timestamp=now,
                duration=self.min_duration,
                sentiment_positive_pct=positive_pct,
                sentiment_dominant=current_stats.dominant_sentiment,
                message_count=current_stats.message_count,
                velocity_peak=current_stats.messages_per_second,
                acceleration=acceleration,
                top_messages=self._extract_top_messages(current_stats),
            )
            self.last_moment_time = now
            self.detected_moments.append(moment)
            return moment

        # detect fail moment
        fail_conditions = [
            negative_pct > self.fail_sentiment_threshold,
            velocity_spike or acceleration > 0.5,
            current_stats.message_count > 30,
        ]

        if all(fail_conditions):
            moment = DetectedMoment(
                moment_type=MomentType.FAIL,
                timestamp=now,
                duration=self.min_duration,
                sentiment_positive_pct=positive_pct,
                sentiment_dominant=current_stats.dominant_sentiment,
                message_count=current_stats.message_count,
                velocity_peak=current_stats.messages_per_second,
                acceleration=acceleration,
                top_messages=self._extract_top_messages(current_stats),
            )
            self.last_moment_time = now
            self.detected_moments.append(moment)
            return moment

        return None

    @staticmethod
    def _extract_top_messages(stats: WindowStatistics, n: int = 5) -> List[str]:
        """Extract top representative messages from a window"""
        # Placeholder: would need access to actual messages
        # For now, return empty list
        return []

    def get_recent_moments(self, n: int = 10) -> List[DetectedMoment]:
        """Get last N detected moments"""
        return list(self.detected_moments)[-n:]



# Realtime Analysis
class RealtimeAnalyzer:
    """
    Orchestrates the full pipeline:
    Chat Listener -> Sentiment Classifier -> Aggregation -> Moment Detection -> Dashboard
    """

    def __init__(
            self,
            chat_listener,
            classifier: Optional[SentimentClassifier] = None,
            on_update_callback: Optional[Callable] = None,
    ):
        self.chat_listener = chat_listener
        self.classifier = classifier or SimpleBoWClassifier()
        self.on_update_callback = on_update_callback

        # pipeline components
        self.aggregator = SentimentAggregator()
        self.moment_detector = MomentDetector()

        # statistics tracking
        self.prev_window_stats = None
        self.baseline_velocity = 1.0

        # metrics
        self.total_messages_processed = 0
        self.total_moments_detected = 0

        self.is_running = False
        logger.info("RealtimeAnalyzer initialized")

    async def process_messages(self):
        self.is_running = True
        last_stats_time = time.time()

        while self.is_running:
            try:
                # get next message (non-blocking)
                msg_context = await self.chat_listener.get_message()

                if msg_context is None:
                    await asyncio.sleep(0.1)
                    continue

                # classify sentiment
                sentiment, confidence = self.classifier.predict(msg_context.text)
                msg_context.sentiment = sentiment
                msg_context.confidence = confidence

                # add to aggregator with context
                self.aggregator.add(
                    timestamp=msg_context.timestamp,
                    sentiment=sentiment,
                    confidence=confidence,
                    text=msg_context.text,
                    context={
                        'emotes': msg_context.emotes,
                        'username': msg_context.username,
                        'game': msg_context.game,
                        'velocity_ratio': msg_context.velocity_ratio,
                    }
                )

                self.total_messages_processed += 1

                # periodically check for moments and update stats
                now = time.time()
                if now - last_stats_time > 30:  # every 30 seconds
                    await self._update_statistics()
                    last_stats_time = now

            except Exception as e:
                logger.error(f"Error in process_messages: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def _update_statistics(self):
        try:
            # get current window stats
            current_stats = self.aggregator.get_window_stats(window_size=30)

            # update baseline velocity
            if current_stats.messages_per_second > 0:
                self.baseline_velocity = (self.baseline_velocity * 0.9 +
                                          current_stats.messages_per_second * 0.1)

            # check for moments
            moment = self.moment_detector.check(
                current_stats,
                self.prev_window_stats,
                self.baseline_velocity
            )

            if moment:
                self.total_moments_detected += 1
                try:
                    logger.info(f"Moment detected: {moment.moment_type.value} "
                                f"({moment.message_count} msgs, {moment.velocity_peak:.1f} msg/s)")
                except:
                    logger.info("Moment detected")

            # call update callback
            if self.on_update_callback:
                await self.on_update_callback({
                    'all_stats': self.aggregator.get_all_stats(),
                    'moment_detected': moment,
                    'total_messages': self.total_messages_processed,
                    'total_moments': self.total_moments_detected,
                    'baseline_velocity': self.baseline_velocity,
                    'recent_moments': self.moment_detector.get_recent_moments(10),
                })

            self.prev_window_stats = current_stats

        except Exception as e:
            logger.error(f"Error updating statistics: {e}", exc_info=True)

    async def start(self):
        """start analysis pipeline"""
        logger.info("Starting RealtimeAnalyzer")

        # start chat listener and message processor in parallel
        listener_task = asyncio.create_task(self.chat_listener.start())
        processor_task = asyncio.create_task(self.process_messages())

        try:
            await asyncio.gather(listener_task, processor_task)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        self.is_running = False
        await self.chat_listener.stop()
        logger.info("RealtimeAnalyzer stopped")

    def get_statistics(self) -> Dict:
        """get current stats"""
        return {
            'total_messages': self.total_messages_processed,
            'total_moments': self.total_moments_detected,
            'baseline_velocity': self.baseline_velocity,
            'window_stats': self.aggregator.get_all_stats(),
            'recent_moments': self.moment_detector.get_recent_moments(10),
        }



# Example Usage
async def example_update_handler(update_data: Dict):
    """callback for pipeline updates"""
    moment = update_data.get('moment_detected')
    if moment:
        print(f"\n🎯 MOMENT DETECTED: {moment.moment_type.value.upper()}")
        print(f"   Time: {moment.timestamp}")
        print(f"   Messages: {moment.message_count}")
        print(f"   Velocity: {moment.velocity_peak:.1f} msg/s")
        print(f"   Sentiment: {moment.sentiment_positive_pct * 100:.0f}% positive")

    stats = update_data['all_stats']
    if 30 in stats:
        w30 = stats[30]
        print(f"\n📊 30-second window:")
        print(f"   {w30.message_count} messages")
        print(f"   Sentiment: {w30.sentiment_dominant}")
        print(f"   {w30.messages_per_second:.1f} msg/s")