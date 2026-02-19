import asyncio
import logging
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import deque, Counter
from typing import Optional, List, Dict, Callable, Tuple, Protocol
from enum import Enum

logger = logging.getLogger(__name__)

BASE_SENTIMENTS = ('Positive', 'Negative', 'Neutral')
NORMALIZED_SENTIMENTS = {
    'positive': 'Positive',
    'negative': 'Negative',
    'neutral': 'Neutral',
}
_EMOTE_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")


def normalize_sentiment(label: str) -> str:
    return NORMALIZED_SENTIMENTS.get(str(label).strip().lower(), 'Neutral')


def empty_scores() -> Dict[str, float]:
    return {sentiment: 0.0 for sentiment in BASE_SENTIMENTS}


def build_keyword_sets(sentiment_keywords: Dict[str, List[str]]) -> Dict[str, set]:
    return {
        sentiment: {keyword.lower() for keyword in keywords}
        for sentiment, keywords in sentiment_keywords.items()
    }


def _token_variants(token: str) -> List[str]:
    base = str(token).strip()
    if not base:
        return []
    stripped = _EMOTE_EDGE_RE.sub("", base)
    raw = (base, base.lower(), stripped, stripped.lower())
    out = []
    seen = set()
    for item in raw:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _score_to_bin(score: float) -> str:
    if score > 0.05:
        return 'Positive'
    if score < -0.05:
        return 'Negative'
    return 'Neutral'


def _score_to_confidence(score: float) -> float:
    # VADER style valence typically falls in [-4, 4]
    return max(0.25, min(0.95, abs(float(score)) / 4.0))


def _load_vader_style_lexicon(lexicon_path: Path) -> Dict:
    emote_sentiment = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            token = parts[0].strip()
            if not token:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue

            entry = {
                'score': score,
                'bin': _score_to_bin(score),
                'confidence': _score_to_confidence(score),
            }
            emote_sentiment[token] = entry
            low = token.lower()
            if low not in emote_sentiment:
                emote_sentiment[low] = entry
    return emote_sentiment


def load_emote_lexicon(emote_lexicon_path: Optional[str] = None) -> Tuple[Dict, Dict[str, List[str]]]:
    emote_sentiment = {}
    raw_keywords = {
        'Positive': [],
        'Negative': [],
        'Neutral': [],
    }

    if emote_lexicon_path and Path(emote_lexicon_path).exists():
        emote_sentiment = _load_vader_style_lexicon(Path(emote_lexicon_path))

    sentiment_keywords = {sentiment: [] for sentiment in BASE_SENTIMENTS}
    for sentiment, keywords in raw_keywords.items():
        sentiment_keywords[normalize_sentiment(sentiment)].extend(keywords)

    for emote_data in emote_sentiment.values():
        emote_data['bin'] = normalize_sentiment(emote_data.get('bin', 'Neutral'))

    return emote_sentiment, sentiment_keywords


# import vader
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not installed. Install with: pip install vaderSentiment")


# data classes
class MomentType(Enum):
    """chat moment labels."""
    HYPE = "hype"
    FAIL = "fail"
    NEUTRAL = "neutral"


@dataclass
class DetectedMoment:
    """single detected hype/fail moment."""
    moment_type: MomentType
    timestamp: float
    duration: float  # how long it lasted (seconds)
    sentiment_positive_pct: float  # % of messages that were positive
    sentiment_dominant: str  # most common sentiment
    message_count: int  # how many messages in the moment
    velocity_peak: float  # peak messages/second
    acceleration: float  # how quickly it accelerated
    top_messages: List[str] = field(default_factory=list)  # top 5 representative messages

    def to_dict(self):
        return {
            **asdict(self),
            'moment_type': self.moment_type.value,
        }


@dataclass
class WindowStatistics:
    """stats for one rolling window."""
    timestamp: float  # window timestamp
    duration: int  # window size (30, 60, 300)
    message_count: int
    sentiment_distribution: Dict[str, int]  # {'Positive': 45, 'Negative': 12, ...}
    sentiment_ratios: Dict[str, float]  # {'Positive': 0.75, ...}
    dominant_sentiment: str
    messages_per_second: float
    velocity_ratio: float
    top_emotes: List[tuple] = field(default_factory=list)  # [('POGGERS', 15), ...]


# sentiment classifier interface
class Classifier(Protocol):
    def predict(self, text: str) -> tuple[str, float]:
        ...


class LexiconClassifier:
    """emote + keyword classifier backed by VADER-style emote lexicon."""

    def __init__(self, emote_lexicon_path: Optional[str] = None):
        self.emote_sentiment, self.sentiment_keywords = load_emote_lexicon(
            emote_lexicon_path=emote_lexicon_path,
        )
        if not self.emote_sentiment:
            logger.warning("No emote lexicon found. Using minimal defaults.")
        self.sentiment_keyword_sets = build_keyword_sets(self.sentiment_keywords)

        logger.info(f"Loaded {len(self.emote_sentiment)} emotes across "
                    f"{len(self.sentiment_keywords)} sentiment categories")

    def predict(self, text: str) -> tuple[str, float]:
        """score text and return (label, confidence)."""
        if not text or not text.strip():
            return ('Neutral', 0.5)

        sentiment_scores = empty_scores()

        words = text.split()
        text_lower = text.lower()

        emote_count = 0
        for word in words:
            for cand in _token_variants(word):
                if cand in self.emote_sentiment:
                    emote_data = self.emote_sentiment[cand]
                    bin_type = emote_data['bin']
                    confidence = emote_data['confidence']
                    sentiment_scores[bin_type] += confidence * 3.0
                    emote_count += 1
                    break

        for sentiment, keywords in self.sentiment_keyword_sets.items():
            for keyword in keywords:
                if keyword in text_lower:
                    sentiment_scores[sentiment] += 0.5

        if len(text) > 3 and text.isupper() and emote_count == 0:
            sentiment_scores['Positive'] += 1.0

        max_score = max(sentiment_scores.values())

        if max_score == 0:
            return ('Neutral', 0.5)

        best_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]

        total_score = sum(sentiment_scores.values())
        if total_score > 0:
            confidence = min(0.95, max_score / total_score)
            if emote_count > 0:
                confidence = min(0.95, confidence * 1.1)
        else:
            confidence = 0.5

        return (best_sentiment, confidence)


class HybridClassifier:
    """blend emote scores with vader when emotes are weak."""

    def __init__(
            self,
            emote_lexicon_path: Optional[str] = None,
            use_vader: bool = True,
    ):
        """initialize hybrid classifier."""
        self.emote_sentiment, self.sentiment_keywords = load_emote_lexicon(
            emote_lexicon_path=emote_lexicon_path,
        )
        if not self.emote_sentiment:
            logger.warning("No emote lexicon found. Using minimal emote set.")
        self.sentiment_keyword_sets = build_keyword_sets(self.sentiment_keywords)

        self.use_vader = use_vader and VADER_AVAILABLE
        if self.use_vader:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self.vader = None
            if use_vader and not VADER_AVAILABLE:
                logger.warning("VADER requested but not available. Install: pip install vaderSentiment")

        logger.info(f"HybridClassifier initialized - "
                    f"Emotes: {len(self.emote_sentiment)}, "
                    f"VADER: {'enabled' if self.use_vader else 'disabled'}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Classify text using hybrid approach."""
        if not text or not text.strip():
            return ('Neutral', 0.5)

        emote_scores, emote_confidence = self._analyze_emotes(text)

        vader_scores = None
        vader_confidence = 0.0

        if self.use_vader and emote_confidence < 0.7:
            vader_scores, vader_confidence = self._analyze_with_vader(text)

        final_sentiment, final_confidence = self._combine_scores(
            emote_scores, emote_confidence,
            vader_scores, vader_confidence
        )

        return (final_sentiment, final_confidence)

    def _analyze_emotes(self, text: str) -> Tuple[Dict[str, float], float]:
        """Analyze text using emote-based BoW."""
        sentiment_scores = empty_scores()

        words = text.split()
        text_lower = text.lower()

        emote_count = 0
        emote_confidence_sum = 0.0

        for word in words:
            for cand in _token_variants(word):
                if cand in self.emote_sentiment:
                    emote_data = self.emote_sentiment[cand]
                    bin_type = emote_data['bin']
                    confidence = emote_data['confidence']
                    sentiment_scores[bin_type] += confidence * 3.0
                    emote_count += 1
                    emote_confidence_sum += confidence
                    break

        for sentiment, keywords in self.sentiment_keyword_sets.items():
            for keyword in keywords:
                if keyword in text_lower:
                    sentiment_scores[sentiment] += 0.5

        if len(text) > 3 and text.isupper() and emote_count == 0:
            sentiment_scores['Positive'] += 1.0

        max_score = max(sentiment_scores.values())
        total_score = sum(sentiment_scores.values())

        if max_score == 0:
            confidence = 0.0
        elif emote_count > 0:
            avg_emote_conf = emote_confidence_sum / emote_count
            confidence = min(0.95, avg_emote_conf * (max_score / total_score))
        else:
            confidence = min(0.7, max_score / total_score) if total_score > 0 else 0.0

        return sentiment_scores, confidence

    def _analyze_with_vader(self, text: str) -> Tuple[Optional[Dict[str, float]], float]:
        """Analyze text using VADER."""
        if not self.vader:
            return None, 0.0

        vader_result = self.vader.polarity_scores(text)

        sentiment_scores = empty_scores()

        pos = vader_result['pos']
        neg = vader_result['neg']
        neu = vader_result['neu']
        compound = vader_result['compound']

        if compound > 0.05:
            sentiment_scores['Positive'] = pos * 10
        elif compound < -0.05:
            sentiment_scores['Negative'] = neg * 10
        else:
            sentiment_scores['Neutral'] = neu * 10

        confidence = abs(compound)
        return sentiment_scores, confidence

    def _combine_scores(
            self,
            emote_scores: Dict[str, float],
            emote_confidence: float,
            vader_scores: Optional[Dict[str, float]],
            vader_confidence: float
    ) -> Tuple[str, float]:
        """Combine emote and VADER scores."""
        if vader_scores is None:
            max_score = max(emote_scores.values())
            if max_score == 0:
                return ('Neutral', 0.5)
            best_sentiment = max(emote_scores.items(), key=lambda x: x[1])[0]
            return (best_sentiment, max(0.5, emote_confidence))

        if emote_confidence > 0.7:
            emote_weight = 0.9
            vader_weight = 0.1
        elif emote_confidence < 0.3:
            emote_weight = 0.2
            vader_weight = 0.8
        else:
            total_conf = emote_confidence + vader_confidence
            if total_conf > 0:
                emote_weight = emote_confidence / total_conf
                vader_weight = vader_confidence / total_conf
            else:
                emote_weight = 0.5
                vader_weight = 0.5

        combined_scores = {
            sentiment: (emote_scores[sentiment] * emote_weight +
                        vader_scores[sentiment] * vader_weight)
            for sentiment in emote_scores.keys()
        }

        max_score = max(combined_scores.values())
        if max_score == 0:
            return ('Neutral', 0.5)

        best_sentiment = max(combined_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(combined_scores.values())
        score_confidence = max_score / total_score if total_score > 0 else 0.5

        final_confidence = min(0.95, (
                score_confidence * 0.6 +
                emote_confidence * emote_weight * 0.2 +
                vader_confidence * vader_weight * 0.2
        ))

        return (best_sentiment, final_confidence)


# sentiment aggregation
class SentimentAggregator:
    """keeps rolling sentiment windows."""

    def __init__(self, window_sizes: List[int] = None):
        if window_sizes is None:
            window_sizes = [30, 60, 300]

        self.window_sizes = window_sizes

        self.windows = {
            size: deque() for size in window_sizes
        }

        self.all_messages = deque(maxlen=10000)

    def add(self, timestamp: float, sentiment: str, confidence: float,
            text: str, context: Dict = None):
        """add one classified message."""
        msg = {
            'timestamp': timestamp,
            'sentiment': sentiment,
            'confidence': confidence,
            'text': text,
            'context': context or {},
        }

        now = time.time()
        for window_size, window in self.windows.items():
            window.append(msg)
            while window and window[0]['timestamp'] < now - window_size:
                window.popleft()

        self.all_messages.append(msg)

    def get_window_stats(self, window_size: int = 30) -> WindowStatistics:
        """return stats for a given window size."""
        if window_size not in self.windows:
            raise ValueError(f"Invalid window size: {window_size}")

        window = self.windows[window_size]
        now = time.time()

        if not window:
            return WindowStatistics(
                timestamp=now,
                duration=window_size,
                message_count=0,
                sentiment_distribution={},
                sentiment_ratios={},
                dominant_sentiment='Neutral',
                messages_per_second=0.0,
                velocity_ratio=1.0,
                top_emotes=[],
            )

        sentiment_counts = Counter(msg['sentiment'] for msg in window)
        total_msgs = len(window)

        sentiment_ratios = {
            s: count / total_msgs for s, count in sentiment_counts.items()
        }

        dominant = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else 'Neutral'

        time_span = now - window[0]['timestamp']
        msgs_per_sec = total_msgs / max(time_span, 1.0)

        emote_counter = Counter()
        for msg in window:
            if 'emotes' in msg.get('context', {}):
                emotes = msg['context']['emotes']
                if isinstance(emotes, list):
                    emote_counter.update(emotes)

        top_emotes = emote_counter.most_common(10)

        return WindowStatistics(
            timestamp=now,
            duration=window_size,
            message_count=total_msgs,
            sentiment_distribution=dict(sentiment_counts),
            sentiment_ratios=sentiment_ratios,
            dominant_sentiment=dominant,
            messages_per_second=msgs_per_sec,
            velocity_ratio=1.0,
            top_emotes=top_emotes,
        )

    def get_all_stats(self) -> Dict[int, WindowStatistics]:
        """get stats for all windows"""
        return {
            size: self.get_window_stats(size)
            for size in self.window_sizes
        }


# moment detection
class MomentDetector:
    """detects hype/fail spikes from sentiment and velocity."""

    def __init__(
            self,
            hype_sentiment_threshold: float = 0.7,
            fail_sentiment_threshold: float = 0.6,
            velocity_spike_threshold: float = 2.0,
            min_duration: float = 5.0,
            cooldown: float = 30.0,
    ):
        self.hype_sentiment_threshold = hype_sentiment_threshold
        self.fail_sentiment_threshold = fail_sentiment_threshold
        self.velocity_spike_threshold = velocity_spike_threshold
        self.min_duration = min_duration
        self.cooldown = cooldown

        self.last_moment_time = 0.0
        self.detected_moments = deque(maxlen=100)

    def check(
            self,
            current_stats: WindowStatistics,
            previous_stats: Optional[WindowStatistics],
            baseline_velocity: float,
    ) -> Optional[DetectedMoment]:
        """return a moment when thresholds are met, else None."""
        now = time.time()

        if now - self.last_moment_time < self.cooldown:
            return None

        if current_stats.message_count < 20:
            return None

        positive_pct = current_stats.sentiment_ratios.get('Positive', 0)
        negative_pct = current_stats.sentiment_ratios.get('Negative', 0)

        velocity_spike = current_stats.messages_per_second > (
                baseline_velocity * self.velocity_spike_threshold
        )

        acceleration = 0.0
        if previous_stats and previous_stats.messages_per_second > 0:
            acceleration = (
                    (current_stats.messages_per_second - previous_stats.messages_per_second) /
                    previous_stats.messages_per_second
            )

        hype_conditions = [
            positive_pct > self.hype_sentiment_threshold,
            velocity_spike or acceleration > 1.0,
            current_stats.message_count > 30,
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
        """extract top representative messages from a window"""
        return []

    def get_recent_moments(self, n: int = 10) -> List[DetectedMoment]:
        """get last n moments."""
        return list(self.detected_moments)[-n:]


# realtime analysis
class RealtimeAnalyzer:
    """wires listener, classifier, aggregation, and moment detection."""

    def __init__(
            self,
            chat_listener,
            classifier: Optional[Classifier] = None,
            on_update_callback: Optional[Callable] = None,
    ):
        self.chat_listener = chat_listener
        self.classifier = classifier or LexiconClassifier()
        self.on_update_callback = on_update_callback

        self.aggregator = SentimentAggregator()
        self.moment_detector = MomentDetector()

        self.prev_window_stats = None
        self.baseline_velocity = 1.0

        self.total_messages_processed = 0
        self.total_moments_detected = 0

        self.is_running = False
        logger.info("RealtimeAnalyzer initialized")

    async def process_messages(self):
        self.is_running = True
        last_stats_time = time.time()

        while self.is_running:
            try:
                msg_context = await self.chat_listener.get_message()

                if msg_context is None:
                    await asyncio.sleep(0.1)
                    continue

                sentiment, confidence = self.classifier.predict(msg_context.text)
                msg_context.sentiment = sentiment
                msg_context.confidence = confidence

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

                now = time.time()
                if now - last_stats_time > 30:
                    await self._update_statistics()
                    last_stats_time = now

            except Exception as e:
                logger.error(f"Error in process_messages: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def _update_statistics(self):
        try:
            current_stats = self.aggregator.get_window_stats(window_size=30)

            if current_stats.messages_per_second > 0:
                self.baseline_velocity = (self.baseline_velocity * 0.9 +
                                          current_stats.messages_per_second * 0.1)

            moment = self.moment_detector.check(
                current_stats,
                self.prev_window_stats,
                self.baseline_velocity
            )

            if moment:
                self.total_moments_detected += 1
                try:
                    logger.info(
                        f"moment detected: {moment.moment_type.value} "
                        f"({moment.message_count} msgs, {moment.velocity_peak:.1f} msg/s)"
                    )
                except Exception:
                    logger.info("moment detected")

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


# example usage
async def example_update_handler(update_data: Dict):
    """callback for pipeline updates"""
    moment = update_data.get('moment_detected')
    if moment:
        print(f"\nmoment detected: {moment.moment_type.value.upper()}")
        print(f"   Time: {moment.timestamp}")
        print(f"   Messages: {moment.message_count}")
        print(f"   Velocity: {moment.velocity_peak:.1f} msg/s")
        print(f"   Sentiment: {moment.sentiment_positive_pct * 100:.0f}% positive")

    stats = update_data['all_stats']
    if 30 in stats:
        w30 = stats[30]
        print(f"\n30-second window:")
        print(f"   {w30.message_count} messages")
        print(f"   Sentiment: {w30.sentiment_dominant}")
        print(f"   {w30.messages_per_second:.1f} msg/s")

