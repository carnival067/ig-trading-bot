"""Repository for news articles, crisis alerts, economic events, and geopolitical risk scores.

Provides async CRUD operations for the News Engine's persistence layer.
Validates Requirements 23.1, 23.3, 23.15.
"""

from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CrisisAlert, EconomicEvent, GeopoliticalRiskScore, NewsArticle


class NewsRepository:
    """Async repository for news-related database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ─── News Articles ────────────────────────────────────────────────────────

    async def store_article(self, article_data: dict) -> NewsArticle:
        """Persist a news article.

        Args:
            article_data: Dictionary with fields matching NewsArticle columns.

        Returns:
            The persisted NewsArticle instance.
        """
        article = NewsArticle(**article_data)
        self._session.add(article)
        await self._session.flush()
        return article

    async def get_recent_articles(self, limit: int = 50) -> list[NewsArticle]:
        """Get the most recent articles ordered by received_at descending.

        Args:
            limit: Maximum number of articles to return (default 50).

        Returns:
            List of NewsArticle instances ordered by most recent first.
        """
        stmt = (
            select(NewsArticle)
            .order_by(NewsArticle.received_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def article_exists(self, body_hash: str) -> bool:
        """Check for duplicate articles by body hash.

        Args:
            body_hash: SHA-256 hash of the article body content.

        Returns:
            True if an article with this hash already exists, False otherwise.
        """
        stmt = select(NewsArticle).where(NewsArticle.body_hash == body_hash).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def get_articles_by_impact(
        self, impact_level: str, since: datetime
    ) -> list[NewsArticle]:
        """Filter articles by impact level within a time window.

        Args:
            impact_level: Impact classification (HIGH, MEDIUM, LOW).
            since: Only return articles received after this timestamp.

        Returns:
            List of matching NewsArticle instances ordered by received_at desc.
        """
        stmt = (
            select(NewsArticle)
            .where(
                NewsArticle.impact_level == impact_level,
                NewsArticle.received_at >= since,
            )
            .order_by(NewsArticle.received_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_articles_by_category(
        self, category: str, since: datetime
    ) -> list[NewsArticle]:
        """Filter articles by category within a time window.

        Args:
            category: News category (e.g., monetary_policy, geopolitical_conflict).
            since: Only return articles received after this timestamp.

        Returns:
            List of matching NewsArticle instances ordered by received_at desc.
        """
        stmt = (
            select(NewsArticle)
            .where(
                NewsArticle.category == category,
                NewsArticle.received_at >= since,
            )
            .order_by(NewsArticle.received_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # Keep backward-compatible alias
    async def get_articles_by_region(
        self, category: str, since: datetime
    ) -> list[NewsArticle]:
        """Alias for get_articles_by_category (backward compatibility)."""
        return await self.get_articles_by_category(category, since)

    async def get_high_impact_articles_in_window(
        self, since: datetime | None = None, sentiment_threshold: float = -0.7, *, minutes: int = 10
    ) -> list[NewsArticle]:
        """Get high-impact articles within a time window for crisis detection.

        Used to detect crisis events per Requirement 23.7: 3+ HIGH-impact articles
        with sentiment < -0.7 within a 10-minute window.

        Args:
            since: Only return articles received after this timestamp.
                   If None, defaults to `minutes` ago from now.
            sentiment_threshold: Maximum sentiment score to include (default -0.7).
                                 Articles with sentiment <= this value are returned.
            minutes: Fallback lookback window in minutes if `since` is not provided (default 10).

        Returns:
            List of HIGH-impact NewsArticle instances within the window
            that have sentiment at or below the threshold.
        """
        cutoff = since if since is not None else datetime.now(timezone.utc) - timedelta(minutes=minutes)
        stmt = (
            select(NewsArticle)
            .where(
                NewsArticle.impact_level == "HIGH",
                NewsArticle.received_at >= cutoff,
                NewsArticle.sentiment_score <= sentiment_threshold,
            )
            .order_by(NewsArticle.received_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ─── Crisis Alerts ────────────────────────────────────────────────────────

    async def create_crisis_alert(self, alert_data: dict) -> CrisisAlert:
        """Create a new crisis alert.

        Args:
            alert_data: Dictionary with fields matching CrisisAlert columns.

        Returns:
            The persisted CrisisAlert instance.
        """
        alert = CrisisAlert(**alert_data)
        self._session.add(alert)
        await self._session.flush()
        return alert

    async def get_active_crises(self) -> list[CrisisAlert]:
        """Get all unresolved crisis alerts.

        Returns:
            List of CrisisAlert instances where resolved_at is None.
        """
        stmt = (
            select(CrisisAlert)
            .where(CrisisAlert.resolved_at.is_(None))
            .order_by(CrisisAlert.started_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def resolve_crisis(
        self,
        alert_id: str | UUID,
        resolved_at: datetime | None = None,
        escalated: bool = False,
    ) -> CrisisAlert | None:
        """Mark a crisis alert as resolved by setting resolved_at timestamp.

        Args:
            alert_id: UUID string or UUID of the crisis alert to resolve.
            resolved_at: Timestamp when the crisis was resolved. Defaults to now (UTC).
            escalated: Whether the crisis was escalated to the kill switch.

        Returns:
            The updated CrisisAlert, or None if not found.
        """
        if isinstance(alert_id, str):
            alert_id = UUID(alert_id)
        stmt = select(CrisisAlert).where(CrisisAlert.id == alert_id)
        result = await self._session.execute(stmt)
        alert = result.scalar_one_or_none()
        if alert is None:
            return None
        alert.resolved_at = resolved_at if resolved_at is not None else datetime.now(timezone.utc)
        alert.active = False
        if escalated:
            alert.escalated_to_kill_switch = True
        await self._session.flush()
        return alert

    async def escalate_crisis(self, alert_id: str | UUID) -> CrisisAlert | None:
        """Mark a crisis alert as escalated to the kill switch.

        Args:
            alert_id: UUID string or UUID of the crisis alert to escalate.

        Returns:
            The updated CrisisAlert, or None if not found.
        """
        if isinstance(alert_id, str):
            alert_id = UUID(alert_id)
        stmt = select(CrisisAlert).where(CrisisAlert.id == alert_id)
        result = await self._session.execute(stmt)
        alert = result.scalar_one_or_none()
        if alert is None:
            return None
        alert.escalated_to_kill_switch = True
        await self._session.flush()
        return alert

    # Keep backward-compatible alias
    async def update_crisis_escalation(self, crisis_id: UUID) -> CrisisAlert | None:
        """Alias for escalate_crisis (backward compatibility)."""
        return await self.escalate_crisis(crisis_id)

    # ─── Economic Events ──────────────────────────────────────────────────────

    async def store_events(self, events: list[dict]) -> list[EconomicEvent]:
        """Bulk insert economic events.

        Args:
            events: List of dictionaries with fields matching EconomicEvent columns.

        Returns:
            List of persisted EconomicEvent instances.
        """
        event_objects = [EconomicEvent(**data) for data in events]
        self._session.add_all(event_objects)
        await self._session.flush()
        return event_objects

    async def get_upcoming_events(self, within_minutes: int = 15) -> list[EconomicEvent]:
        """Get economic events scheduled within a time window from now.

        Used for pre-event risk adjustments per Requirement 23.4:
        reduce position sizes when high-impact events are within 15 minutes.

        Args:
            within_minutes: Lookahead window in minutes (default 15).

        Returns:
            List of EconomicEvent instances scheduled within the window.
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=within_minutes)
        stmt = (
            select(EconomicEvent)
            .where(
                EconomicEvent.scheduled_at >= now,
                EconomicEvent.scheduled_at <= cutoff,
            )
            .order_by(EconomicEvent.scheduled_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_events_for_date(self, target_date: date | datetime) -> list[EconomicEvent]:
        """Get all economic events for a specific date.

        Args:
            target_date: The date to query events for. Accepts both date and datetime objects.

        Returns:
            List of EconomicEvent instances scheduled on that date.
        """
        if isinstance(target_date, datetime):
            day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            day_start = datetime(
                target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
            )
        day_end = day_start + timedelta(days=1)
        stmt = (
            select(EconomicEvent)
            .where(
                EconomicEvent.scheduled_at >= day_start,
                EconomicEvent.scheduled_at < day_end,
            )
            .order_by(EconomicEvent.scheduled_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_event_actual(
        self, event_id: str | UUID, actual_value: float | str
    ) -> EconomicEvent | None:
        """Update the actual value of an economic event after release.

        Args:
            event_id: UUID or UUID string of the economic event to update.
            actual_value: The actual released value for the event (float or string).

        Returns:
            The updated EconomicEvent, or None if not found.
        """
        if isinstance(event_id, str):
            event_id = UUID(event_id)
        stmt = select(EconomicEvent).where(EconomicEvent.id == event_id)
        result = await self._session.execute(stmt)
        event = result.scalar_one_or_none()
        if event is None:
            return None
        event.actual_value = str(actual_value) if isinstance(actual_value, (int, float)) else actual_value
        await self._session.flush()
        return event

    # ─── Geopolitical Risk ────────────────────────────────────────────────────

    async def upsert_score(
        self, region: str, score: float, indicators: dict
    ) -> GeopoliticalRiskScore:
        """Upsert a geopolitical risk score for a region.

        If a score already exists for the region, updates it in place.
        Otherwise creates a new record.

        Args:
            region: Geographic region identifier.
            score: Risk score from 0 (no risk) to 100 (extreme risk).
            indicators: Dictionary of risk indicator data.

        Returns:
            The created or updated GeopoliticalRiskScore instance.
        """
        stmt = (
            select(GeopoliticalRiskScore)
            .where(GeopoliticalRiskScore.region == region)
            .order_by(GeopoliticalRiskScore.updated_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)

        if existing is not None:
            existing.score = score
            existing.indicators_json = indicators
            existing.updated_at = now
            await self._session.flush()
            return existing

        risk_score = GeopoliticalRiskScore(
            region=region,
            score=score,
            indicators_json=indicators,
            updated_at=now,
        )
        self._session.add(risk_score)
        await self._session.flush()
        return risk_score

    # Keep backward-compatible alias
    async def update_risk_score(
        self, region: str, score: float, indicators: dict
    ) -> GeopoliticalRiskScore:
        """Alias for upsert_score (backward compatibility)."""
        return await self.upsert_score(region, score, indicators)

    async def get_all_scores(self) -> list[GeopoliticalRiskScore]:
        """Get the most recent risk score for all regions.

        Returns:
            List of GeopoliticalRiskScore instances (latest per region).
        """
        from sqlalchemy import func as sa_func

        subq = (
            select(
                GeopoliticalRiskScore.region,
                sa_func.max(GeopoliticalRiskScore.updated_at).label("max_updated"),
            )
            .group_by(GeopoliticalRiskScore.region)
            .subquery()
        )
        stmt = select(GeopoliticalRiskScore).join(
            subq,
            (GeopoliticalRiskScore.region == subq.c.region)
            & (GeopoliticalRiskScore.updated_at == subq.c.max_updated),
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # Keep backward-compatible alias
    async def get_all_risk_scores(self) -> list[GeopoliticalRiskScore]:
        """Alias for get_all_scores (backward compatibility)."""
        return await self.get_all_scores()

    async def get_high_risk_regions(
        self, threshold: float = 70.0
    ) -> list[GeopoliticalRiskScore]:
        """Get regions with risk scores above the threshold.

        Per Requirement 23.15/23.16: regions with score > 70 trigger
        exposure reduction to 50% of standard limits.

        Args:
            threshold: Minimum risk score to include (default 70.0).

        Returns:
            List of GeopoliticalRiskScore instances above the threshold.
        """
        from sqlalchemy import func as sa_func

        subq = (
            select(
                GeopoliticalRiskScore.region,
                sa_func.max(GeopoliticalRiskScore.updated_at).label("max_updated"),
            )
            .group_by(GeopoliticalRiskScore.region)
            .subquery()
        )
        stmt = (
            select(GeopoliticalRiskScore)
            .join(
                subq,
                (GeopoliticalRiskScore.region == subq.c.region)
                & (GeopoliticalRiskScore.updated_at == subq.c.max_updated),
            )
            .where(GeopoliticalRiskScore.score >= threshold)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_score_for_region(self, region: str) -> GeopoliticalRiskScore | None:
        """Get the most recent risk score for a single region.

        Args:
            region: Geographic region identifier.

        Returns:
            The latest GeopoliticalRiskScore for the region, or None if not found.
        """
        stmt = (
            select(GeopoliticalRiskScore)
            .where(GeopoliticalRiskScore.region == region)
            .order_by(GeopoliticalRiskScore.updated_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # Aliases matching the task specification interface
    async def upsert_risk_score(
        self, region: str, score: float, indicators: dict
    ) -> GeopoliticalRiskScore:
        """Upsert a geopolitical risk score for a region (spec-compliant alias).

        Args:
            region: Geographic region identifier.
            score: Risk score from 0 (no risk) to 100 (extreme risk).
            indicators: Dictionary of risk indicator data.

        Returns:
            The created or updated GeopoliticalRiskScore instance.
        """
        return await self.upsert_score(region, score, indicators)

    async def get_risk_score(self, region: str) -> GeopoliticalRiskScore | None:
        """Get the most recent risk score for a single region (spec-compliant alias).

        Args:
            region: Geographic region identifier.

        Returns:
            The latest GeopoliticalRiskScore for the region, or None if not found.
        """
        return await self.get_score_for_region(region)

    async def get_risk_score_for_region(self, region: str) -> GeopoliticalRiskScore | None:
        """Get the most recent risk score for a single region (spec interface name).

        Args:
            region: Geographic region identifier.

        Returns:
            The latest GeopoliticalRiskScore for the region, or None if not found.
        """
        return await self.get_score_for_region(region)
