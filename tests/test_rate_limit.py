from orchestro_mesh.rate_limit import SlidingWindowLimiter


def test_unlimited_when_unconfigured():
    limiter = SlidingWindowLimiter()
    for _ in range(100):
        allowed, _, _ = limiter.check("keith")
        assert allowed


def test_blocks_after_threshold():
    t = [0.0]
    limiter = SlidingWindowLimiter(per_minute={"keith": 3}, clock=lambda: t[0])
    for _ in range(3):
        allowed, _, _ = limiter.check("keith")
        assert allowed
    allowed, limit, retry_after = limiter.check("keith")
    assert allowed is False
    assert limit == 3
    assert retry_after is not None and retry_after > 0


def test_window_slides_after_60s():
    t = [0.0]
    limiter = SlidingWindowLimiter(per_minute={"keith": 2}, clock=lambda: t[0])
    assert limiter.check("keith")[0]
    assert limiter.check("keith")[0]
    assert limiter.check("keith")[0] is False
    t[0] = 61.0
    assert limiter.check("keith")[0]


def test_default_applies_when_requester_not_in_map():
    t = [0.0]
    limiter = SlidingWindowLimiter(default_per_minute=1, clock=lambda: t[0])
    assert limiter.check("alice")[0]
    assert limiter.check("alice")[0] is False


def test_zero_disables_for_that_requester():
    limiter = SlidingWindowLimiter(per_minute={"keith": 0})
    for _ in range(50):
        assert limiter.check("keith")[0]
