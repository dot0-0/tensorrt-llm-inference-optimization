import random, statistics

def simulate(arrival_rate_qps=50, max_delay_ms=20, batch_limit=8, service_tps=1500, req_tokens=64, duration_s=10):
    """Poisson-ish arrivals, simple batcher with max_delay_ms & batch_limit.
    service_tps: tokens/sec the backend can do for a single stream; batch scales idealized linearly here."""
    now = 0.0
    arrivals = []
    while now < duration_s:
        # exponential inter-arrival
        gap = random.expovariate(arrival_rate_qps)
        now += gap
        if now <= duration_s:
            arrivals.append(now)
    # batching
    i, latencies = 0, []
    queue = []
    t = 0.0
    while i < len(arrivals) or queue:
        if i < len(arrivals) and arrivals[i] <= t:
            queue.append(arrivals[i]); i += 1; continue
        if not queue and i < len(arrivals):
            t = arrivals[i]; continue
        # wait up to max_delay_ms to collect more
        deadline = t + max_delay_ms/1000.0
        while i < len(arrivals) and arrivals[i] <= deadline and len(queue) < batch_limit:
            queue.append(arrivals[i]); i += 1
        bsz = min(len(queue), batch_limit)
        if bsz == 0:
            t = deadline; continue
        # service time: req_tokens / (service_tps * bsz) (idealized)
        svc = req_tokens / (service_tps * bsz)
        start = t
        finish = start + svc
        for _ in range(bsz):
            a = queue.pop(0)
            latencies.append(finish - a)
        t = finish
    return {
        "count": len(latencies),
        "p50_s": statistics.quantiles(latencies, n=100)[49],
        "p90_s": statistics.quantiles(latencies, n=100)[89],
        "avg_s": sum(latencies)/len(latencies) if latencies else 0.0
    }

