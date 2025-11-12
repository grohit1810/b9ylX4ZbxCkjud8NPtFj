"""
Prometheus metrics instrumentation for Movie Recommender Chatbot.

Provides HTTP request metrics, error tracking, and service health monitoring
using prometheus-fastapi-instrumentator v7.1.0.
"""

from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info
from typing import Callable
import time
from logging_config.logger import get_logger

logger = get_logger(__name__)

# Define custom metric prefix
METRIC_NAMESPACE = "movie"
METRIC_SUBSYSTEM = "chatbot"

# ========== Custom Prometheus Metrics ==========

# HTTP Request Error Tracking
http_request_errors = Counter(
    "http_errors_total",
    "Total count of HTTP errors by endpoint and type",
    labelnames=["method", "handler", "status_code", "error_type"],
    namespace=METRIC_NAMESPACE,
    subsystem=METRIC_SUBSYSTEM
)

# Service Health Gauges
service_health_database = Gauge(
    "service_health_database",
    "Health status of database (1=healthy, 0=unhealthy)",
    namespace=METRIC_NAMESPACE,
    subsystem=METRIC_SUBSYSTEM
)

service_health_cache = Gauge(
    "service_health_cache",
    "Health status of Redis cache (1=healthy, 0=unhealthy)",
    namespace=METRIC_NAMESPACE,
    subsystem=METRIC_SUBSYSTEM
)

service_health_agent = Gauge(
    "service_health_agent",
    "Health status of LLM agent (1=healthy, 0=unhealthy)",
    namespace=METRIC_NAMESPACE,
    subsystem=METRIC_SUBSYSTEM
)


# ========== Custom Instrumentation Functions ==========

def http_error_counter() -> Callable[[Info], None]:
    """
    Custom instrumentation function to track HTTP errors.
    
    This function is called for every request and increments error counters
    when status codes indicate errors (4xx, 5xx).
    
    Returns:
        Callable: Instrumentation function compatible with Instrumentator
    """
    def instrumentation(info: Info) -> None:
        """Track errors based on response status code."""
        # Only track if we have a response and it's an error
        if info.response and hasattr(info.response, "status_code"):
            status_code = info.response.status_code
            
            if status_code >= 400:
                error_type = classify_error(status_code)
                http_request_errors.labels(
                    method=info.method,
                    handler=info.modified_handler,
                    status_code=str(status_code),
                    error_type=error_type
                ).inc()
                
                logger.debug(
                    f"Error tracked: {info.method} {info.modified_handler} "
                    f"- {status_code} - {error_type}"
                )
    
    return instrumentation


# ========== Instrumentator Setup ==========

def setup_metrics(app) -> Instrumentator:
    """
    Setup Prometheus metrics instrumentation for FastAPI app.
    
    Configures automatic HTTP metrics collection including:
    - Request count by method, endpoint, and status
    - Request latency histograms (with and without streaming)
    - Request size and response size
    - Requests in progress
    - Custom error tracking
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Instrumentator: Configured instrumentator instance
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,  # Group 2xx, 4xx, 5xx
        should_ignore_untemplated=False,
        should_respect_env_var=False,  # Always enable metrics
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],  # Don't track metrics endpoint
        inprogress_name="requests_inprogress",
        inprogress_labels=True,
    )
    
    # Add default metrics with namespace and subsystem
    # This adds: request count, latency, size metrics
    instrumentator.add(
        metrics.default(
            metric_namespace=METRIC_NAMESPACE,
            metric_subsystem=METRIC_SUBSYSTEM,
            should_only_respect_2xx_for_highr=False,
        )
    )
    
    # Add custom error tracking metric
    instrumentator.add(http_error_counter())
    
    # Instrument the app
    instrumentator.instrument(
        app,
        metric_namespace=METRIC_NAMESPACE,
        metric_subsystem=METRIC_SUBSYSTEM,
    )
    
    # Expose metrics endpoint
    instrumentator.expose(
        app,
        endpoint="/metrics",
        include_in_schema=True,
        should_gzip=True,  # Compress metrics response
    )
    
    logger.info(
        f"Prometheus metrics instrumentation enabled at /metrics "
        f"(namespace: {METRIC_NAMESPACE}, subsystem: {METRIC_SUBSYSTEM})"
    )
    
    return instrumentator


# ========== Helper Functions ==========

def classify_error(status_code: int) -> str:
    """
    Classify HTTP status code into error type.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        str: Error type classification
    """
    if 400 <= status_code < 500:
        error_map = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            422: "validation_error",
        }
        return error_map.get(status_code, "client_error")
    elif 500 <= status_code < 600:
        return "server_error"
    else:
        return "unknown"


def update_service_health(component: str, is_healthy: bool):
    """
    Update health status of service component.
    
    Args:
        component: Name of component (database, cache, agent)
        is_healthy: True if healthy, False otherwise
    """
    health_value = 1.0 if is_healthy else 0.0
    
    if component == "database":
        service_health_database.set(health_value)
    elif component == "cache":
        service_health_cache.set(health_value)
    elif component == "agent":
        service_health_agent.set(health_value)
    else:
        logger.warning(f"Unknown component for health tracking: {component}")
    
    logger.debug(
        f"Updated health for {component}: "
        f"{'healthy' if is_healthy else 'unhealthy'}"
    )
