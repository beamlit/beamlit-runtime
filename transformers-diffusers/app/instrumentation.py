import inference_server
from prometheus_client import Counter, Histogram
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import set_tracer_provider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging


INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total", "Total number of inference requests"
)
INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total", "Total number of inference errors"
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Inference request latency in seconds"
)


def setup_instrumentation(logger: logging.Logger):
    """Setup instrumentation for Prometheus and OpenTelemetry"""
    resource = Resource.create()

    # Set up the TracerProvider
    provider = TracerProvider(resource=resource)
    set_tracer_provider(provider)

    # Set up OTLP Exporter using environment variables
    span_exporter = OTLPSpanExporter()  # No arguments needed, uses env vars
    span_processor = BatchSpanProcessor(span_exporter)
    provider.add_span_processor(span_processor)
    set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app=inference_server.app, tracer_provider=provider
    )
    logger.info("Instrumentation setup complete")
