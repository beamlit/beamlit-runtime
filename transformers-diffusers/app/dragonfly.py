import requests
from requests.adapters import HTTPAdapter
from typing import Optional

DRAGONFLY_URL: Optional[str] = (
    "http://dragonfly-seed-client.dragonfly-system.svc.cluster.local:4001"
)


def set_dragonfly_url(url: str):
    """Set the Dragonfly URL."""
    global DRAGONFLY_URL
    DRAGONFLY_URL = url


class DragonflyAdapter(HTTPAdapter):
    def get_connection_with_tls_context(
        self,
        request: requests.Request,
        verify: bool,
        proxies: Optional[dict],
        cert: Optional[str],
    ) -> requests.Session:
        if (
            request.url.startswith("https://cdn-lfs.hf.co")
            or request.url.startswith("https://cdn-lfs-eu-1.hf.co")
            or request.url.startswith("https://cdn-lfs-us-1.hf.co")
        ):
            request.url = request.url.replace("https://", "http://")
        return super().get_connection_with_tls_context(request, verify, proxies, cert)


# Create a factory function that returns a new Session.
def backend_factory() -> requests.Session:
    session = requests.Session()
    if DRAGONFLY_URL is not None:
        session.mount("http://", DragonflyAdapter())
        session.mount("https://", DragonflyAdapter())
        session.proxies = {
            "https": DRAGONFLY_URL,
            "http": DRAGONFLY_URL,
        }
        session.verify = False
    return session
