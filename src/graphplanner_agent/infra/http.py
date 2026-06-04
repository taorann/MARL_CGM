from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse
import urllib.request


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def should_bypass_proxy(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    if host in LOCAL_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def urlopen_no_proxy_for_localhost(req: urllib.request.Request, timeout: int):
    if _bypass_all_proxies() or should_bypass_proxy(req.full_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _bypass_all_proxies() -> bool:
    return str(os.environ.get("GRAPHPLANNER_BYPASS_HTTP_PROXY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
