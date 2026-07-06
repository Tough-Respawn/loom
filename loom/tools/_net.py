"""Primitive anti-SSRF partagée : catégorisation d'une IP.

web.py et browser.py ont des politiques différentes (web bloque tout ce qui
est interne, browser autorise loopback/LAN privé pour le dev local) mais
partagent la même logique de catégorisation `ipaddress`.
"""

from __future__ import annotations

import ipaddress


def categorize_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Retourne la catégorie d'une IP : "loopback", "private", "link-local",
    "reserved", "multicast", "unspecified", ou "public"."""
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_private:
        return "private"
    if ip.is_reserved:
        return "reserved"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    return "public"
