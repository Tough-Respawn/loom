"""Primitive anti-SSRF partagée : catégorisation d'une IP.

web.py et browser.py ont des politiques différentes (web bloque tout ce qui
est interne, browser autorise loopback/LAN privé pour le dev local) mais
partagent la même logique de catégorisation `ipaddress`.
"""

from __future__ import annotations

import ipaddress

# Préfixe NAT64 « well-known » (RFC 6052) : sur un réseau IPv6-only avec DNS64,
# TOUT site IPv4-only se résout en 64:ff9b::<ipv4> — github.com compris. Python
# classe ce préfixe is_reserved=True (table RFC 2373) alors qu'il est routable :
# sans traitement, l'anti-SSRF bloquait la moitié d'Internet sur ces réseaux
# (vécu 2026-08-02). La bonne question de sécurité porte sur l'IPv4 EMBARQUÉE
# (32 bits bas) : 64:ff9b::7f00:1 = 127.0.0.1 -> loopback, toujours bloqué.
_NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")


def categorize_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Retourne la catégorie d'une IP : "loopback", "private", "link-local",
    "reserved", "multicast", "unspecified", ou "public"."""
    if isinstance(ip, ipaddress.IPv6Address) and ip in _NAT64_WKP:
        return categorize_ip(ipaddress.IPv4Address(int(ip) & 0xFFFF_FFFF))
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
