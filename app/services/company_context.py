"""Read a company's public web page, safely, so a personal line can be grounded in real company text.

The address is typed by a user, so it is treated as hostile. Before anything is fetched:
  * the scheme must be https (an http:// address is upgraded; nothing is ever sent unencrypted) and the port 443;
  * no credentials in the address, no IP-address hosts, no single-label or internal-looking names ("localhost", ".local");
  * the name is resolved here and EVERY address must be public (no private, loopback, link-local, metadata,
    carrier-grade NAT, multicast or reserved ranges, including IPv4-mapped IPv6);
  * the connection is then made to the address that was checked (not resolved a second time), with the
    certificate verified for the original name, so DNS cannot change its answer between the check and the fetch.
Redirects are followed by hand, at most three, and each one passes every check above and must stay on the same site.
The response is limited in time, in bytes and in type (HTML or plain text only, never compressed), and the text taken
from it is untrusted data: scripts and styles are dropped, instruction-like lines are removed, and the result is
used only as evidence, never as instructions.

Nothing here raises to the caller: `get_company_context` always returns a result that says whether it worked and why not.
"""

from __future__ import annotations

import http.client
import ipaddress
import logging
import re
import socket
import ssl
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

MAX_BYTES = 300_000          # read at most this much of a page
MAX_TEXT_CHARS = 2500        # keep at most this much text from it
CONNECT_TIMEOUT = 6.0
TOTAL_TIMEOUT = 12.0         # for the whole fetch, redirects included
MAX_REDIRECTS = 3
CACHE_SECONDS = 600
CACHE_ENTRIES = 64
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
USER_AGENT = "RoleSync-company-context/1.0 (reads one public page; no login)"
_INTERNAL_SUFFIXES = (".local", ".localhost", ".internal", ".intranet", ".lan", ".home", ".corp", ".localdomain", ".home.arpa", ".test", ".invalid", ".example")


# IPv6 ranges that carry an IPv4 address inside (NAT64, 6to4, Teredo): a private IPv4 address could hide in them
_IPV6_TRANSITION = [ipaddress.ip_network(n) for n in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")]


class UnsafeUrl(ValueError):
    """The address (or a redirect target) may not be fetched. The message is safe to show the user."""


@dataclass
class CompanyContext:
    text: str = ""                 # untrusted page text, cleaned (empty unless status == "ok")
    url: str | None = None         # the page the text came from
    status: str = "none"           # "ok", "none" (no website given) or "failed"
    reason: str | None = None      # why it failed, in plain words


# ---------------------------------------------------------------------------------------------------------------------
# the address

def normalize_url(raw: str) -> str:
    """The https address to fetch for what the user typed, or UnsafeUrl."""
    text = (raw or "").strip()
    if not text:
        raise UnsafeUrl("No website was given.")
    if len(text) > 300 or re.search(r"[\s\x00-\x1f]", text):
        raise UnsafeUrl("That does not look like a website address.")
    if "://" not in text:
        text = "https://" + text
    elif text.lower().startswith("http://"):
        text = "https://" + text[len("http://"):]               # never fetch over plain http: use https
    parts = urlsplit(text)
    if parts.scheme.lower() != "https":
        raise UnsafeUrl("Only https websites can be read.")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UnsafeUrl("Addresses with a username or password are not allowed.")
    try:
        port = parts.port
    except ValueError:
        raise UnsafeUrl("That does not look like a website address.") from None
    if port not in (None, 443):
        raise UnsafeUrl("Only the standard https port can be used.")
    host = (parts.hostname or "").strip(".").lower()
    if not host:
        raise UnsafeUrl("That does not look like a website address.")
    try:
        ipaddress.ip_address(host)
        raise UnsafeUrl("Websites given as a numeric address are not allowed. Use the company's domain name.")
    except ValueError as exc:
        if isinstance(exc, UnsafeUrl):
            raise
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise UnsafeUrl("That does not look like a website address.") from None
    labels = host.split(".")
    if len(labels) < 2 or len(host) > 253 or host.endswith(_INTERNAL_SUFFIXES) or host == "localhost":
        raise UnsafeUrl("That is not a public website name.")
    if not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels):
        raise UnsafeUrl("That does not look like a website address.")
    if not (re.fullmatch(r"[a-z]{2,63}", labels[-1]) or labels[-1].startswith("xn--")):
        raise UnsafeUrl("That does not look like a website address.")
    path = parts.path or "/"
    if len(path) > 200:
        path = "/"
    return f"https://{host}{path}"                              # query and fragment are dropped on purpose


def _same_site(a: str, b: str) -> bool:
    """A redirect may only stay on the same site: the same host, its www. twin, or a subdomain of it."""
    strip = lambda h: h[4:] if h.startswith("www.") else h
    ha, hb = strip(urlsplit(a).hostname or ""), strip(urlsplit(b).hostname or "")
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha)


def _public_addresses(host: str) -> list[str]:
    """Every address `host` resolves to, if all of them are public; UnsafeUrl otherwise."""
    try:
        info = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise UnsafeUrl("That website could not be found.") from None
    addresses: list[str] = []
    for entry in info:
        raw = entry[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise UnsafeUrl("That website could not be found.") from None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global is False for private, loopback, link-local (incl. cloud metadata), CGNAT, multicast and reserved ranges
        if isinstance(ip, ipaddress.IPv6Address) and any(ip in net for net in _IPV6_TRANSITION):
            raise UnsafeUrl("That website points to a private or reserved address, so it cannot be read.")
        if not ip.is_global or ip.is_multicast:
            raise UnsafeUrl("That website points to a private or reserved address, so it cannot be read.")
        if str(ip) not in addresses:
            addresses.append(str(ip))
    if not addresses:
        raise UnsafeUrl("That website could not be found.")
    return sorted(addresses, key=lambda a: ":" in a)              # IPv4 first


# ---------------------------------------------------------------------------------------------------------------------
# the connection

class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects to an address that was already checked, but verifies the certificate for the original host name."""

    def __init__(self, host: str, ip: str, timeout: float, context: ssl.SSLContext):
        super().__init__(host, 443, timeout=timeout, context=context)
        self._pinned_ip = ip

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _fetch_once(url: str, deadline: float) -> tuple[int, dict[str, str], bytes]:
    """One GET of `url` (already normalized). Returns status, lower-cased headers and at most MAX_BYTES of body."""
    parts = urlsplit(url)
    host, path = parts.hostname or "", parts.path or "/"
    addresses = _public_addresses(host)
    context = ssl.create_default_context()                        # verifies the certificate chain and the host name
    last_error: Exception | None = None
    for ip in addresses[:3]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("took too long")
        conn = _PinnedHTTPSConnection(host, ip, min(CONNECT_TIMEOUT, remaining), context)
        try:
            conn.request("GET", path, headers={"Host": host, "User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.8",
                                               "Accept-Encoding": "identity", "Connection": "close"})
            response = conn.getresponse()
            headers = {k.lower(): v for k, v in response.getheaders()}
            if 300 <= response.status < 400:
                return response.status, headers, b""
            body = b""
            while len(body) < MAX_BYTES:
                if time.monotonic() > deadline:
                    raise TimeoutError("took too long")
                chunk = response.read(min(16_384, MAX_BYTES - len(body)))
                if not chunk:
                    break
                body += chunk
            return response.status, headers, body
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            last_error = exc
        finally:
            conn.close()
    raise last_error or OSError("could not connect")


# ---------------------------------------------------------------------------------------------------------------------
# the text

_SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "form", "iframe", "template"}
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "blockquote", "div", "section", "article", "td"}
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)|disregard|forget\s+(?:all|everything|your)|system\s+prompt|"
    r"\byou\s+are\s+(?:now\s+)?(?:an?\s+)?(?:ai|assistant|chatbot|language\s+model)|\bas\s+an\s+ai\b|new\s+instructions?|"
    r"\b(?:assistant|system|user)\s*:|</?\s*(?:system|assistant|instructions?)\b|reveal\s+(?:your|the)\s+(?:prompt|instructions)",
    re.IGNORECASE,
)


class _TextExtractor(HTMLParser):
    """Title, descriptions, headings and paragraphs: what a page says about itself. Never executes anything."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth_skip = 0
        self.title = ""
        self.meta: list[str] = []
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.depth_skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            if a.get("name", "").lower() in {"description", "twitter:description"} or a.get("property", "").lower() in {"og:description", "og:title"}:
                self.meta.append(a.get("content", ""))
        if tag in _BLOCK or tag == "br":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self.depth_skip:
            self.depth_skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self.depth_skip:
            self._buf.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if len(text) >= 25:
            self.lines.append(text)

    def close(self) -> None:
        super().close()
        self._flush()


def clean_untrusted_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """Page text made safe to use as evidence: control characters and instruction-like lines are removed, repeats
    collapsed, and the length capped. It remains DATA: callers must never present it to a model as instructions."""
    seen: set[str] = set()
    out: list[str] = []
    total = 0
    for raw in re.split(r"[\r\n]+", text or ""):
        line = " ".join(re.sub(r"[\x00-\x1f\x7f​-‏‪-‮⁦-⁩]", " ", raw).split())
        key = line.lower()
        if len(line) < 20 or key in seen or _INJECTION.search(line) or re.search(r"https?://|www\.|\S+@\S+\.\S+", line):
            continue
        seen.add(key)
        if total + len(line) > limit:
            line = line[: max(0, limit - total)].rsplit(" ", 1)[0]
        if line:
            out.append(line)
            total += len(line) + 1
        if total >= limit:
            break
    return "\n".join(out)


def extract_text(body: bytes, content_type: str = "") -> str:
    match = re.search(r"charset=([\w-]+)", content_type or "", re.IGNORECASE)
    try:
        html = body.decode(match.group(1) if match else "utf-8", errors="replace")
    except LookupError:
        html = body.decode("utf-8", errors="replace")
    if "html" not in (content_type or "").lower() and "<" not in html[:2000]:
        return clean_untrusted_text(html)
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:                                                # broken markup must never break the caller
        logger.debug("page markup could not be fully parsed", exc_info=True)
    pieces = [parser.title.strip(), *[m.strip() for m in parser.meta], *parser.lines]
    return clean_untrusted_text("\n".join(p for p in pieces if p))


# ---------------------------------------------------------------------------------------------------------------------
# the public function

_CACHE: dict[str, tuple[float, CompanyContext]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def fetch_page(raw_url: str) -> CompanyContext:
    """Fetch and read one public page. Returns a CompanyContext; failures carry a plain-words reason."""
    try:
        url = normalize_url(raw_url)
    except UnsafeUrl as exc:
        return CompanyContext(status="failed", reason=str(exc))
    deadline = time.monotonic() + TOTAL_TIMEOUT
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            status, headers, body = _fetch_once(url, deadline)
            if 300 <= status < 400:
                location = headers.get("location", "")
                if not location:
                    return CompanyContext(status="failed", reason="The website redirected without saying where.")
                target = normalize_url(urljoin(url, location))     # every hop passes every check again
                if not _same_site(url, target):
                    return CompanyContext(status="failed", reason="The website redirects to a different site, so it was not read.")
                url = target
                continue
            if status != 200:
                return CompanyContext(status="failed", reason=f"The website answered with an error ({status}).")
            kind = headers.get("content-type", "").split(";")[0].strip().lower()
            if kind not in ALLOWED_TYPES:
                return CompanyContext(status="failed", reason="The website did not return a readable page.")
            if headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                return CompanyContext(status="failed", reason="The website returned a compressed page, which is not read.")
            text = extract_text(body, headers.get("content-type", ""))
            if len(text) < 60:
                return CompanyContext(url=url, status="failed", reason="The page has too little readable text (it may need JavaScript).")
            return CompanyContext(text=text, url=url, status="ok")
        return CompanyContext(status="failed", reason="The website redirected too many times.")
    except UnsafeUrl as exc:
        return CompanyContext(status="failed", reason=str(exc))
    except TimeoutError:
        return CompanyContext(status="failed", reason="The website took too long to answer.")
    except ssl.SSLError:
        return CompanyContext(status="failed", reason="The website's security certificate could not be verified.")
    except Exception as exc:                                         # anything else: report, never raise
        logger.info("company website could not be read: %s", type(exc).__name__)
        return CompanyContext(status="failed", reason="The website could not be reached.")


def get_company_context(website: str | None) -> CompanyContext:
    """The cleaned text of the company's website, or a result saying why there is none. Cached for ten minutes."""
    if not (website or "").strip():
        return CompanyContext(status="none")
    try:
        key = normalize_url(website)
    except UnsafeUrl as exc:
        return CompanyContext(status="failed", reason=str(exc))
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    result = fetch_page(key)
    if len(_CACHE) >= CACHE_ENTRIES:
        _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]))
    _CACHE[key] = (now, result)
    return result
