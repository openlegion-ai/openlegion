#!/bin/bash
# Browser container entrypoint — install iptables egress filter, drop privileges, exec.
#
# The browser container runs Camoufox (stealth Firefox) under Playwright, directed
# by agents. Without a network-layer filter, any browser-initiated request —
# redirects, subresource loads (img/script/iframe), XHR, fetch, WebSockets — could
# target private IPs on the Docker network, on the host, or at cloud metadata
# endpoints (169.254.169.254). Mesh-side URL validation only guards the initial
# navigate action; everything else bypasses it and cannot be hooked via the
# Playwright/Firefox API.
#
# This script installs iptables rules that REJECT outbound packets to private /
# reserved IP ranges at the kernel. Firefox sees a clean connection error; the
# filter is invisible to websites (no proxy, no TLS termination), so Camoufox
# anti-bot fingerprints are entirely unaffected.
#
# Escape hatches:
#   BROWSER_EGRESS_DISABLE=1            Skip the egress filter entirely.
#   BROWSER_EGRESS_ALLOWLIST=cidr,...   Extra ACCEPT rules (e.g. a private proxy).

set -e

log() { echo "[egress-filter] $*" >&2; }

# Validate that a string is a strict IPv4 literal (dotted quad, each octet 0-255).
is_valid_ipv4() {
  local s="$1" a b c d extra
  case "$s" in *[!0-9.]*|'') return 1;; esac
  local ifs_save="$IFS"
  IFS=.
  # shellcheck disable=SC2086
  set -- $s
  IFS="$ifs_save"
  [ "$#" -eq 4 ] || return 1
  for octet in "$@"; do
    case "$octet" in
      ''|*[!0-9]*) return 1;;
    esac
    # Strip leading zeros to avoid octal interpretation in arithmetic.
    octet=$((10#$octet))
    [ "$octet" -ge 0 ] && [ "$octet" -le 255 ] || return 1
  done
  return 0
}

# Validate that a string is either a strict IPv4 literal or IPv4/CIDR (prefix 0-32).
is_valid_ipv4_cidr() {
  local s="$1" ip prefix
  case "$s" in *[!0-9./]*|'') return 1;; esac
  case "$s" in
    */*) ip="${s%/*}"; prefix="${s#*/}";;
    *)   ip="$s";       prefix="32";;
  esac
  is_valid_ipv4 "$ip" || return 1
  case "$prefix" in ''|*[!0-9]*) return 1;; esac
  prefix=$((10#$prefix))
  [ "$prefix" -ge 0 ] && [ "$prefix" -le 32 ] || return 1
  return 0
}

install_egress_filter() {
  if [ "${BROWSER_EGRESS_DISABLE:-}" = "1" ]; then
    log "BROWSER_EGRESS_DISABLE=1 — skipping firewall setup (browser has unrestricted network access)"
    return 0
  fi

  if ! command -v iptables-restore >/dev/null 2>&1; then
    log "iptables-restore not installed — skipping firewall setup"
    return 0
  fi

  # Sanity check: can we actually read the OUTPUT chain? If NET_ADMIN is absent
  # (dev setups, restricted runtimes) bail out loudly instead of failing boot.
  if ! iptables -L OUTPUT -n >/dev/null 2>&1; then
    log "WARNING: cannot access iptables (missing NET_ADMIN?). Egress filter NOT installed."
    return 0
  fi

  # Build ACCEPT rules for configured nameservers so DNS keeps working. Docker's
  # embedded resolver at 127.0.0.11 is reached over lo (allowed separately).
  # Strict validation: reject anything that isn't a dotted-quad v4 literal.
  local dns_rules=""
  local ns
  for ns in $(awk '/^nameserver[ \t]/ { print $2 }' /etc/resolv.conf 2>/dev/null); do
    case "$ns" in *:*) continue ;; esac  # v4 table: skip v6 nameservers
    if ! is_valid_ipv4 "$ns"; then
      continue
    fi
    dns_rules="${dns_rules}-A OUTPUT -d ${ns}/32 -p udp --dport 53 -j ACCEPT
"
    dns_rules="${dns_rules}-A OUTPUT -d ${ns}/32 -p tcp --dport 53 -j ACCEPT
"
  done

  # Operator allowlist (comma-separated CIDRs). Useful for private-network
  # proxies. Strict validation: reject anything that isn't an IPv4 / CIDR
  # literal, and explicitly reject 0.0.0.0/0 (would defeat the filter).
  local allow_rules=""
  local cidr
  local old_ifs="$IFS"
  IFS=','
  for cidr in ${BROWSER_EGRESS_ALLOWLIST:-}; do
    # Strip leading and trailing whitespace (handles multiple spaces/tabs).
    cidr="${cidr#"${cidr%%[![:space:]]*}"}"
    cidr="${cidr%"${cidr##*[![:space:]]}"}"
    [ -z "$cidr" ] && continue
    if ! is_valid_ipv4_cidr "$cidr"; then
      log "WARNING: ignoring invalid allowlist entry: ${cidr}"
      continue
    fi
    if [ "$cidr" = "0.0.0.0/0" ]; then
      log "WARNING: refusing to allowlist 0.0.0.0/0 — would defeat the egress filter"
      continue
    fi
    allow_rules="${allow_rules}-A OUTPUT -d ${cidr} -j ACCEPT
"
  done
  IFS="$old_ifs"

  if ! iptables-restore <<EOF
*filter
:INPUT ACCEPT [0:0]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [0:0]

# Loopback is allowed: Firefox can reach in-container services on 127.0.0.1
# (KasmVNC on 6080, FastAPI on 8500). /browser/* requires bearer auth; the
# /uploads/* endpoint is intentionally unauthenticated so the browser can
# navigate to user-uploaded files. Narrowing this rule would require per-port
# allowlists that must be kept in sync with service.py.
-A OUTPUT -o lo -j ACCEPT
-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
${dns_rules}${allow_rules}
-A OUTPUT -d 0.0.0.0/8       -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 10.0.0.0/8      -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 100.64.0.0/10   -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 127.0.0.0/8     -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 169.254.0.0/16  -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 172.16.0.0/12   -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 192.0.0.0/24    -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 192.0.2.0/24    -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 192.168.0.0/16  -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 198.18.0.0/15   -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 198.51.100.0/24 -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 203.0.113.0/24  -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 224.0.0.0/4     -j REJECT --reject-with icmp-net-unreachable
-A OUTPUT -d 240.0.0.0/4     -j REJECT --reject-with icmp-net-unreachable

COMMIT
EOF
  then
    log "ERROR: iptables-restore failed to install the egress filter."
    log "  Likely causes: kernel lacks nf_conntrack, or CAP_NET_ADMIN was"
    log "  not granted at rule-install time (check cap_add in runtime.py)."
    log "  Set BROWSER_EGRESS_DISABLE=1 to start without the filter (insecure)."
    exit 1
  fi

  # IPv6 filtering. Kernel may lack IPv6 entirely; treat errors as non-fatal
  # because IPv6 being off is a valid deployment state. If the kernel HAS
  # IPv6 but the rule install fails, that is a hard error — we do not want
  # to silently leave the v6 egress path wide open.
  if command -v ip6tables-restore >/dev/null 2>&1 && ip6tables -L OUTPUT -n >/dev/null 2>&1; then
    if ! ip6tables-restore <<'EOF'
*filter
:INPUT ACCEPT [0:0]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [0:0]

# Loopback is allowed — see IPv4 table comment above for rationale.
-A OUTPUT -o lo -j ACCEPT
-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

-A OUTPUT -d ::/128          -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d ::1/128         -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d ::ffff:0:0/96   -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d 64:ff9b::/96    -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d 100::/64        -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d 2001::/32       -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d 2001:db8::/32   -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d 2002::/16       -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d fc00::/7        -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d fe80::/10       -j REJECT --reject-with icmp6-adm-prohibited
-A OUTPUT -d ff00::/8        -j REJECT --reject-with icmp6-adm-prohibited

COMMIT
EOF
    then
      log "ERROR: ip6tables-restore failed — IPv6 is enabled but rules could not be installed."
      log "  Likely causes: kernel lacks nf_conntrack IPv6 module, or CAP_NET_ADMIN"
      log "  was not granted at rule-install time (check cap_add in runtime.py)."
      log "  Refusing to continue with an unprotected IPv6 egress path."
      exit 1
    fi
  fi

  log "iptables egress filter installed (private IP ranges blocked)"
}

# When this script is executed directly (as Docker ENTRYPOINT), install the
# egress filter then exec into tini→gosu→python. When sourced (e.g. from a
# unit test that wants to call is_valid_ipv4 / is_valid_ipv4_cidr directly),
# do nothing at load time so the tests can invoke the helpers without side
# effects. ${BASH_SOURCE[0]} differs from ${0} when sourced.
if [ "${BASH_SOURCE[0]:-$0}" = "${0}" ]; then
  install_egress_filter
  # tini must be PID 1 to reap zombies. exec into tini, which then runs gosu
  # to drop to the non-root browser user before launching the FastAPI service.
  exec tini -- gosu browser:browser python -m src.browser
fi
