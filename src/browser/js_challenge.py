"""§19 — JS-challenge detection for tier-1 anti-bot frameworks.

Beyond CAPTCHA: tier-1 anti-bot frameworks (Akamai Bot Manager, Kasada,
FingerprintJS Pro, Imperva Advanced Bot Protection, F5 Distributed Cloud
Bot Defense) throw JS challenges that are **NOT API-solvable** — vendors
deliberately rotate their JS fingerprinting every release and refuse
third-party solver integrations. The only correct behavior when an agent
hits one is to escalate to the operator via ``request_captcha_help`` so
the operator can intervene through the existing VNC handoff.

Detection-only — never solving. Each vendor exposes one or more well-known
DOM / cookie anchors that survive obfuscation (because legitimate
customers of the same anti-bot vendor need them stable to work). We walk
the live page in a single :meth:`page.evaluate` call (mirrors the §11.3
:func:`_classify_cf_state` pattern in :mod:`src.browser.captcha`) and
return the matched vendor name, or ``None`` when no anchor matches.

Vendor anchors (sources: vendor docs + observed live sites as of 2026-04):

================  =========================================================
Vendor            Anchors detected
================  =========================================================
``akamai``        ``<script src*="ak-bmsc">`` OR ``<script src*="bm/sc">``
                  OR ``_abck`` cookie present.
``kasada``        ``<script src*="ips.js">`` OR ``KP_UIDz`` cookie.
                  Response-header detection (``x-kpsdk-ct``) is documented
                  as a follow-up — Playwright surfaces response headers
                  via the network listener path, not :func:`page.evaluate`,
                  so a header-only Kasada page (no script, no cookie yet)
                  will pass through this classifier.
``fingerprintjs`` ``<script src*="fpjs.io">`` OR
                  ``<script src*="fingerprint.com">`` OR ``_iidt`` cookie.
``imperva``       ``<script src*="incapsula">`` OR ``_imp_apg_r_`` cookie
                  OR ``Incap_ses_*`` cookie prefix match.
``f5``            ``<script src*="f5cdn.net">`` OR ``TS01`` cookie OR
                  ``f5_cspm`` cookie.
================  =========================================================

The caller (``_check_captcha``) wraps the returned vendor name as
``f"js-challenge-{vendor}"`` for the §11.13 envelope kind.

Classifier precedence (when multiple vendors match a single page — should
be rare in practice; sites typically deploy one) is the order listed in
the JS body below: akamai → kasada → fingerprintjs → imperva → f5.
"""

from __future__ import annotations

from src.shared.utils import setup_logging

logger = setup_logging("browser.js_challenge")


# Vendor anchor walk. Single ``page.evaluate`` — keeps the cost of
# detection at one round-trip, mirrors the §11.3 ``_classify_cf_state``
# pattern, and lets us hand the test mocks one stable JS string to match
# on (see :data:`tests.test_js_challenge_detection`).
#
# Anchor logic notes:
# * Script src checks use ``includes()`` (substring match) so vendor URL
#   variants survive (CDN subpath, version suffix, region rotation).
# * Cookie checks use ``startsWith()`` for the Imperva ``Incap_ses_``
#   PREFIX and exact-name match for everything else — vendor docs spell
#   the exact cookie name for those cases.
# * The function returns the FIRST matching vendor (precedence in the
#   order listed) and short-circuits — no need to enumerate all matches.
_JS_CLASSIFY_VENDOR_JS = r"""
() => {
  try {
    // Lower-case script srcs so URL-component case (e.g. Imperva's
    // ``/_Incapsula_Resource`` capitalized path, observed across real
    // sites) matches our anchor needles. Cookie names are case-sensitive
    // by RFC 6265 §4.1.2 — keep those as-is.
    const scripts = Array.from(document.scripts)
      .map(s => (s.src || "").toLowerCase());
    const has_script = (needle) => scripts.some(s => s.indexOf(needle) !== -1);

    const cookieRaw = (document.cookie || "");
    const cookieNames = cookieRaw.split(";").map(c => {
      const t = c.trim();
      const eq = t.indexOf("=");
      return eq === -1 ? t : t.slice(0, eq);
    });
    const has_cookie = (name) => cookieNames.indexOf(name) !== -1;
    const has_cookie_prefix = (prefix) =>
      cookieNames.some(n => n.indexOf(prefix) === 0);

    // Akamai Bot Manager.
    if (
      has_script("ak-bmsc") ||
      has_script("bm/sc") ||
      has_cookie("_abck")
    ) {
      return "akamai";
    }
    // Kasada.
    if (
      has_script("ips.js") ||
      has_cookie("KP_UIDz")
    ) {
      return "kasada";
    }
    // FingerprintJS Pro.
    if (
      has_script("fpjs.io") ||
      has_script("fingerprint.com") ||
      has_cookie("_iidt")
    ) {
      return "fingerprintjs";
    }
    // Imperva Advanced Bot Protection.
    if (
      has_script("incapsula") ||
      has_cookie("_imp_apg_r_") ||
      has_cookie_prefix("Incap_ses_")
    ) {
      return "imperva";
    }
    // F5 Distributed Cloud Bot Defense.
    if (
      has_script("f5cdn.net") ||
      has_cookie("TS01") ||
      has_cookie("f5_cspm")
    ) {
      return "f5";
    }
  } catch (e) { /* defensive — never throw to the page */ }
  return null;
}
"""


# Vendor names emitted by the classifier — mirrors the JS body. Kept here
# as a frozenset so callers (tests, ``_check_captcha`` integration) can
# assert membership without re-listing the vendor names. The §11.13
# envelope ``kind`` is built as ``f"js-challenge-{vendor}"`` for each
# entry in this set.
_VENDORS: frozenset[str] = frozenset({
    "akamai",
    "kasada",
    "fingerprintjs",
    "imperva",
    "f5",
})


async def classify_js_challenge(page) -> str | None:
    """Detect tier-1 anti-bot JS challenges on the live page.

    Returns the matched vendor name (one of the values in
    :data:`_VENDORS`) or ``None`` when no documented anchor matches.

    Vendors detected (returns the corresponding kind enum value WITHOUT
    the ``js-challenge-`` prefix; the caller wraps for the envelope):

    * ``"akamai"`` — Akamai Bot Manager
    * ``"kasada"`` — Kasada
    * ``"fingerprintjs"`` — FingerprintJS Pro
    * ``"imperva"`` — Imperva Advanced Bot Protection
    * ``"f5"`` — F5 Distributed Cloud Bot Defense
    * ``None`` — no JS challenge detected; caller continues normal flow.

    Wrapped in try/except so a closed page or :func:`page.evaluate`
    failure simply collapses to ``None`` rather than raising — consistent
    with the :func:`_classify_cf_state` / :func:`_classify_behavioral`
    pattern. Never crashes ``_check_captcha``.

    Limitation: detection ≠ solving. None of these vendors expose an
    API-solvable widget (by design); the agent must escalate to the
    operator via :meth:`request_captcha_help` so the operator can
    intervene through VNC. The caller emits a behavioral-only envelope.
    """
    try:
        vendor = await page.evaluate(_JS_CLASSIFY_VENDOR_JS)
    except Exception:
        logger.debug(
            "classify_js_challenge: page.evaluate failed", exc_info=True,
        )
        return None
    if not isinstance(vendor, str):
        return None
    if vendor not in _VENDORS:
        # Defensive — the JS body only returns members of ``_VENDORS`` or
        # ``null``, but a future edit that adds a vendor to the JS without
        # updating ``_VENDORS`` should fail safe rather than emit a
        # phantom kind.
        return None
    return vendor
